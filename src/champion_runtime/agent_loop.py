import copy
import datetime
import json
import os
import re
import time
from typing import Optional

import json5
import tiktoken
from openai import AsyncOpenAI

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .offline_search import batch_search
from .observability import NullObserver


DASHSCOPE_API_KEY = os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.5-plus")
MAX_ROUNDS = 100
TIMEOUT_SECONDS = 540  # 9 minutes (leave 1 min buffer for 10 min limit)
MAX_TOKENS_ESTIMATE = 500000  # Upgraded from 80K - qwen-plus supports 128K, leave buffer
MAX_TOOL_RESULT_CHARS = 15000  # Max chars per tool result to keep context manageable

_POLICY_BACKEND = None


def set_policy_backend(backend):
    """Inject a stage-agnostic policy backend and return the previous one."""
    global _POLICY_BACKEND
    if backend is not None and not callable(getattr(backend, "generate", None)):
        raise TypeError("policy backend must expose an async generate(...) method")
    previous = _POLICY_BACKEND
    _POLICY_BACKEND = backend
    return previous


def get_policy_backend():
    """Return the currently injected policy backend, if any."""
    return _POLICY_BACKEND

_client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=DASHSCOPE_API_KEY,
)

# Initialize tiktoken encoder for accurate token counting
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding, works well for Qwen
    print("[agent_loop] Using tiktoken for accurate token counting")
except Exception as e:
    print(f"[agent_loop] Warning: Failed to load tiktoken encoder: {e}")
    _tokenizer = None


def _today_date() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")


def _extract_between(text: str, start_tag: str, end_tag: str) -> str:
    """Extract content between two tags."""
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return ""
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()
    return text[start_idx:end_idx].strip()


def _estimate_tokens(messages: list) -> int:
    """Accurate token counting using tiktoken (fallback to rough estimation)."""
    if _tokenizer is not None:
        try:
            total_tokens = 0
            for msg in messages:
                content = msg.get("content", "")
                total_tokens += len(_tokenizer.encode(content))
            return total_tokens
        except Exception as e:
            print(f"[agent_loop] Tokenizer error: {e}, falling back to char estimation")

    # Fallback: rough estimation
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars / 1.5)


def _truncate_tool_result(result: str) -> str:
    """Truncate tool results to keep context manageable."""
    if len(result) > MAX_TOOL_RESULT_CHARS:
        return result[:MAX_TOOL_RESULT_CHARS] + "\n\n[... result truncated due to length ...]"
    return result


def _normalize_answer(answer: str) -> str:
    """Clean up answer format: remove common prefixes, trailing punctuation, and quote wrapping."""
    text = answer.strip()
    # Remove common prefixes
    for prefix in [
        "The answer is ", "the answer is ",
        "Answer: ", "answer: ",
        "答案是", "答案：",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Remove wrapping quotes (Chinese and English)
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()
        if (text[0] == '\u201c' and text[-1] == '\u201d') or (text[0] == '\u2018' and text[-1] == '\u2019'):
            text = text[1:-1].strip()
        if (text[0] == '\u300c' and text[-1] == '\u300d'):
            text = text[1:-1].strip()
    return text


def _extract_tool_call(content: str) -> str | None:
    """Extract tool call JSON from content, supporting multiple formats."""
    # Format 1: <tool_call>...</tool_call>
    if "<tool_call>" in content and "</tool_call>" in content:
        return _extract_between(content, "<tool_call>", "</tool_call>")
    if "<tool_call>" in content:
        return content[content.find("<tool_call>") + len("<tool_call>"):].strip()

    # Format 2: <function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>
    func_match = re.search(r'<function=(search|visit)>', content)
    if func_match:
        tool_name = func_match.group(1)
        func_start = func_match.start()
        func_end_tag = content.find("</function>", func_start)
        if func_end_tag == -1:
            func_block = content[func_start:]
        else:
            func_block = content[func_start:func_end_tag]
        args = {}
        for param_match in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', func_block, re.DOTALL):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            try:
                args[param_name] = json5.loads(param_value)
            except Exception:
                args[param_name] = param_value
        result = json.dumps({"name": tool_name, "arguments": args})
        print(f"[agent] Converted <function={tool_name}> to JSON: {result[:200]}")
        return result

    # Format 3: bare JSON tool call (no tags)
    bare_match = re.search(r'\{"name":\s*"(search|visit)"', content)
    if bare_match:
        json_start = bare_match.start()
        brace_depth = 0
        json_end = json_start
        for i in range(json_start, len(content)):
            if content[i] == '{':
                brace_depth += 1
            elif content[i] == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = i + 1
                    break
        return content[json_start:json_end]
    return None


CONTENT_FILTER_MARKER = "__CONTENT_FILTERED__"
_OBSERVER = NullObserver()
_CURRENT_ROUND = 0
def _observe(event_type, payload=None):
    try: _OBSERVER.record_event(event_type, copy.deepcopy(payload or {}))
    except Exception: pass


async def _call_llm(
    messages: list,
    stop: Optional[list] = None,
    temperature: float = 0.4,
    max_tokens: int = 8192,
) -> str:
    """Call LLM via DashScope OpenAI-compatible API with retries."""
    if _POLICY_BACKEND is not None:
        _observe("round_request", {"round_idx": _CURRENT_ROUND, "model": getattr(_POLICY_BACKEND, "identity", "injected_policy"), "messages_snapshot": messages})
        content = await _POLICY_BACKEND.generate(messages, stop=stop, temperature=temperature, max_tokens=max_tokens)
        _observe("round_response", {"round_idx": _CURRENT_ROUND, "content": content, "finish_reason": "stop", "policy_backend": getattr(_POLICY_BACKEND, "identity", "injected_policy")})
        return content
    for attempt in range(3):
        try:
            _observe("round_request", {"round_idx": _CURRENT_ROUND, "model": AGENT_MODEL, "temperature": temperature, "max_tokens": max_tokens, "stop": stop or ["<tool_response>"], "extra_body": {"enable_thinking": True}, "messages_snapshot": messages})
            resp = await _client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                stop=stop or ["<tool_response>"],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": True},
            )
            msg = resp.choices[0].message
            # qwen3.5 with thinking mode: reasoning is in reasoning_content field
            reasoning = getattr(msg, "reasoning_content", None) or ""
            content = msg.content or ""
            _observe("round_response", {"round_idx": _CURRENT_ROUND, "reasoning_content": reasoning, "content": content, "finish_reason": getattr(resp.choices[0], "finish_reason", None)})
            # Clean up garbled tag fragments at start of content (e.g. "ynchroneg>")
            content = re.sub(r'^[a-z]*>', '', content.lstrip())
            # Reassemble: wrap reasoning in <think> tags + content
            if reasoning:
                full = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                full = content
            if full and full.strip():
                return full.strip()
            print(f"[agent] Empty response on attempt {attempt + 1}")
        except Exception as e:
            err_str = str(e)
            print(f"[agent] LLM call attempt {attempt + 1} failed: {e}")
            # Content safety filter - no point retrying the same prompt
            if "data_inspection_failed" in err_str:
                print(f"[agent] Content filter triggered, signaling redirect")
                return CONTENT_FILTER_MARKER
            if attempt < 2:
                await _async_sleep(2 ** attempt)

    return "I was unable to process this request due to an API error."


async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)


async def _force_answer(messages: list) -> str:
    """Force the LLM to generate a final answer based on gathered info."""
    force_msg = (
        "You have reached the limit. Based on ALL information gathered, provide your final answer.\n"
        "IMPORTANT: Use <think> tags to:\n"
        "1. Re-read the original question\n"
        "2. List ALL constraints from the question\n"
        "3. Choose the candidate that best satisfies ALL constraints\n"
        "Then answer:\n<think>your final reasoning</think>\n<answer>your answer</answer>"
    )
    messages.append({"role": "user", "content": force_msg})

    content = await _call_llm(messages, stop=["</answer>"], temperature=0.3)
    _observe("policy_output_processed", {"round_idx": _CURRENT_ROUND, "runtime_accepted_text": content, "transformation_type": [], "generation_path": "force_answer"})
    messages.append({"role": "assistant", "content": content})

    if "<answer>" in content:
        answer = _extract_between(content, "<answer>", "</answer>")
        if answer:
            return _normalize_answer(answer)
        # If </answer> was the stop token, the content after <answer> is the answer
        idx = content.find("<answer>")
        answer = content[idx + len("<answer>"):].strip()
        if answer:
            return _normalize_answer(answer)

    # Last resort: return the raw content
    return content


async def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool call and return the result string."""
    try:
        if tool_name == "search":
            queries = tool_args.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            engines = tool_args.get("engine", None)
            if isinstance(engines, str):
                engines = [engines]
            _observe("search_start", {"round_idx": _CURRENT_ROUND, "queries": queries, "engines": engines})
            started = time.perf_counter()
            result = batch_search(queries, engines=engines)
            _observe("search_finish", {"round_idx": _CURRENT_ROUND, "success": True, "latency_ms": (time.perf_counter()-started)*1000, "result_snapshot": result})
            return _truncate_tool_result(result)
        elif tool_name == "visit":
            return "Visit is disabled in this Search-only policy runtime."
        else:
            return f"Error: Unknown tool '{tool_name}'"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


async def react_agent(question: str, initial_messages: list | None = None) -> str:
    """
    Main ReAct agent loop.

    Follows the DeepResearch pattern:
    - System prompt with behavioral principles
    - User prompt with tool definitions and format examples
    - Iterative think → tool_call → tool_response → ... → answer
    - Timeout and token limit management
    """
    messages = copy.deepcopy(initial_messages) if initial_messages is not None else [
        {"role": "system", "content": SYSTEM_PROMPT + _today_date()},
        {"role": "user", "content": USER_PROMPT_TEMPLATE + question},
    ]

    _observe("trajectory_start", {"question": question, "messages_snapshot": messages})
    start_time = time.time()
    early_answer_blocked = False  # only block early answer once
    content_filter_count = 0  # track consecutive content filter hits

    global _CURRENT_ROUND
    for round_idx in range(MAX_ROUNDS):
        _CURRENT_ROUND = round_idx + 1
        # Timeout check
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_SECONDS:
            print(f"[agent] Timeout after {elapsed:.0f}s at round {round_idx + 1}")
            return await _force_answer(messages)

        # Call LLM
        current_tokens = _estimate_tokens(messages)
        print(f"[agent] Round {round_idx + 1}, elapsed {elapsed:.0f}s, tokens {current_tokens}")
        content = await _call_llm(messages)

        # Handle content filter: redirect or force a neutral search
        if content == CONTENT_FILTER_MARKER:
            _observe("policy_output_processed", {"round_idx": _CURRENT_ROUND, "runtime_accepted_text": "", "transformation_type": ["filter"], "generation_path": "content_filter"})
            content_filter_count += 1
            print(f"[agent] Content filter hit #{content_filter_count} at round {round_idx + 1}")
            if content_filter_count <= 2:
                redirect_msg = (
                    "The previous response was blocked by content filters. "
                    "Please try a different approach: rephrase your search using more neutral, "
                    "academic, or indirect terms. Focus on factual and historical aspects. "
                    "Use <think> to plan a new search angle, then make a tool call."
                )
                messages.append({"role": "assistant", "content": "<think>My previous response was filtered. I need to rephrase my approach using more neutral terms.</think>"})
                messages.append({"role": "user", "content": redirect_msg})
            else:
                # After 2 redirects, context itself is toxic — force answer with sanitized context
                print(f"[agent] Too many content filters, forcing answer with sanitized context")
                # Keep only system prompt, original question, and a summary of findings
                sanitized_messages = [messages[0], messages[1]]  # system + user question
                # Check if model already found an answer in earlier rounds
                for msg in messages:
                    if msg["role"] == "assistant":
                        c = msg["content"]
                        if "<answer>" in c:
                            answer_text = _extract_between(c, "<answer>", "</answer>")
                            if answer_text:
                                return _normalize_answer(answer_text)
                # Extract key findings from assistant messages (avoid sensitive details)
                findings = []
                for msg in messages:
                    if msg["role"] == "assistant":
                        for line in msg["content"].split("\n"):
                            if any(kw in line for kw in ["答案", "answer", "结论", "确认", "因此", "全名", "全称"]):
                                findings.append(line[:200])
                if findings:
                    sanitized_messages.append({
                        "role": "user",
                        "content": "Based on your earlier research, you found:\n" + "\n".join(findings[-5:]) + "\nNow give your final answer."
                    })
                return await _force_answer(sanitized_messages)
            continue

        transformations = []
        # Clean up: remove any hallucinated <tool_response> tags
        if "<tool_response>" in content:
            content = content[: content.find("<tool_response>")]
            transformations.append("truncate_after_tool_response")

        # Detect stalled output (very short meaningless content like "<" or empty)
        if len(content.strip()) < 5 and "<answer>" not in content:
            _observe("policy_output_processed", {"round_idx": _CURRENT_ROUND, "runtime_accepted_text": "", "transformation_type": transformations + ["stalled_generation_drop"], "generation_path": "standard"})
            print(f"[agent] Stalled output detected: '{content.strip()}', retrying")
            continue

        _observe("policy_output_processed", {"round_idx": _CURRENT_ROUND, "runtime_accepted_text": content, "transformation_type": transformations, "generation_path": "standard"})
        messages.append({"role": "assistant", "content": content})

        # Debug: print first 500 chars of model output
        print(f"[agent] Output preview: {content[:500]}")
        # Check for <think> usage and prepare reminder
        think_reminder = "<think>" not in content and round_idx > 0

        # Check for final answer (with or without closing tag)
        if "<answer>" in content:
            if "</answer>" in content:
                answer_text = _extract_between(content, "<answer>", "</answer>")
            else:
                # No closing tag — extract everything after <answer>
                answer_text = content[content.find("<answer>") + len("<answer>"):].strip()
                # Take only the first line to avoid grabbing trailing content
                if "\n" in answer_text:
                    answer_text = answer_text.split("\n")[0].strip()
                print(f"[agent] Detected <answer> without </answer>, extracted: {answer_text[:100]}")

            if answer_text:
                # Early answer gating: if answered too quickly, inject verification prompt
                elapsed = time.time() - start_time
                if round_idx < 3 and elapsed < 120 and not early_answer_blocked:
                    early_answer_blocked = True
                    verify_msg = (
                        "You answered very quickly. For complex multi-hop questions, "
                        "this often leads to errors. Please verify:\n"
                        "1. Re-read the original question\n"
                        "2. Check EVERY constraint\n"
                        "3. Search for one more confirming source\n"
                        "If correct, repeat your answer. If not, search more."
                    )
                    messages.append({"role": "user", "content": verify_msg})
                    print(f"[agent] Early answer blocked at round {round_idx + 1}, requesting verification")
                    continue

                print(f"[agent] Got answer at round {round_idx + 1}: {answer_text[:100]}")
                return _normalize_answer(answer_text)

        # Check for tool call — support both <tool_call> tags and bare JSON
        tool_call_str = _extract_tool_call(content)
        _observe("tool_extracted", {"round_idx": round_idx + 1, "extractor_result": tool_call_str})

        if tool_call_str:
            try:
                tool_data = json5.loads(tool_call_str)
                _observe("tool_json_parse", {"round_idx": round_idx + 1, "json_parse_success": True, "parsed_tool_name": tool_data.get("name"), "parsed_tool_args": tool_data.get("arguments")})
                tool_name = tool_data.get("name", "")
                tool_args = tool_data.get("arguments", {})
                print(f"[agent] Calling tool: {tool_name}")

                result = await _execute_tool(tool_name, tool_args)

                prefix = ""
                if think_reminder:
                    prefix = "Reminder: Use <think> tags to reason before every tool call or answer.\n\n"

                messages.append({
                    "role": "user",
                    "content": f"{prefix}<tool_response>\n{result}\n</tool_response>",
                })
                _observe("tool_response_appended", {"round_idx": round_idx + 1, "tool_response": result, "messages_snapshot": messages})
            except Exception as e:
                _observe("tool_json_parse", {"round_idx": round_idx + 1, "json_parse_success": False, "extractor_result": tool_call_str})
                error_msg = (
                    f'Error: Tool call is not valid JSON. '
                    f'Tool call must contain a valid "name" and "arguments" field. '
                    f'Error: {str(e)}'
                )
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{error_msg}\n</tool_response>",
                })
        else:
            # No tool call and no answer — nudge the LLM
            nudge_msg = (
                "You did not make a tool call or provide an answer. "
                "Please use <think> to reason, then either make a tool call or provide your answer."
            )
            messages.append({"role": "user", "content": nudge_msg})
            _observe("nudge_appended", {"round_idx": round_idx + 1, "messages_snapshot": messages})

        # Token limit check
        if _estimate_tokens(messages) > MAX_TOKENS_ESTIMATE:
            print(f"[agent] Token limit reached at round {round_idx + 1}")
            return await _force_answer(messages)

    # Exhausted all rounds
    print(f"[agent] Max rounds ({MAX_ROUNDS}) exhausted")
    return await _force_answer(messages)

async def react_agent_with_sidecar(
    question: str,
    sidecar=None,
    initial_messages: list | None = None,
    policy_backend=None,
) -> str:
    """Thin observability wrapper; it never changes Champion messages or actions."""
    global _OBSERVER
    previous = _OBSERVER
    previous_backend = get_policy_backend()
    _OBSERVER = sidecar or NullObserver()
    if policy_backend is not None:
        set_policy_backend(policy_backend)
    try:
        answer = await react_agent(question, initial_messages=initial_messages)
        _observe("trajectory_finish", {"answer": answer, "observed_termination_path": "normal_or_force_answer"})
        return answer
    finally:
        _OBSERVER = previous
        if policy_backend is not None:
            set_policy_backend(previous_backend)
