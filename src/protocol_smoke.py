#!/usr/bin/env python3
"""One-question local protocol diagnostic; no Visit, Guard, or public Web."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from urllib.request import Request, urlopen

QUESTION = "Who was Marie Curie's husband?"

def search(url: str, query: str) -> list[dict]:
    request = Request(url.rstrip("/") + "/search", data=json.dumps({"query": query, "top_n": 5, "return_score": False}).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read())

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model-path", required=True); p.add_argument("--protocol", choices=["champion","smartsearch"], required=True); p.add_argument("--retriever-url", required=True); p.add_argument("--out", required=True); a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device="cuda" if torch.cuda.is_available() else "cpu"; dtype=torch.float16 if device=="cuda" else torch.float32
    tok=AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=dtype, local_files_only=True).to(device).eval()
    if a.protocol == "champion":
        from champion_agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
        diagnostic_rule = "\nDiagnostic requirement: before any answer, emit exactly one search <tool_call>; do not answer from memory. Do not use visit.\n"
        prompt=[{"role": "system", "content": SYSTEM_PROMPT + "2026-09-02"}, {"role": "user", "content": USER_PROMPT_TEMPLATE + QUESTION + diagnostic_rule}]
        pattern=r'<tool_call>.*?\"query\"\s*:\s*\[\s*\"(.*?)\"'
    else:
        prompt=("You can use Wikipedia search. Write search queries only in <search>query</search>, then answer in <answer>answer</answer>. Question: " + QUESTION)
        pattern=r'<search>\s*(.*?)\s*</search>'
    def generate(text: str | list[dict]) -> str:
        if isinstance(text, list): text=tok.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        inputs=tok(text, return_tensors="pt").to(device)
        with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    start=time.time(); first=generate(prompt); match=re.search(pattern, first, re.S); record={"question":QUESTION,"protocol":a.protocol,"model_path":a.model_path,"first_output":first,"search_query":match.group(1).strip() if match else None,"steps":[]}
    if match:
        docs=search(a.retriever_url, match.group(1).strip()); evidence="\n".join(d["contents"] for d in docs)
        if a.protocol == "champion":
            second_input=prompt + [{"role": "assistant", "content": first}, {"role": "user", "content": "<tool_response>\n" + evidence + "\n</tool_response>"}]
        else:
            second_input=prompt+first+"<result>"+evidence+"</result>"
        second=generate(second_input)
        record["steps"]=[{"query":match.group(1).strip(),"doc_ids":[d["id"] for d in docs],"second_output":second}]
    record["latency_seconds"]=time.time()-start
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(record,ensure_ascii=False) + "\n")
    print(json.dumps({"protocol":a.protocol,"search_emitted":bool(match),"trace":a.out},ensure_ascii=False))
if __name__ == "__main__": main()
