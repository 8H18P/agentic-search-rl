"""Strict, deterministic SmartSearch-to-Champion syntax adapter."""
from __future__ import annotations
import json, re

def adapt_smartsearch_action(raw_output: str) -> dict:
    raw = raw_output or ""
    s = raw.strip()
    if re.fullmatch(r"<tool_call>\s*\{.*?\}\s*</tool_call>", s, re.S):
        return {"matched": True, "adapter_rule": "champion_native_passthrough", "kind": "invalid", "query": "", "answer": "", "raw": raw}
    m = re.fullmatch(r"search\s*(\{.*\})", s, re.I | re.S)
    if m:
        try:
            obj=json.loads(m.group(1)); q=obj.get("query", "")
            if isinstance(q,list): q=q[0] if len(q)==1 else ""
            if isinstance(q,str) and q.strip():
                return {"matched":True,"adapter_rule":"bare_search_json","kind":"search","query":q.strip(),"answer":"","raw":raw}
        except Exception: pass
    m = re.fullmatch(r"<search>\s*(.*?)\s*</search>", s, re.I | re.S)
    if m and m.group(1).strip():
        return {"matched":True,"adapter_rule":"smartsearch_xml_search","kind":"search","query":m.group(1).strip(),"answer":"","raw":raw}
    m = re.fullmatch(r"<answer>\s*(.*?)\s*</answer>", s, re.I | re.S)
    if m: return {"matched":True,"adapter_rule":"smartsearch_xml_answer","kind":"answer","query":"","answer":m.group(1).strip(),"raw":raw}
    m = re.fullmatch(r"\(?\s*answer\s*:\s*(.*?)\s*\)?", s, re.I | re.S)
    if m:
        rule="parenthesized_answer" if s.lstrip().startswith("(") else "bare_answer_prefix"
        ans=m.group(1).strip()
        if rule=="parenthesized_answer" and ans.endswith(")"): ans=ans[:-1].rstrip()
        return {"matched":True,"adapter_rule":rule,"kind":"answer","query":"","answer":ans,"raw":raw}
    if re.search(r"\bvisit\b|<visit", s, re.I):
        return {"matched":False,"adapter_rule":"adapter_no_explicit_action","kind":"forbidden_policy_tool","query":"","answer":"","raw":raw,"reason":"forbidden_policy_tool","proposed_tool":"visit"}
    return {"matched":False,"adapter_rule":"adapter_no_explicit_action","kind":"invalid","query":"","answer":"","raw":raw}
