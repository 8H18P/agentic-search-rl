#!/usr/bin/env python3
"""Run one physically isolated V3 Judge channel on its frozen calibration set."""
from __future__ import annotations

import argparse, concurrent.futures, json, os, re, time
from pathlib import Path
import requests

INTENT_RE = re.compile(r"^\s*<intent>([01])</intent>\s*<intent_failure_type>(none|irrelevant|wrong_entity|malformed|other)</intent_failure_type>\s*<explanation>(.+?)</explanation>\s*$", re.S)
RETRIEVAL_RE = re.compile(r"^\s*<retrieval>([01])</retrieval>\s*<retrieval_failure_type>(none|retrieval_miss|retrieval_partial|result_entity_mismatch|other)</retrieval_failure_type>\s*<explanation>(.+?)</explanation>\s*$", re.S)


def parse(channel: str, text: str):
    match = (INTENT_RE if channel == "intent" else RETRIEVAL_RE).match(text or "")
    if not match: return None, "missing_or_invalid_required_tags"
    score, failure, explanation = match.groups(); score = int(score); explanation = explanation.strip()
    if not explanation: return None, "empty_explanation"
    if (score == 1) != (failure == "none"): return None, "score_failure_inconsistent"
    return {channel: score, f"{channel}_failure_type": failure, "explanation": explanation}, None


def run_one(unit, args, key):
    attempts=[]; parsed=None; error=None; status="api_error"
    for idx in range(1,4):
        start=time.time()
        try:
            response=requests.post(args.base_url.rstrip('/')+'/chat/completions',headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},json={'model':'qwen3.5-flash','messages':[{'role':'user','content':unit['judge_prompt']}],'temperature':0,'max_tokens':384,'enable_thinking':False},timeout=(15,180))
            raw=response.json() if response.content else {}; choice=(raw.get('choices') or [{}])[0]; msg=choice.get('message') or {}; content=msg.get('content') or ''; parsed,error=parse(args.channel,content); usage=raw.get('usage') or {}
            attempts.append({'attempt_idx':idx,'http_status':response.status_code,'latency_seconds':time.time()-start,'finish_reason':choice.get('finish_reason'),'content':content,'reasoning_content':msg.get('reasoning_content'),'usage':usage,'parse_error':error,'response_error':raw.get('error')})
            if response.status_code==200 and parsed is not None: status='ok'; break
            status='parse_error' if response.status_code==200 else 'api_error'
        except Exception as exc:
            attempts.append({'attempt_idx':idx,'http_status':None,'latency_seconds':time.time()-start,'exception_type':type(exc).__name__,'exception':str(exc)}); status='api_error'
        if idx<3: time.sleep(min(4,2**(idx-1)))
    expected=unit[f'expected_{args.channel}']; observed=parsed[args.channel] if parsed else None
    return {'calibration_id':unit['calibration_id'],'unit_id':unit.get('unit_id') or unit.get('source_unit_id'),'case_type':unit.get('case_type',unit.get('stratum')),'channel':args.channel,'status':status,'expected_label':expected,'parsed':parsed,'correct':status=='ok' and observed==expected,'attempt_count':len(attempts),'attempts':attempts,'prompt_sha256':unit['prompt_sha256']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--channel',choices=['intent','retrieval'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--base-url',required=True);p.add_argument('--concurrency',type=int,default=4);p.add_argument('--api-key-env',default='DMX_API_KEY');args=p.parse_args()
    key=os.environ.get(args.api_key_env)
    if not key: raise SystemExit('missing API key env')
    units=[json.loads(x) for x in args.input.read_text(encoding='utf-8').splitlines() if x.strip()]
    args.output_dir.mkdir(parents=True,exist_ok=True); out=args.output_dir/'results.jsonl'
    if out.exists() and out.stat().st_size: raise SystemExit('refusing existing results')
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,out.open('a',encoding='utf-8') as fh:
        futures=[pool.submit(run_one,u,args,key) for u in units]
        for f in concurrent.futures.as_completed(futures):
            row=f.result();rows.append(row);fh.write(json.dumps(row,ensure_ascii=False)+'\n');fh.flush();os.fsync(fh.fileno());print(json.dumps({'completed':len(rows),'status':row['status'],'correct':row['correct']}))
    order={u['calibration_id']:i for i,u in enumerate(units)};rows.sort(key=lambda r:order[r['calibration_id']])
    with out.open('w',encoding='utf-8') as fh:
        for r in rows:fh.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'channel':args.channel,'model':'qwen3.5-flash','units':len(rows),'ok':sum(r['status']=='ok' for r in rows),'correct':sum(r['correct'] for r in rows),'parse_error':sum(r['status']=='parse_error' for r in rows),'api_error':sum(r['status']=='api_error' for r in rows),'attempts':sum(r['attempt_count'] for r in rows)}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 0

if __name__=='__main__':raise SystemExit(main())
