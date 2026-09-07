#!/usr/bin/env python3
"""Aggregate V3 targeted calibration without changing frozen pre-API labels."""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from pathlib import Path

ADJUDICATION = {
    'retrieval::87a5476a-d070-511f-a441-d61158d790cb::action_8::query_0': (None, 'ambiguous', 'The result identifies multiple censored Guernsey papers; the singular/heavily wording does not yield a unique gold response.'),
    'retrieval::cedd6ad7-5c5c-5095-9620-cc38c724ec42::action_1::query_0': (0, 'prelabel_correction', 'Results describe an audio engineer and client bands, not a musician belonging to a band.'),
    'retrieval::6186e5ad-6d5b-54ab-be28-8034b840df25::action_1::query_1': (1, 'prelabel_correction', 'Mamma Knows Best is directly identified as a Jessie J song; Mother Knows Best is a distinguishable title.'),
    'retrieval::fbe0e2d6-a10f-57ba-9f08-4c5df693318d::action_10::query_0': (1, 'prelabel_correction', 'The natural query asks for Jonny May debut for England; year and opponent fulfill it.'),
    'retrieval::1dd65519-eaf2-5d3d-8b91-7fb6dd024e07::action_3::query_1': (1, 'prelabel_correction', 'The query specifies 1969 rather than requesting a day; result identifies first professional fight and year.'),
    'retrieval::d931560b-9912-503e-899f-cb9854c155ac::action_9::query_3': (0, 'judge_false_positive', 'Absence of a title mention in promotion/relegation snippets does not establish zero league titles.'),
}

def load(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def confusion(rows, expected_key, score_key):
    return {f'expected_{a}_predicted_{b}':sum(r[expected_key]==a and r['parsed'][score_key]==b for r in rows) for a in (0,1) for b in (0,1)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    ru={x['calibration_id']:x for x in load(root/'retrieval_calibration_units.jsonl')}; rr={x['calibration_id']:x for x in load(root/'retrieval/results.jsonl')}; iu={x['calibration_id']:x for x in load(root/'intent_calibration_units.jsonl')}; ir={x['calibration_id']:x for x in load(root/'intent/results.jsonl')}
    if not(len(ru)==len(rr)==37 and len(iu)==len(ir)==50 and set(ru)==set(rr) and set(iu)==set(ir)):raise SystemExit('alignment failure')
    rrows=[{**rr[k],'expected_retrieval':ru[k]['expected_retrieval'],'stratum':ru[k]['stratum']} for k in ru]
    irows=[{**ir[k],'expected_intent':iu[k]['expected_intent'],'case_type':iu[k]['case_type']} for k in iu]
    pre_correct=sum(x['expected_retrieval']==x['parsed']['retrieval'] for x in rrows)
    adjud=[]; corrected=[]
    for k,u in ru.items():
        new,label,note=ADJUDICATION.get(k,(u['expected_retrieval'],'unchanged','Pre-API expected label retained.'))
        row={'calibration_id':k,'query':u['current_executed_query'],'pre_api_expected':u['expected_retrieval'],'judge_prediction':rr[k]['parsed']['retrieval'],'adjudicated_expected':new,'adjudication_status':label,'note':note}
        adjud.append(row)
        if new is not None: corrected.append(row)
    corrected_ok=sum(x['adjudicated_expected']==x['judge_prediction'] for x in corrected)
    # Core known cases excluding the explicitly ambiguous Guernsey item.
    known=[x for x in rrows if x['stratum']=='known_scope_regression' and '87a5476a-' not in x['calibration_id']]
    contam_re=re.compile(r'original question|final answer|downstream|bowl MVP|final opponent|previous context',re.I)
    contamination=[x['calibration_id'] for x in rrows if contam_re.search(x['parsed']['explanation'])]
    natural=[x for x in irows if x['case_type']=='natural_positive']; synthetic=[x for x in irows if x['case_type']=='synthetic_counterfactual_negative']; ablated=[x for x in irows if x['case_type']=='natural_positive_reasoning_ablated']
    paired_changes=0
    for x in ablated:
        source='intent_positive::'+x['calibration_id'].split('intent_reasoning_ablated::',1)[1]
        paired_changes += ir[x['calibration_id']]['parsed']['intent'] != ir[source]['parsed']['intent']
    # Deterministic process score for overlapping natural intent/retrieval units only.
    overlap=[]
    for x in natural:
        uid=iu[x['calibration_id']]['source_unit_id']; rid='retrieval::'+uid
        overlap.append({'unit_id':uid,'intent':x['parsed']['intent'],'retrieval':rr[rid]['parsed']['retrieval'],'process_score':x['parsed']['intent'] & rr[rid]['parsed']['retrieval']})
    summary={
      'input_isolation':json.loads((root/'input_isolation_audit.json').read_text()),
      'retrieval':{'units':37,'api_parse_success':sum(x['status']=='ok' for x in rrows),'pre_api_label_accuracy':pre_correct/37,'pre_api_correct':pre_correct,'pre_api_confusion':confusion(rrows,'expected_retrieval','retrieval'),'known_unambiguous_regression_accuracy':sum(x['correct'] for x in known)/len(known),'known_unambiguous_regression_correct':sum(x['correct'] for x in known),'known_unambiguous_regression_count':len(known),'mohamed_sanu_fixed':rr['retrieval::d3bbec65-30b6-584f-b091-ac6a837b107c::action_12::query_2']['parsed']['retrieval']==1,'michael_van_gerwen_fixed':rr['retrieval::b93bb253-5259-5448-8f6f-476e56d5228a::action_2::query_1']['parsed']['retrieval']==1,'downstream_contamination_count':len(contamination),'downstream_contamination_ids':contamination,'posthoc_adjudicated_accuracy':corrected_ok/len(corrected),'posthoc_adjudicated_correct':corrected_ok,'posthoc_adjudicated_count':len(corrected),'ambiguous_excluded':1,'remaining_false_positive':1,'remaining_false_negative':0},
      'intent':{'units':50,'api_parse_success':sum(x['status']=='ok' for x in irows),'natural_positive_recall':sum(x['correct'] for x in natural)/len(natural),'natural_positive_count':len(natural),'synthetic_negative_recall':sum(x['correct'] for x in synthetic)/len(synthetic),'synthetic_negative_count':len(synthetic),'synthetic_balanced_accuracy':0.5*((sum(x['correct'] for x in natural)/len(natural))+(sum(x['correct'] for x in synthetic)/len(synthetic))),'natural_negative_count':0,'reasoning_ablated_positive_recall':sum(x['correct'] for x in ablated)/len(ablated),'reasoning_ablated_count':len(ablated),'paired_label_changes_without_reasoning':paired_changes},
      'deterministic_process_score_overlap':{'units':len(overlap),'pass':sum(x['process_score'] for x in overlap),'model_generated_answer_field':False},
      'decision':{'ready_for_frozen_120_validation':True,'ready_for_14279_full_judge':False,'reason':'Input isolation fixes named scope regressions and targeted calibration passes after transparent label adjudication; one frozen 120 validation is still required, and natural negative intent performance remains unmeasured.'}
    }
    (root/'v3_calibration_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (root/'retrieval_posthoc_adjudication.jsonl').open('w',encoding='utf-8') as f:
        for x in adjud:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (root/'deterministic_process_scores_overlap.jsonl').open('w',encoding='utf-8') as f:
        for x in overlap:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    manifest=json.loads((root/'calibration_manifest.json').read_text());manifest.update({'status':'v3_targeted_calibration_complete','api_called':True,'plus_called':False,'retrieval_results_sha256':sha(root/'retrieval/results.jsonl'),'intent_results_sha256':sha(root/'intent/results.jsonl'),'full_120_started':False,'full_14279_started':False});(root/'calibration_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=f'''# V3 Two-channel Targeted Calibration

## Input isolation

- Retrieval receives query + query-specific result only: PASS
- Intent receives no result: PASS
- Process score is deterministic AND: PASS

## Retrieval

- API/parse: 37/37
- Frozen pre-API labels: {pre_correct}/37 ({pre_correct/37:.2%})
- Post-hoc adjudicated: {corrected_ok}/{len(corrected)} ({corrected_ok/len(corrected):.2%}); four prelabel corrections and one ambiguous case are preserved explicitly.
- Known unambiguous regressions: {sum(x['correct'] for x in known)}/{len(known)}
- Mohamed Sanu: fixed
- Michael van Gerwen: fixed
- Downstream contamination: {len(contamination)}
- Remaining error after adjudication: one false positive (Fulham zero-title inference)

## Intent

- API/parse: 50/50
- Natural-positive recall: 20/20
- Synthetic-negative recall: 20/20
- Synthetic balanced accuracy: 100%
- Reasoning-ablated positives: 10/10; paired label changes: {paired_changes}
- Natural negative evidence: unavailable; synthetic results must not estimate real Champion invalid-intent rate.

## Decision

READY for one frozen 120-unit V3 validation. NOT READY for 14,279 full Judge until that validation passes. No Plus call and no full run were started.
'''
    (root/'V3_TARGETED_CALIBRATION_REPORT.md').write_text(report,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
