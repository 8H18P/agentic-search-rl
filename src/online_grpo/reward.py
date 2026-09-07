"""Frozen V3 query scoring from authoritative execution events, not text actions."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_v3(root):
    manifest = json.loads((root/'configs/process_judge/v3_frozen/V3_FREEZE_MANIFEST.json').read_text())
    checked = {}
    for key, path in [
        ('judge_contract', 'src/agentic_search_rl/rewards/judge_v3.py'),
        ('v3_channel_runner_and_parser', 'scripts/prm/run_query_level_judge_v3_channel.py'),
        ('intent_prompt', 'configs/process_judge/v3_frozen/intent_prompt.txt'),
        ('retrieval_prompt', 'configs/process_judge/v3_frozen/retrieval_prompt.txt'),
        ('judge_config', 'configs/process_judge/v3_frozen/judge_config.json'),
        ('output_schema', 'configs/process_judge/v3_frozen/output_schema.json')]:
        actual = sha(root/path)
        if actual != manifest['sha256'][key]:
            raise ValueError(f'frozen V3 hash mismatch: {path}')
        checked[path] = actual
    return checked


def build_units(root, question, capture, trace_path):
    from agentic_search_rl.rewards import judge_v3
    history, units = [], []
    events = capture.events
    for event_index, event in enumerate(events):
        if event['event_type'] != 'search_start':
            continue
        p = event['payload']; round_idx = p['round_idx']; queries = p['queries']
        tail = events[event_index+1:]
        finish = next((e for e in tail if e['event_type']=='search_finish' and e['payload'].get('round_idx')==round_idx), None)
        appended = next((e for e in tail if e['event_type']=='tool_response_appended' and e['payload'].get('round_idx')==round_idx), None)
        if finish is None or appended is None or not finish['payload'].get('success'):
            raise ValueError('incomplete authoritative Search event chain')
        observation = appended['payload']['tool_response']
        sections = judge_v3.parse_sections(observation, queries)
        response = next(e['payload']['content'] for e in reversed(events[:event_index]) if e['event_type']=='round_response')
        # Only reasoning is read from policy text; action/query come exclusively from search_start.
        reasoning = response.split('<think>',1)[-1].split('</think>',1)[0] if '<think>' in response else ''
        action_idx = len(history)+1
        for section in sections:
            unit = {'unit_id': f'{capture.trajectory_id}::action_{action_idx}::query_{section["query_idx"]}',
                'trajectory_id': capture.trajectory_id, 'question_id': question['id'], 'question': question['question'],
                'golden_answer': question['golden_answers'], 'action_idx': action_idx, 'query_idx': section['query_idx'],
                'authoritative_history_before_action': list(history), 'current_action_reasoning': reasoning,
                'current_executed_query': section['query'], 'current_query_result': section['result_section'],
                'source_refs': {'trace_path': str(trace_path), 'search_start_event_index': event_index,
                                'observation_source': 'tool_response_appended', 'action_source': 'search_start'},
                'current_query_result_sha256': section['result_sha256']}
            unit['intent_prompt'] = judge_v3.intent_prompt(unit)
            unit['retrieval_prompt'] = judge_v3.retrieval_prompt(unit['current_executed_query'],unit['current_query_result'])
            units.append(unit)
        history.append({'action_idx':action_idx, 'reasoning':reasoning,
            'executed_action':{'name':'search','arguments':{'query':list(queries)}},
            'observation':observation,'observation_sha256':hashlib.sha256(observation.encode()).hexdigest()})
    return units


def score_units(root, units, config, cache_dir):
    verify_v3(root)
    runner = load_module(root/'scripts/prm/run_query_level_judge_v3_channel.py', 'grpo_v3_runner')
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = os.environ.get(config['api_key_env'], '')
    records = []
    for unit in units:
        scores = {}
        for channel in ('intent','retrieval'):
            prompt = unit[channel+'_prompt']
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            cache_key = hashlib.sha256((sha(root/'scripts/prm/run_query_level_judge_v3_channel.py')+channel+config['base_url']+unit['unit_id']+prompt_hash).encode()).hexdigest()
            cache_path = cache_dir/(cache_key+'.json')
            if cache_path.exists():
                result = json.loads(cache_path.read_text())
            else:
                if not key:
                    raise RuntimeError('V3 API key not available in configured environment variable')
                payload = {**unit,'judge_prompt':prompt,'prompt_sha256':prompt_hash,
                           'calibration_id':channel+'::'+unit['unit_id'],'expected_'+channel:None}
                result = runner.run_one(payload, SimpleNamespace(channel=channel,base_url=config['base_url']), key)
                # Do not serialize credentials even if an upstream service echoes one.
                serialized = json.dumps(result,ensure_ascii=False).replace(key,'[REDACTED]')
                cache_path.write_text(serialized+'\n')
            if result.get('status')!='ok':
                raise RuntimeError(f'V3 unscorable query: {unit["unit_id"]}/{channel}; see redacted cache')
            scores[channel] = result['parsed'][channel]
            scores[channel+'_explanation'] = result['parsed']['explanation']
            scores[channel+'_cache_path'] = str(cache_path)
        records.append({'unit_id':unit['unit_id'],'trajectory_id':unit['trajectory_id'],
                        'action_idx':unit['action_idx'],'query_idx':unit['query_idx'],**scores,
                        'process_score':int(scores['intent']==1 and scores['retrieval']==1)})
    return records


def mini_reward(outcome_correct, scores, process_weight):
    if not 0 <= process_weight < 1:
        raise ValueError('process_weight must preserve correct > incorrect')
    if any(s not in (0,1) for s in scores):
        raise ValueError('missing/non-binary process score')
    quality = sum(scores)/len(scores) if scores else 0.0
    return {'profile':'champion_mini_grpo_v1_query_mean', 'outcome':int(outcome_correct),
            'process_quality':quality, 'good_query_count':sum(scores),'total_query_count':len(scores),
            'process_weight':process_weight,'reward':int(outcome_correct)+process_weight*quality,
            'formula':'outcome_correct + process_weight * good_queries / total_queries; empty quality=0',
            'smartsearch_source_equivalent':False}


def smartsearch_reward_reference(correct, good, bad, format_valid):
    """Source formula kept only as an explicit future ablation reference."""
    return (max(1-.1*bad,.7) if correct else min(.1*good,.3))+.1*format_valid


def metadata_reward(completions=None, authoritative_rewards=None, **kwargs):
    if authoritative_rewards is None:
        raise ValueError('reward requires authoritative metadata; decoded completion is not an action source')
    return list(authoritative_rewards)
