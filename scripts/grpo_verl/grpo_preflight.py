#!/usr/bin/env python3
"""Real Champion / TRL GRPO forward-only preflight. Never a training entrypoint."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from online_grpo.reward import sha, verify_v3, build_units, score_units, mini_reward, metadata_reward


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values))


def select_question(config):
    """Exclude by IDs and normalized question text; no answer-based selection."""
    from rl_agent.evaluator import normalize
    manifest = read(ROOT/config['split_manifest'])
    pool = ROOT/config['question_pool']
    assert sha(pool) == manifest['files'][pool.name]['sha256']
    assert manifest['posttrain_validation_intersection'] == 0
    excluded_ids, excluded_questions, files = set(), set(), {}

    def collect(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                collect(v, k)
        elif isinstance(value, list):
            for v in value:
                collect(v, key)
        elif isinstance(value, str):
            if key in ('question', 'sample_questions'):
                excluded_questions.add(normalize(value))
            if key in ('id','question_id','sample_ids','trajectory_id','unit_id','source_unit_id','calibration_id'):
                excluded_ids.add(value.split('::')[0])

    for name in config['exclusion_files']:
        path = ROOT/name
        value = rows(path) if path.suffix == '.jsonl' else read(path)
        files[name] = sha(path)
        if path.name in manifest['files']:
            assert files[name] == manifest['files'][path.name]['sha256']
        collect(value)
    candidates = [r for r in rows(pool) if r['id'] not in excluded_ids and normalize(r['question']) not in excluded_questions]
    # Deterministic sampling from legal pool; not selected for correctness or predicted Search count.
    candidates.sort(key=lambda r:r['id'])
    selected = random.Random(config['seed']).choice(candidates)
    return selected, {'question':selected,'selection':'sorted legal IDs + Random(seed).choice',
        'seed':config['seed'],'legal_pool_count':len(candidates),'pool_sha256':sha(pool),
        'exclusion_sha256':files,'split_manifest_sha256':sha(ROOT/config['split_manifest']),
        'exclusion_id_count':len(excluded_ids),'exclusion_question_count':len(excluded_questions),
        'selected_id_overlap':False,'selected_question_overlap':False,
        'validation400_used':False,'validation_boundary_evidence':'frozen split manifest; no validation400 content read',
        'judge_dev_boundary':'entire SFT pool excluded, plus explicit heldout/dev/targeted calibration IDs'}


def main(config_path, resume=False):
    config = read(config_path)
    out = ROOT/config['output_dir']
    if out.exists() and any(out.iterdir()) and not resume:
        raise RuntimeError('refusing existing artifacts; explicit --resume needed')
    out.mkdir(parents=True, exist_ok=True)
    summary = {'backward_called':False,'optimizer_steps':0,'parameter_updated':False,
        'grpo_training_started':False,'validation400_used':False,'query_refinement_started':False,
        'ready_for_mini_grpo_smoke_training':False}
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig
        from online_grpo.adapter import (ChampionSegmentGRPOTrainer, parameter_groups, parameter_hash,
            enforce_default_ownership, ownership_audit)
        from online_grpo.capture import RolloutCapture, replay_segment, audit_segment, audit_flat, digest
        # Champion constructs an unused API client at import. All generation below
        # is injected locally; this non-secret sentinel is not an API credential.
        os.environ.setdefault('LLM_API_KEY', 'local-policy-unused')
        from champion_runtime import agent_loop, offline_search
        from champion_runtime.local_policy import HFPolicyBackend, HFPolicyConfig
        from champion_runtime.observability import JsonlObserver
        from rl_agent.evaluator import em_f1

        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        tag = subprocess.check_output(['git','rev-parse',config['baseline_tag']+'^{commit}'],cwd=ROOT,text=True).strip()
        assert head == tag == config['baseline_commit']
        frozen = verify_v3(ROOT)
        question, selection = select_question(config)
        write(out/'selection_manifest.json', selection)
        identity = read(ROOT/config['identity_path'])
        assert identity['expected_revision'] == identity['resolved_revision']
        base_path, tokenizer_path = Path(identity['model_local_path']), Path(identity['tokenizer_local_path'])
        for path, key in [(base_path/'config.json','config_sha256'),
                          (tokenizer_path/'tokenizer.json','tokenizer_json_sha256'),
                          (tokenizer_path/'tokenizer_config.json','tokenizer_config_sha256')]:
            assert sha(path) == identity[key], key
        adapter_path = ROOT/config['adapter_path']
        adapter_file = adapter_path/'adapter_model.safetensors'
        assert sha(adapter_file) == config['adapter_sha256']
        index = read(base_path/'model.safetensors.index.json')
        for shard in set(index['weight_map'].values()):
            assert (base_path/shard).is_file()
        with urllib.request.urlopen(config['retriever_url']+'/health',timeout=10) as response:
            health = json.load(response)
        versions = {p:importlib.metadata.version(p) for p in ('torch','transformers','peft','trl','accelerate')}
        assert versions['trl'] == '1.12.0'
        manifest = {'config':config,'config_sha256':sha(config_path),'model_identity':identity,
                    'adapter_sha256':sha(adapter_file),'baseline':head,'frozen_v3':frozen,'versions':versions,
                    'retriever_health':health,'question_id':question['id'],'group_id':'grpo_preflight_'+question['id']}
        manifest_path = out/'run_manifest.json'
        if resume and manifest_path.exists():
            prior = read(manifest_path)
            for key in ('config_sha256','adapter_sha256','baseline','frozen_v3','question_id','versions'):
                assert prior[key] == manifest[key], 'resume identity mismatch: '+key
        write(manifest_path,manifest)
        print('Loading exact local DPO policy',flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
        assert __import__('hashlib').sha256(tokenizer.chat_template.encode()).hexdigest() == identity['chat_template_sha256']
        base = AutoModelForCausalLM.from_pretrained(base_path,local_files_only=True,dtype=torch.bfloat16,
                attn_implementation=config['attention_backend'],low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base,adapter_path,local_files_only=True,is_trainable=True,
                                         autocast_adapter_dtype=False).to('cuda')
        groups = parameter_groups(model)
        initial_hash = parameter_hash(groups['default'])
        base_before = parameter_hash(groups['base'])
        adapter_config = read(adapter_path/'adapter_config.json')
        print('Constructing real TRL trainer and optimizer; steps prohibited',flush=True)
        args = GRPOConfig(output_dir=str(out/'trainer_scratch'),seed=config['seed'],**config['trl'])
        state = {'rollouts':[], 'segments':[]}

        def rollout_func(prompts, trainer):
            if state['rollouts']:
                raise RuntimeError('one group only; no implicit re-rollout')
            assert len(prompts) == config['trl']['num_generations']
            assert all(p == question['question'] for p in prompts)
            for slot in range(len(prompts)):
                seed = config['seed'] + slot
                trajectory_id = manifest['group_id']+f'_slot{slot}'
                trace_path = out/(trajectory_id+'_events.jsonl')
                cached = out/(trajectory_id+'_captured.json')
                sink = RolloutCapture(tokenizer,JsonlObserver(trace_path,trajectory_id),trajectory_id,
                    manifest['group_id'],question['id'],seed,config['max_seq_length'])
                if resume and cached.exists():
                    saved = read(cached)
                    assert saved['policy_hash'] == initial_hash
                    sink.records,sink.events,answer = saved['records'],saved['events'],saved['answer']
                else:
                    if trace_path.exists() and trace_path.stat().st_size:
                        raise RuntimeError('incomplete prior rollout; preserve it and select new output directory')
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    policy.capture_sink = sink
                    print(f'Champion group slot {slot}, seed {seed}',flush=True)
                    answer = asyncio.run(agent_loop.react_agent_with_sidecar(question['question'],sink,policy_backend=policy))
                    write(cached,{'records':sink.records,'events':sink.events,'answer':answer,'policy_hash':initial_hash})
                assert sink.records and all(r['accepted_text_observed'] for r in sink.records)
                assert parameter_hash(parameter_groups(model)['default']) == initial_hash
                segments = [replay_segment(r) for r in sink.records]
                for segment in segments:
                    audit_segment(segment)
                em,f1 = em_f1(answer,question['golden_answers'])
                state['rollouts'].append({'trajectory_id':trajectory_id,'question_id':question['id'],
                    'group_id':manifest['group_id'],'seed':seed,'answer':answer,'answer_em':em,'answer_f1':f1,
                    'outcome_correct':f1>=.8,'trace_path':str(trace_path),'policy_hash':initial_hash,'capture':sink})
                state['segments'].append(segments)
                write_rows(out/'rollout_capture.jsonl',[r for v in state['rollouts'] for r in v['capture'].records])
            return {'turn_segments':state['segments'],'trajectory_metadata':state['rollouts']}

        trainer = ChampionSegmentGRPOTrainer(model=model,args=args,processing_class=tokenizer,
            train_dataset=Dataset.from_list([{'prompt':question['question']}]),reward_funcs=metadata_reward,
            rollout_func=rollout_func,tools=None)
        # PEFT add_adapter defaults may promote the newly constructed ref to fp32.
        # Verify values first, then align only that construction dtype to the DPO
        # policy. Never recast/mutate default or hide a numerical copy mismatch.
        copy_audit=[]
        for name,p in parameter_groups(model)['default']:
            ref=model.get_parameter(name.replace('.default.','.ref.'))
            torch.testing.assert_close(p.float(),ref.float(),rtol=0,atol=0)
            copy_audit.append({'name':name,'default_dtype':str(p.dtype),'ref_created_dtype':str(ref.dtype),
                'values_exact':True,'ref_dtype_aligned':ref.dtype!=p.dtype})
            if ref.dtype!=p.dtype:
                ref.data=ref.data.to(p.dtype)
        write(out/'reference_construction_dtype_audit.json',copy_audit)
        enforce_default_ownership(model)
        model.eval()
        trainer.create_optimizer()
        ownership = ownership_audit(model,trainer.optimizer)
        write(out/'optimizer_ownership.json',ownership)
        groups = parameter_groups(model)
        ref_hash = parameter_hash(groups['ref'])
        assert parameter_hash(groups['default']) == ref_hash == initial_hash, 'post-construction adapter hashes differ'
        assert parameter_hash(groups['base']) == base_before
        write(out/'policy_reference_audit.json',{'policy_initial_hash':initial_hash,'reference_snapshot_hash':ref_hash,
            'base_full_parameter_hash':base_before,'policy_equals_pre_grpo_dpo':True,
            'reference_equals_pre_grpo_dpo':True,'default_adapter_trainable':True,'ref_adapter_frozen':True,
            'base_frozen':True,'construction':'TRL constructor clones pretrained default into ref on shared base',
            'trainable_params':sum(p.numel() for _,p in groups['default']),
            'total_params':sum(p.numel() for p in model.parameters())})
        summary.update(dpo_initial_policy_load_pass=True,default_adapter_trainable=True,ref_adapter_frozen=True,
            base_frozen=True,policy_equals_pre_grpo_dpo=True,reference_equals_pre_grpo_dpo=True,
            optimizer_owns_default_only=True)
        policy = HFPolicyBackend.from_model(model,tokenizer,HFPolicyConfig(
            base_model_path=str(base_path),tokenizer_path=str(tokenizer_path),adapter_path=str(adapter_path),
            policy_id='live_pre_grpo_dpo',freeze_parameters=False,generation=config['generation']))
        agent_loop.MAX_ROUNDS=config['max_rounds']; agent_loop.TIMEOUT_SECONDS=config['timeout_seconds']
        offline_search.RETRIEVER_URL=config['retriever_url']
        torch.cuda.reset_peak_memory_stats()
        callback_output = trainer.rollout_func([question['question']]*args.num_generations,trainer)
        captures = [r['capture'] for r in state['rollouts']]
        audits = [audit_segment(s) for ts in state['segments'] for s in ts]
        write_rows(out/'token_context_audit.jsonl',audits)
        flat = [audit_flat(c.records) for c in captures]
        write(out/'flat_transcript_audit.json',flat)
        temperatures = sorted({r['generation_temperature'] for c in captures for r in c.records})
        env_count = sum(a['state_origin_counts'].get('ENVIRONMENT_OBSERVATION',0) for a in audits)
        summary.update(token_capture_implemented=True,actual_sampling_ids_captured=True,
            env_token_mask_pass=env_count>0,environment_state_token_count=env_count,
            flat_transcript_replay_supported=all(x['flat_transcript_replay_supported'] for x in flat),
            turn_segment_replay_required=any(not x['flat_transcript_replay_supported'] for x in flat),
            token_context_exact_match_rate=1.0,single_temperature_replay_valid=len(temperatures)==1,
            actual_temperatures=temperatures,group_size=len(captures),group_uid_match=True,
            policy_version_constant_within_group=True,champion_real_rollout_pass=True)
        rngs = [c.records[0]['cuda_rng_sha256'] for c in captures]
        assert len(set(rngs))==args.num_generations
        write(out/'group_audit.json',{'group_uid_match':True,'group_size':len(captures),
            'question_id':question['id'],'group_id':manifest['group_id'],'initial_rng_hashes':rngs,
            'distinct_rng':True,'policy_version_constant_within_group':True,'all_rollouts_before_update':True,
            'rollouts':[{k:v for k,v in r.items() if k!='capture'} for r in state['rollouts']]})
        write(out/'summary.json',summary)
        units = [u for r in state['rollouts'] for u in build_units(ROOT,question,r['capture'],r['trace_path'])]
        write_rows(out/'authoritative_query_units.jsonl',units)
        print(f'Complete group captured; scoring {len(units)} authoritative queries with frozen V3',flush=True)
        scores = score_units(ROOT,units,config['judge'],out/'judge_cache')
        write_rows(out/'process_scores.jsonl',scores)
        rewards = []
        for row in state['rollouts']:
            own = [s['process_score'] for s in scores if s['trajectory_id']==row['trajectory_id']]
            expected = sum(u['trajectory_id']==row['trajectory_id'] for u in units)
            assert len(own) == expected
            value = mini_reward(row['outcome_correct'],own,config['process_weight'])
            value['trajectory_id']=row['trajectory_id']; rewards.append(value)
        write(out/'reward_audit.json',rewards)
        # Reward function consumes authoritative metadata only, not decoded completions.
        callback_output['authoritative_rewards']=[r['reward'] for r in rewards]
        values = metadata_reward(authoritative_rewards=callback_output['authoritative_rewards'])
        print('Full stock TRL GRPO objective forward, no_grad',flush=True)
        objective = trainer.objective_forward(callback_output['turn_segments'],values)
        write(out/'objective_forward.json',objective)
        ownership_audit(model,trainer.optimizer)
        after = parameter_groups(model)
        assert parameter_hash(after['default']) == initial_hash
        assert parameter_hash(after['ref']) == ref_hash
        assert parameter_hash(after['base']) == base_before
        assert not trainer.optimizer.state and all(p.grad is None for p in model.parameters())
        assert sha(adapter_file)==config['adapter_sha256']
        summary.update(process_judge_v3_pass=bool(units),reward_aggregation_pass=True,
            full_grpo_objective_forward_pass=True,grpo_loss_finite=objective['grpo_loss_finite'],
            active_adapter_returns_to_default=objective['active_adapter_returns_to_default'],
            default_ref_output_equal_at_initialization=True,parameter_updated=False,
            reference_unchanged=True,base_unchanged=True,optimizer_state_empty=True,
            peak_vram_gb=torch.cuda.max_memory_allocated()/1e9,
            liger_grpo_compatible=False,liger_status='not runtime-validated; ordinary TRL path used',
            liger_adapter_static_gates={'lm_head_targeted':'lm_head' in adapter_config['target_modules'],
                'prompt_learning':adapter_config['peft_type']!='LORA','target_parameters':adapter_config.get('target_parameters')},
            ready_for_mini_grpo_smoke_training=bool(units) and env_count>0,
            training_loop_status='preflight gates only; train() intentionally blocked pending next authorized stage')
        write(out/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        summary['blocker_type']=type(exc).__name__
        message=str(exc)
        secret=os.environ.get(config.get('judge',{}).get('api_key_env','DMX_API_KEY'),'')
        if secret:
            message=message.replace(secret,'[REDACTED]')
        summary['blocker']=message
        write(out/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        # Traceback locations are useful without echoing arbitrary upstream secrets.
        for frame in traceback.extract_tb(exc.__traceback__):
            print(f'{frame.filename}:{frame.lineno} in {frame.name}',flush=True)
        return 1
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    parser.add_argument('--resume',action='store_true')
    cli=parser.parse_args()
    raise SystemExit(main(cli.config,cli.resume))
