import copy
import pytest
import torch

from online_grpo.capture import replay_segment, audit_segment, audit_flat, digest
from online_grpo.reward import mini_reward, metadata_reward
from online_grpo.adapter import ownership_audit, group_advantages


def sample_record():
    state, action = [10,11,12], [20,21]
    return {'trajectory_id':'t','generation_idx':0,'round_idx':1,'input_ids':state,
        'generated_ids':action,'actual_input_sha256':digest(state),'actual_generated_sha256':digest(action),
        'generated_length':len(action),'generation_temperature':.4,
        'input_token_origins':['PROMPT','ENVIRONMENT_OBSERVATION','RUNTIME_INJECTED']}


def test_exact_context_and_masks():
    segment=replay_segment(sample_record())
    assert audit_segment(segment)['environment_mask_pass']
    assert segment['attention_mask']==[1]*5
    assert segment['loss_mask']==[0,0,0,1,1]


@pytest.mark.parametrize('field',['state_ids','action_ids','input_ids','loss_mask','attention_mask'])
def test_fail_closed_on_replay_mutations(field):
    segment=replay_segment(sample_record())
    segment[field][0] += 1
    with pytest.raises(ValueError): audit_segment(segment)


def test_eos_or_newline_not_ignored():
    a=sample_record(); b=copy.deepcopy(a)
    b['input_ids']=[10,11,12,20,99,21]
    assert not audit_flat([a,b])['flat_transcript_replay_supported']


def test_reward_boundaries():
    assert mini_reward(False,[],.25)['process_quality']==0
    assert mini_reward(True,[0],.25)['reward'] > mini_reward(False,[1],.25)['reward']
    with pytest.raises(ValueError): mini_reward(False,[None],.25)
    with pytest.raises(ValueError): metadata_reward(completions=['fake textual search'])
    with pytest.raises(ValueError): mini_reward(True,[1],1)
    assert metadata_reward(authoritative_rewards=[0,.25])==[0,.25]


def test_group_scaling():
    values,audit=group_advantages([0,.25])
    assert values[0]<0<values[1]
    assert sum(values)==0
    assert group_advantages([0,0])[0]==[0,0]


def test_optimizer_excludes_base_and_ref():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora=torch.nn.ModuleDict({key:torch.nn.Linear(2,2,bias=False) for key in ['default','ref']})
            self.base=torch.nn.Linear(2,2,bias=False)
            self.requires_grad_(False)
            self.lora['default'].requires_grad_(True)
    model=Model()
    optimizer=torch.optim.AdamW(model.lora['default'].parameters())
    assert ownership_audit(model,optimizer)['optimizer_owns_default_only']
    wrong=torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError): ownership_audit(model,wrong)


def test_capture_hook_is_opt_in_and_text_preserving():
    from champion_runtime.local_policy import HFPolicyBackend, HFPolicyConfig
    class Batch(dict):
        def to(self,device): return self
    class Tokenizer:
        eos_token_id=0
        def apply_chat_template(self,messages,**kwargs): return 'prompt'
        def __call__(self,prompt,**kwargs): return Batch(input_ids=torch.tensor([[1,2]]))
        def decode(self,ids,**kwargs): return ' raw text \n'
    class Model(torch.nn.Module):
        def generate(self,input_ids,**kwargs):
            return torch.cat([input_ids,torch.tensor([[3,4]])],dim=1)
    class Sink:
        def before_generation(self,batch,prompt,messages,kwargs,model): self.prefix=batch['input_ids'].tolist()
        def after_generation(self,ids,raw,accepted): self.result=(ids.tolist(),raw,accepted)
    policy=HFPolicyBackend.from_model(Model(),Tokenizer(),HFPolicyConfig('loaded://base','loaded://tok',device='cpu'))
    plain=policy._generate_sync([{'role':'user','content':'test'}],.4,8)
    sink=Sink();policy.capture_sink=sink
    captured=policy._generate_sync([{'role':'user','content':'test'}],.4,8)
    assert plain==captured=='raw text'
    assert sink.prefix==[[1,2]] and sink.result==([3,4],' raw text \n','raw text')
