from dataclasses import dataclass,field
from pathlib import Path
import torch
from .loader import CanonicalSFTDataset,RoleAwareCollator
@dataclass
class SFTConfig:
 model_path:str; tokenizer_path:str; train_dataset_path:str; mode:str='lora'; output_dir:str='artifacts/sft'; max_seq_length:int=16384; truncation_mode:str='error'; supervision_mode:str='assistant_only'; per_device_train_batch_size:int=1; gradient_accumulation_steps:int=1; learning_rate:float=2e-5; weight_decay:float=0.0; max_grad_norm:float=1.0; num_train_epochs:float=1.0; max_steps:int=-1; lr_scheduler_type:str='constant'; warmup_ratio:float=0.0; scheduler_total_steps:int|None=None; gradient_checkpointing:bool=False; use_cache:bool=False; dtype:str='bfloat16'; attention_backend:str='sdpa'; loss_implementation:str='sparse_supervised'; lora_r:int=16; lora_alpha:int=32; lora_dropout:float=0.05; lora_target_modules:list[str]=field(default_factory=lambda:['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']); report_to:list[str]=field(default_factory=list); wandb_project:str='agentic-search-rl'; wandb_run_name:str=''; wandb_job_type:str='sft'; wandb_tags:list[str]=field(default_factory=list); wandb_mode:str='offline'; wandb_api_key_env:str='WANDB_API_KEY'; seed:int=20260905
 def __post_init__(self):
  if self.mode not in ('lora','full'): raise ValueError('mode must be lora or full')
  if self.truncation_mode not in ('error','cutoff'): raise ValueError('truncation_mode must be error or cutoff')
  if self.supervision_mode not in ('assistant_only','trajectory_response'): raise ValueError('invalid supervision_mode')
  if self.loss_implementation not in ('standard','sparse_supervised','liger_sparse_supervised'): raise ValueError('invalid loss_implementation')
  if self.lr_scheduler_type not in ('constant','cosine'): raise ValueError('lr_scheduler_type must be constant or cosine')
  if not 0.0<=self.warmup_ratio<1.0: raise ValueError('warmup_ratio must be in [0,1)')
  if self.max_seq_length<=0 or self.per_device_train_batch_size<=0: raise ValueError('invalid sizing')
def load_tokenizer(cfg):
 from transformers import AutoTokenizer
 return AutoTokenizer.from_pretrained(cfg.tokenizer_path,local_files_only=True,trust_remote_code=True)
def load_model(cfg):
 from transformers import AutoConfig,AutoModelForCausalLM
 config=AutoConfig.from_pretrained(cfg.model_path,local_files_only=True,trust_remote_code=True)
 dtype=getattr(torch,cfg.dtype) if cfg.dtype else None
 return AutoModelForCausalLM.from_pretrained(cfg.model_path,config=config,local_files_only=True,torch_dtype=dtype,attn_implementation=cfg.attention_backend,low_cpu_mem_usage=True)
def prepare_model(model,cfg):
 if cfg.mode=='full':
  for p in model.parameters(): p.requires_grad=True
  return model
 from peft import LoraConfig,get_peft_model
 lora=LoraConfig(r=cfg.lora_r,lora_alpha=cfg.lora_alpha,lora_dropout=cfg.lora_dropout,target_modules=cfg.lora_target_modules,bias='none',task_type='CAUSAL_LM')
 model=get_peft_model(model,lora)
 if cfg.gradient_checkpointing:
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
 model.config.use_cache=cfg.use_cache
 return model
def build_dataloader(cfg,tokenizer):
 from torch.utils.data import DataLoader
 return DataLoader(CanonicalSFTDataset(cfg.train_dataset_path),batch_size=cfg.per_device_train_batch_size,shuffle=False,collate_fn=RoleAwareCollator(tokenizer,cfg.max_seq_length,cfg.truncation_mode,cfg.supervision_mode))
def parameter_state(model):
 total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); return {'total':total,'trainable':trainable,'frozen':total-trainable}
def build_optimizer(model,cfg):
 params=[p for p in model.parameters() if p.requires_grad]
 if not params: raise RuntimeError('no trainable parameters')
 return torch.optim.AdamW(params,lr=cfg.learning_rate,weight_decay=cfg.weight_decay)

def wandb_settings(cfg):
 import os
 enabled='wandb' in cfg.report_to
 return {'enabled':enabled,'project':cfg.wandb_project,'run_name':cfg.wandb_run_name,'mode':cfg.wandb_mode,'api_key_configured':bool(os.getenv(cfg.wandb_api_key_env)),'api_key_env':cfg.wandb_api_key_env}
