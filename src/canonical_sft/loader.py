import json,re
from pathlib import Path
import torch
from torch.utils.data import Dataset
class CanonicalSFTDataset(Dataset):
 def __init__(self,path): self.rows=[json.loads(x) for x in Path(path).open() if x.strip()]
 def __len__(self): return len(self.rows)
 def __getitem__(self,i): return self.rows[i]
class RoleAwareCollator:
 def __init__(self,tokenizer,max_length=16384,truncation_mode='error',supervision_mode='assistant_only'):
  if truncation_mode not in ('error','cutoff'): raise ValueError('truncation_mode must be error or cutoff')
  if supervision_mode not in ('assistant_only','trajectory_response'): raise ValueError('invalid supervision_mode')
  self.tok=tokenizer; self.max_length=max_length; self.truncation_mode=truncation_mode; self.supervision_mode=supervision_mode
 def __call__(self,batch):
  seqs=[]; labs=[]; original_lengths=[]
  for row in batch:
   ms=row['messages']; text=self.tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); enc=self.tok(text,add_special_tokens=False,return_offsets_mapping=True); ids=enc['input_ids']; offs=enc.get('offset_mapping',[])
   original_length=len(ids)
   lab=[-100]*len(ids)
   if self.supervision_mode=='trajectory_response':
    first=re.search(r'<\|im_start\|>assistant\n',text)
    if not first: raise ValueError('canonical trajectory has no assistant response')
    # Mask only the initial prompt. The full canonical response suffix,
    # including real tool observations, remains supervised.
    start=first.end()
    for i,(x,y) in enumerate(offs):
     if y>start: lab[i]=ids[i]
   else:
    # Legacy environment-masked view: supervise assistant block interiors.
    for m in ms:
     if m['role']!='assistant': continue
     for block in re.finditer(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',text,re.S):
      a,b=block.span(1)
      for i,(x,y) in enumerate(offs):
       if x<b and y>a: lab[i]=ids[i]
     break
   if original_length>self.max_length:
    if self.truncation_mode=='error': raise ValueError(f'silent truncation blocked: {original_length} > {self.max_length}')
    ids=ids[:self.max_length]; lab=lab[:self.max_length]
   if not any(token!=-100 for token in lab): raise ValueError('sample has zero supervised tokens after collation/cutoff')
   original_lengths.append(original_length)
   seqs.append(ids); labs.append(lab)
  n=max(map(len,seqs)); pad=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
  x=torch.full((len(batch),n),pad,dtype=torch.long); y=torch.full((len(batch),n),-100,dtype=torch.long); a=torch.zeros((len(batch),n),dtype=torch.long)
  for i,(ids,lab) in enumerate(zip(seqs,labs)): x[i,:len(ids)]=torch.tensor(ids); y[i,:len(ids)]=torch.tensor(lab); a[i,:len(ids)]=1
  original=torch.tensor(original_lengths,dtype=torch.long); effective=torch.tensor([len(ids) for ids in seqs],dtype=torch.long)
  return {'input_ids':x,'attention_mask':a,'labels':y,'original_lengths':original,'effective_lengths':effective,'truncated':original>effective,'supervised_tokens':(y!=-100).sum(dim=1),'masked_tokens':(y==-100).sum(dim=1)}
