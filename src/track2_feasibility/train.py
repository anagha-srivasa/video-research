"""Training-participant stopping folds and twelve fixed final refits."""
import collections,hashlib,json,logging,os,random,time
import numpy as np
import torch
from torch import nn
from .data import CFG,DATA,REPORT,WORK,Data,normalization,atomic_json
from .models import Dynamics,loss
LOG=logging.getLogger(__name__)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(CFG['cpu_threads_per_worker']);torch.use_deterministic_algorithms(True)

def weight_hash(model):
 h=hashlib.sha256()
 for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def tensors(data,norm):
 x=data.x.copy();x[:,:,:154]=(x[:,:,:154]-np.array(norm['x_mean']))/np.array(norm['x_scale']);y=(data.y-np.array(norm['y_mean']))/np.array(norm['y_scale']);return torch.from_numpy(x.astype(np.float32)),torch.from_numpy(y.astype(np.float32)),torch.from_numpy(data.mask)

def participant_weights(data,indices):
 counts=collections.Counter(data.participants[indices]);return np.array([1/counts[p] for p in data.participants[indices]],np.float32)*len(indices)/len(counts)

def score(model,x,y,mask,indices,data):
 model.compile_inference();values=[]
 with torch.no_grad():
  for participant in sorted(set(data.participants[indices])):
   ix=indices[data.participants[indices]==participant];pred,reg=model(x[ix]);value=loss(pred,y[ix],mask[ix],reg);values.append(float(value))
 return float(np.mean(values))

def fit_one(family,seed,train_ids,stopping_ids=None,epochs=None,tag=''):
 seed_all(seed);data=Data();norm=normalization(data,train_ids);x,y,mask=tensors(data,norm);model=Dynamics(family);initial_hash=weight_hash(model);optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);weights=torch.from_numpy(participant_weights(data,train_ids));history=[];best=float('inf');best_epoch=0;best_state=None;stale=0;rng=np.random.default_rng(seed);started=time.monotonic();limit=epochs or CFG['maximum_epochs']
 for epoch in range(1,limit+1):
  model.train();permutation=rng.permutation(len(train_ids));total=0.;count=0
  for offset in range(0,len(permutation),CFG['batch_size']):
   chosen=permutation[offset:offset+CFG['batch_size']];ids=train_ids[chosen];optimizer.zero_grad(set_to_none=True);pred,reg=model(x[ids]);value=loss(pred,y[ids],mask[ids],reg,weights[chosen])
   if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss '+tag)
   value.backward();gradient=nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_clip'],error_if_nonfinite=True);optimizer.step();total+=float(value.detach())*len(ids);count+=len(ids)
  val=score(model,x,y,mask,stopping_ids,data) if stopping_ids is not None else None
  history.append(dict(epoch=epoch,training_loss=total/count,stopping_loss=val))
  if val is not None:
   if val<best-CFG['minimum_improvement']:
    best=val;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
   else:stale+=1
  else:best_epoch=epoch
  if epoch==1 or epoch%10==0:LOG.info('%s epoch=%d train=%.6f stopping=%s elapsed=%.1fs',tag,epoch,total/count,format(val,'.6f') if val is not None else 'not_used',time.monotonic()-started)
  if stopping_ids is not None and stale>=CFG['patience']:break
  if epoch%10==0:
   from track1_retrain.common import guard
   guard()
 if best_state is not None:model.load_state_dict(best_state)
 model.compile_inference();return model,norm,dict(history=history,best_epoch=best_epoch,completed_epochs=len(history),initial_weights_sha256=initial_hash,final_weights_sha256=weight_hash(model),seconds=time.monotonic()-started,training_participants=norm['training_participants'])

def fold_job(job):
 family,seed,held=job;logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True);path=WORK/'folds'/f'{family}.{seed}.{held}.json'
 if path.exists():return json.loads(path.read_text())
 data=Data();train_ids=data.indices(exclude=held);stopping_ids=data.indices(only=held)
 if len(stopping_ids)==0:raise ValueError('Empty training participant stopping fold')
 model,norm,record=fit_one(family,seed,train_ids,stopping_ids,tag=f'{family}/{seed}/held-{held}');assert held not in norm['training_participants'];record.update(family=family,seed=seed,held_participant=held,normalization=norm);atomic_json(path,record);return record

def final_job(job):
 family,seed,epochs=job;logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True);data=Data();ids=data.indices();model,norm,record=fit_one(family,seed,ids,epochs=epochs,tag=f'{family}/{seed}/final');path=DATA/'models'/f'{family}.{seed}.pt';path.parent.mkdir(exist_ok=True);temp=path.with_suffix('.tmp');torch.save(dict(family=family,seed=seed,selected_epochs=epochs,state_dict=model.state_dict(),normalization=norm,config=CFG,record=record),temp);temp.replace(path);atomic_json(REPORT/f'training.{family}.{seed}.json',record)
 # Reload before any exploratory-validation evaluation; reproduce training predictions exactly.
 ix=data.indices()[:3];x,_,_=tensors(data,norm)
 with torch.no_grad():expected=model(x[ix])[0]
 reloaded=load_model(path)
 with torch.no_grad():actual=reloaded[0](x[ix])[0]
 torch.testing.assert_close(actual,expected,rtol=0,atol=0);record.update(family=family,seed=seed,selected_epochs=epochs,reload_predictions_bitwise_equal=True);atomic_json(REPORT/f'training.{family}.{seed}.json',record);return record

def load_model(path):
 checkpoint=torch.load(path,map_location='cpu',weights_only=True);model=Dynamics(checkpoint['family']);model.load_state_dict(checkpoint['state_dict']);model.compile_inference();return model,checkpoint['normalization'],checkpoint
