"""Inject controlled targets into the unchanged v6 loader/normalization/stopping path."""
import json
from unittest.mock import patch
import numpy as np
import torch
from scipy.linalg import expm
from pipeline.common import atomic_json,digest
from track1_retrain.common import pool
from track2_feasibility.data import Data
from track2_measurement.anchors import inputs,base
from track2_observability import train as protocol
from track2_observability.model import Dynamics
from .common import C,D,R,W,OLD,validate


def controlled_data(task):
    data=Data();raw=np.load(OLD/'recent_observation.npy');anchor=base(data,raw,'nearest');direction=np.tanh(12*(raw[:,-1,:2]-raw[:,0,:2]));v=.08*direction;a=.16*np.c_[direction[:,1],-direction[:,0]]
    if task=='real':return data
    if task=='shuffled_displacement':
        rng=np.random.default_rng(C['seed']);noise=rng.normal(0,.05,(len(data.samples),3,2))
        # Context-independent targets: no dependency on any feature or future real observation.
        for hi,h in enumerate(C['horizons_seconds']):data.y[:,hi,:2]=anchor[:,hi,:2]+noise[:,hi];data.y[:,hi,2:4]=0
        return data
    g=np.zeros((6,6));g[:2,2:4]=np.eye(2);g[2:4,4:]=np.eye(2);g[4:,4:]=np.array([[-1.,-2.],[2.,-1.]])
    for hi,h in enumerate(C['horizons_seconds']):
        if task=='constant_velocity':delta=h*v;velocity=v
        elif task=='integrated_acceleration':delta=h*v+h*h*a/2;velocity=v+h*a
        elif task=='stable_latent':
            initial=np.c_[np.zeros_like(v),.25*v,a];future=initial@expm(h*g).T;delta=future[:,:2];velocity=future[:,2:4]
        else:raise ValueError('Unknown control')
        data.y[:,hi,:2]=anchor[:,hi,:2]+delta;data.y[:,hi,2:4]=velocity
    return data


def run_job(task):
    kind,held=task;path=W/f'control.{kind}.{held}.json';receipt=path.with_suffix('.receipt.json')
    if receipt.exists():validate(json.loads(receipt.read_text()));return json.loads(path.read_text())
    data=controlled_data(kind);raw=np.load(OLD/'recent_observation.npy');captured={};gradient_rows=[];step_count=0;train_ids=data.indices(exclude=held);ok=np.load(OLD/'input_mask.npy');train_ids=train_ids[ok[train_ids]];batches=int(np.ceil(len(train_ids)/protocol.C['batch_size']))
    original_optimizer=torch.optim.AdamW
    class Instrumented(Dynamics):
        def __init__(self,family):
            super().__init__(family);captured['model']=self;captured['initial_generator']=self.generator().detach().clone();captured['gradient']={}
            for name,p in self.named_parameters():
                group='encoder' if name.startswith('encoder') else ('generator' if name=='A' else ('damping' if name=='damping' else ('acceleration' if name.startswith('acceleration') else 'velocity')))
                def hook(value,group=group):captured['gradient'][group]=captured['gradient'].get(group,0.)+float(value.detach().square().sum());return value
                p.register_hook(hook)
    class Optimizer(original_optimizer):
        def zero_grad(self,*args,**kwargs):captured['gradient']={};return super().zero_grad(*args,**kwargs)
        def step(self,*args,**kwargs):
            nonlocal step_count
            result=super().step(*args,**kwargs);step_count+=1
            if step_count==1 or step_count%batches==0:
                row=dict(step=step_count,epoch=(step_count-1)//batches+1,gradients_before_clipping={k:float(np.sqrt(v)) for k,v in captured['gradient'].items()},generator_change=float(torch.linalg.vector_norm(captured['model'].generator().detach()-captured['initial_generator'])))
                gradient_rows.append(row)
            return result
    with patch.object(protocol,'Data',lambda:data),patch.object(protocol,'Dynamics',Instrumented),patch.object(torch.optim,'AdamW',Optimizer):
        model,record=protocol.fit('anchored_stable',C['seed'],held=held)
    norm=record['normalization'];x=torch.tensor(inputs(data,raw)/norm['x_scale'],dtype=torch.float32);ix=data.indices(only=held);ix=ix[ok[ix]]
    with torch.no_grad():
        selected=model(x[ix]).numpy()*norm['y_scale'];target=data.y[ix,:,:2]-raw[ix,-1:,:2]
        before_a=model.A.detach().clone();before_d=model.damping.detach().clone();model.A.zero_();model.damping.fill_(-3.);reset=model(x[ix]).numpy()*norm['y_scale'];model.A.copy_(before_a);model.damping.copy_(before_d)
    mask=data.mask[ix];nearest=float(np.mean(np.sum(target[mask]**2,1)));learned=float(np.mean(np.sum((selected[:,:,:2]-target)[mask]**2,1)))
    result=dict(task=kind,held=held,record=record,baseline_position_mse=nearest,selected_position_mse=learned,selected_ratio=learned/max(nearest,1e-20),
        selected_generator_effect_rms=float(np.sqrt(np.mean((selected-reset)**2))),gradients=gradient_rows,
        scope='Same actual v6 Data rows, masks, support, train-only normalization, participant weights, optimizer and epoch-zero stopping. Only y labels replaced for synthetic controls; coarse labels unused by this position-only family.')
    atomic_json(path,result);atomic_json(receipt,{str(path.relative_to(D.parents[1])):digest(path)});return result


def run(monitor):
    tasks=[(task,p) for task in C['control_tasks'] for p in C['training_participants']]+[('real',p) for p in C['training_participants']]
    records=pool(run_job,tasks,monitor,workers=C['workers']);summary=[]
    for task in C['control_tasks']+['real']:
        rs=[r for r in records if r['task']==task];summary.append(dict(task=task,folds=len(rs),median_ratio=float(np.median([r['selected_ratio'] for r in rs])),selected_epochs=[r['record']['selected_epochs'] for r in rs],generator_effect_rms=[r['selected_generator_effect_rms'] for r in rs]))
    failures=[r['task'] for r in summary if r['task'] in C['control_tasks'][:-1] and r['median_ratio']>C['control_failure_ratio']]
    atomic_json(R/'optimization_controls.json',dict(summary=summary,positive_control_failure_ratio=C['control_failure_ratio'],positive_controls_requiring_followup=failures,
        raw_records=[str((W/f"control.{r['task']}.{r['held']}.json").relative_to(D.parents[1])) for r in records],
        real_data_new_fits=5,synthetic_control_fits=20,initialization_changed=False,no_generator_identification_claim=True))
