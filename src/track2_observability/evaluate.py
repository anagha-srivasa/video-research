import json,time,collections
import numpy as np
import torch
from pipeline.common import atomic_json
from track1_retrain.common import save
from track2_feasibility.data import Data
from track2_measurement.anchors import Model,inputs,base,shrink_predict
from track2_feasibility.evaluate import prediction_rows,cv
from .model import Dynamics
from .common import C,D,R,OLD


def summarize(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['target_supported']:groups[r['family'],r['seed'],r['horizon_us'],r['participant']].append(r['position_squared_error'])
    per=[dict(family=f,seed=s,horizon_us=h,participant=p,queries=len(v),position_rmse=float(np.sqrt(np.mean(v)))) for (f,s,h,p),v in groups.items()]
    result=[]
    for f,h in sorted({(r['family'],r['horizon_us']) for r in per}):
        rs=[r for r in per if r['family']==f and r['horizon_us']==h];values=[np.mean([r['position_rmse'] for r in rs if r['seed']==s]) for s in sorted({r['seed'] for r in rs})]
        result.append(dict(family=f,horizon_us=h,median_seed_mean_participant_rmse=float(np.median(values)),seed_scores=values))
    return per,result


def evaluate():
    data=Data();raw=np.load(OLD/'recent_observation.npy');ok=np.load(OLD/'input_mask.npy');val=data.indices('validation',usable=False);val=val[ok[val]];rows=[];latency=[]
    def add(f,s,pred):
        carrier=cv(data,val);carrier[:,:,:4]=pred
        records=prediction_rows(data,val,carrier,f,s)
        for r in records:r['coarse_squared_error']=None
        rows.extend(records)
    nearest=base(data,raw,'nearest');add('nearest',0,nearest[val])
    old=json.loads((R.parents[0]/'track2_measurement/baseline_selection.json').read_text());add('shrinkage',0,shrink_predict(data,raw,alpha=old['shrinkage'])[val])
    for f in C['families']:
        for seed in C['seeds']:
            cp=torch.load(D/f'{f}.{seed}.pt',map_location='cpu',weights_only=True);norm=cp['record']['normalization'];structured=f.startswith('anchored_');model=Dynamics(f) if structured else Model();model.load_state_dict(cp['state']);model.eval()
            x=torch.tensor(inputs(data,raw)/norm['x_scale'],dtype=torch.float32);anchor=base(data,raw,'nearest' if structured else f.split('_')[0])
            with torch.no_grad():pred=anchor[val]+model(x[val]).numpy()*norm['y_scale']
            add(f,seed,pred);times=[]
            sample=x[data.indices()[0]:data.indices()[0]+1]
            for repeat in range(35):
                start=time.perf_counter()
                with torch.no_grad():model(sample)
                if repeat>=5:times.append((time.perf_counter()-start)*1000)
            latency.append(dict(family=f,seed=seed,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),scope='CPU model-only cached inputs; excludes RGB/Track1, preprocessing and denormalization'))
    # Thresholds are set from training target displacements only; names deliberately avoid physical-event claims.
    train=data.indices();thresholds=[]
    for hi,h in enumerate(C['horizons_seconds']):
        ix=train[data.mask[train,hi]];d=np.linalg.norm(data.y[ix,hi,:2]-raw[ix,-1,:2],axis=1);thresholds.append(np.quantile(d,C['source_motion_quantiles']).tolist())
    index={s['sequence_id']:i for i,s in enumerate(data.samples)};episodes=collections.defaultdict(list)
    for r in rows:
        i=index[r['sequence_id']];hi=C['horizons_seconds'].index(r['horizon_us']/1e6)
        if r['target_supported']:
            actual=np.array(r['target_position'])-raw[i,-1,:2];pred=np.array(r['predicted_position'])-raw[i,-1,:2];magnitude=float(np.linalg.norm(actual));estimated=float(np.linalg.norm(pred));q=thresholds[hi]
            r['source_displacement_stratum']='low' if magnitude<=q[0] else ('high' if magnitude>q[1] else 'middle')
            r['target_displacement']=magnitude;r['predicted_displacement']=estimated;r['direction_cosine']=float(actual@pred/(magnitude*estimated)) if min(magnitude,estimated)>1e-9 else None
            episodes[r['family'],r['seed'],r['horizon_us'],r['participant'],r['episode_id'],r['source_displacement_stratum']].append(r)
        else:r.update(source_displacement_stratum=None,target_displacement=None,predicted_displacement=None,direction_cosine=None)
    ep=[]
    for (f,s,h,p,e,stratum),rs in episodes.items():
        directions=[r['direction_cosine'] for r in rs if r['direction_cosine'] is not None]
        ep.append(dict(family=f,seed=s,horizon_us=h,participant=p,episode_id=e,source_displacement_stratum=stratum,queries=len(rs),position_rmse=float(np.sqrt(np.mean([r['position_squared_error'] for r in rs]))),direction_scored=len(directions),direction_cosine=float(np.mean(directions)) if directions else None))
    save(R/'predictions.parquet',rows);save(R/'episode_displacement.parquet',ep);per,summary=summarize(rows);save(R/'per_participant.parquet',per)
    common={data.samples[i]['sequence_id'] for i in val if data.mask[i].all()};_,common_summary=summarize([r for r in rows if r['sequence_id'] in common])
    atomic_json(R/'results.json',dict(summary=summary,common_all_horizon=common_summary,validation_histories=len(val),common_histories=len(common),supported_targets=data.mask[val].sum(0).tolist(),thresholds=thresholds,physical_regimes='Unknown: quiet, sustained, onset, reversal, occlusion, interaction have no independent labels.',promotion=False,independent_references=0))
    atomic_json(R/'latency.json',latency)
