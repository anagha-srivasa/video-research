"""Paired source-state and annotation-position scoring; neither is physical truth."""
import json,time,collections
import numpy as np
import torch
import pyarrow.parquet as pq
from pipeline.common import atomic_json,digest
from track1_retrain.common import save
from track2_feasibility.data import Data
from track2_feasibility.evaluate import prediction_rows,cv
from .data import C,D,R
from .anchors import Model,inputs,base,linear_predict,shrink_predict


def aggregate(rows,common):
    per=[]
    for cohort in ('supported_per_horizon','common_all_horizons','observed_annotation_position'):
        groups=collections.defaultdict(list)
        for r in rows:
            if cohort=='observed_annotation_position':error=r.get('observation_squared_error')
            else:
                if cohort=='common_all_horizons' and r['sequence_id'] not in common:continue
                error=r['position_squared_error']
            if error is not None:groups[r['family'],r['seed'],r['horizon_us'],r['participant']].append(error)
        for (f,s,h,p),values in groups.items():per.append(dict(cohort=cohort,family=f,seed=s,horizon_us=h,participant=p,queries=len(values),position_rmse=float(np.sqrt(np.mean(values)))))
    summary=[]
    for cohort,f,h in sorted({(r['cohort'],r['family'],r['horizon_us']) for r in per}):
        rs=[r for r in per if (r['cohort'],r['family'],r['horizon_us'])==(cohort,f,h)]
        scores=[np.mean([r['position_rmse'] for r in rs if r['seed']==s]) for s in sorted({r['seed'] for r in rs})]
        summary.append(dict(cohort=cohort,family=f,horizon_us=h,median_seed_mean_participant_rmse=float(np.median(scores)),seed_scores=scores,participants=sorted({r['participant'] for r in rs})))
    return per,summary


def evaluate():
    if (R/'results.json').exists():return
    data=Data();raw=np.load(D/'recent_observation.npy');supported=np.load(D/'input_mask.npy');ids=data.indices('validation',usable=False);ids=ids[supported[ids]]
    selection=json.loads((R/'baseline_selection.json').read_text());common={data.samples[i]['sequence_id'] for i in ids if data.mask[i].all()}
    predictions={'cached_cv':base(data,raw,'cv'),'nearest':base(data,raw,'nearest'),'raw_anchor_cv':shrink_predict(data,raw,beta=0),
        'shrinkage':shrink_predict(data,raw,alpha=selection['shrinkage']),'decay':shrink_predict(data,raw,beta=selection['selected']['decay']),
        'ridge':linear_predict(data,raw,selection['normalization'],np.load(D/'ridge.npy'))}
    cp=base(data,raw,'nearest');cp[:,:,:2]=data.x[:,-1:,140:142];predictions['cached_position']=cp
    rows=[];latencies=[]
    files=list(D.glob('*.pt'));atomic_json(R/'evaluation_lock.json',dict(checkpoints={p.name:digest(p) for p in files},previously_exposed_validation=True))
    def add(f,s,p):
        carrier=cv(data,ids);carrier[:,:,:4]=p
        new=prediction_rows(data,ids,carrier,f,s)
        for r in new:r['coarse_squared_error']=None
        rows.extend(new)
    for f,p in predictions.items():add(f,0,p[ids])
    for f in C['families']:
        for s in C['seeds']:
            checkpoint=torch.load(D/f'{f}.{s}.pt',weights_only=True,map_location='cpu');norm=checkpoint['record']['normalization'];model=Model();model.load_state_dict(checkpoint['state']);model.eval()
            x=torch.tensor(inputs(data,raw)/norm['x_scale'],dtype=torch.float32)
            with torch.no_grad():p=base(data,raw,f.split('_')[0])[ids]+model(x[ids]).numpy()*norm['y_scale']
            add(f,s,p);elapsed=[];i=data.indices()[0]
            for iteration in range(110):
                start=time.perf_counter()
                # Cached history normalization, network and denormalization; upstream encoding excluded.
                sample=torch.tensor(inputs(data,raw)[i:i+1]/norm['x_scale'],dtype=torch.float32)
                with torch.no_grad():out=model(sample).numpy()*norm['y_scale']+base(data,raw,f.split('_')[0])[i:i+1]
                if iteration>=10:elapsed.append((time.perf_counter()-start)*1000)
            latencies.append(dict(family=f,seed=s,median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),note='CPU cached-history path; includes current whole-array feature helper overhead; excludes RGB and Track 1'))
    observations={(r['sequence_id'],r['horizon_us']):r for r in pq.read_table(R/'target_audit.parquet').to_pylist()}
    for r in rows:
        o=observations[r['sequence_id'],r['horizon_us']];r['observation_squared_error']=float(np.sum((np.array(r['predicted_position'])-o['observation_position'])**2)) if o['observation_target_available'] else None
        r['source_topology_transition']=o['source_topology_transition'];r['observation_source_kind']=o['observation_source_kind'];r['actual_observation_target_us']=o['actual_target_us']
    save(R/'predictions.parquet',rows);per,summary=aggregate(rows,common);save(R/'per_participant.parquet',per)
    # Report rejected-state annotation scoring separately, retaining every emitted prediction.
    strata=[]
    for f in sorted({r['family'] for r in rows}):
        for h in C['horizons_us']:
            for accepted in (False,True):
                rs=[r for r in rows if r['family']==f and r['horizon_us']==h and r['target_supported']==accepted and r['observation_squared_error'] is not None]
                vals=[]
                for s,p in sorted({(r['seed'],r['participant']) for r in rs}):
                    group=[r for r in rs if r['seed']==s and r['participant']==p];vals.append(dict(seed=s,participant=p,queries=len(group),rmse=float(np.sqrt(np.mean([r['observation_squared_error'] for r in group])))))
                strata.append(dict(family=f,horizon_us=h,source_state_accepted=accepted,groups=vals))
    lookup={(r['family'],r['horizon_us']):r['median_seed_mean_participant_rmse'] for r in summary if r['cohort']=='supported_per_horizon'}
    gates={f:all(lookup[f,h]<=lookup['nearest',h]*(.95 if h<1000000 else 1.05) for h in C['horizons_us']) for f in C['families']+['ridge','decay','shrinkage']}
    atomic_json(R/'results.json',dict(summary=summary,validation_histories=len(ids),common_all_horizon_histories=len(common),position_gates=gates,
        rejected_target_strata=strata,scientific_acceptance=False,full_track2_go=False,independent_references=0,
        velocity_position_only='Velocity residual head is untrained and remains the chosen anchor velocity. Do not interpret as a physical velocity model.',
        annotation_positions='Retrospective source-mask centroids, not independent physical targets. Rejected-state strata are descriptive and not training labels.'))
    atomic_json(R/'latency.json',latencies)
