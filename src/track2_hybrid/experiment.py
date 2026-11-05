"""Bounded train-only IMM selection and paired source-state evaluation."""
import itertools
import json
import logging
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json
from track1_retrain.common import save
from track2_feasibility.data import Data, BASE
from track2_feasibility.evaluate import prediction_rows, cv
from .imm import filter_history, forecast

CFG = json.loads((ROOT/'configs/track2_hybrid.json').read_text())
REPORT = ROOT/'reports/track2_hybrid'
DATA = ROOT/'derived/track2_hybrid'
WORK = ROOT/'work/track2_hybrid'
GRID = [dict(process_density=q,measurement_multiplier=r,switch_rate=s) for q,r,s in itertools.product(
    CFG['process_density_grid'],CFG['measurement_multiplier_grid'],CFG['switch_rate_per_second_grid'])]


def prepare():
    data = Data()
    episodes = {r['episode_id']:r for r in pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist()}
    histories, support = {}, []
    for video in sorted({s['video_id'] for s in data.samples}):
        rows = pq.read_table(BASE/'spatial'/f'{video}.parquet').to_pylist()
        arrays = [np.load(BASE/'spatial'/f'{video}.{k}16.npy',mmap_mode='r') for k in ('h','variance')]
        tracks = defaultdict(list)
        for r in rows:
            tracks[r['entity_track_id']].append(r)
        for i,s in enumerate(data.samples):
            if s['video_id']!=video: continue
            end=s['input_times_us'][-1]; ep=episodes[s['episode_id']]
            kept=sorted([r for r in tracks[s['entity_track_id']] if max(ep['start_us'],end-1000000)<=r['timestamp_us']<=end
                         and r['fit16_valid'] and r['epoch16']==s['epoch']], key=lambda r:r['timestamp_us'])
            # Duplicate source timestamps cannot be counted as independent measurements.
            seen=set(); kept=[r for r in kept if not (r['timestamp_us'] in seen or seen.add(r['timestamp_us']))]
            history=dict(times=[(r['timestamp_us']-end)/1e6 for r in kept],
                         positions=[arrays[0][r['row_index'],:2].astype(float).tolist() for r in kept],
                         variances=[arrays[1][r['row_index'],:2].astype(float).tolist() for r in kept],
                         mask_uids=[r['mask_uid'] for r in kept])
            reason=None
            try: filter_history(history['times'],history['positions'],history['variances'],0.,GRID[0])
            except ValueError as exc: reason=str(exc)
            support.append(dict(sequence_id=s['sequence_id'],row_index=i,video_id=video,split=s['split'],
                                supported=reason is None,reason=reason,observations=len(kept)))
            if reason is None: histories[str(i)]=history
        for a in arrays: a._mmap.close()
    atomic_json(DATA/'histories.json',histories)
    save(REPORT/'input_support.parquet',support)
    accepted={r['sequence_id'] for r in support if r['supported']}
    ledger=pq.read_table(ROOT/'reports/track2_feasibility/target_ledger.parquet').to_pylist()
    for r in ledger:
        r['original_input_supported']=r['input_supported']; r['original_target_supported']=r['target_supported']
        if r['input_supported'] and r['sequence_id'] not in accepted:
            r.update(input_supported=False,target_supported=False,reason='new_raw_history_unsupported')
    save(REPORT/'target_ledger.parquet',ledger)
    atomic_json(REPORT/'support.json',dict(records=[dict(split=s,original_valid_inputs=sum(r['split']==s for r in support),
        raw_history_supported=sum(r['split']==s and r['supported'] for r in support)) for s in ('train','validation')],
        original_proposals={s:sum(r['split']==s for r in ledger)//3 for s in ('train','validation')},
        input_difference='IMM sees retained raw centroids in the last second; prior neural models see four cached kinematic states',
        copied_rgb_bytes=0,source_labels_retrospective=True,independent_references=False))


def filtered(history,parameters,mode='imm'):
    return filter_history(history['times'],history['positions'],history['variances'],0.,parameters,mode)


def grid_job(job):
    candidate,participant=job
    checkpoint=WORK/f'grid.{candidate}.{participant}.json'
    if checkpoint.exists(): return json.loads(checkpoint.read_text())
    data=Data(); histories=json.loads((DATA/'histories.json').read_text())
    ids=[i for i in data.indices(only=participant,usable=False) if str(i) in histories]
    predictions=[]
    for i in ids:
        state=filtered(histories[str(i)],GRID[candidate])
        predictions.append([forecast(state,h/1e6)[0] for h in CFG['horizons_us']])
    pred=np.asarray(predictions); truth=data.y[ids,:,:4].astype(float); mask=data.mask[ids]
    cp=np.repeat(data.x[ids,-1,140:144].astype(float)[:,None,:],3,axis=1);cp[:,:,2:]=0
    metrics=[]
    for h in range(3):
        if not mask[:,h].any(): raise ValueError('Empty training participant/horizon')
        errors=[]
        for cols in (slice(0,2),slice(2,4)):
            rmse=lambda x:float(np.sqrt(np.mean(np.sum((x[mask[:,h],h,cols]-truth[mask[:,h],h,cols])**2,axis=1))))
            errors.append(rmse(pred)/max(rmse(cp),.001))
        metrics.append(dict(horizon_us=CFG['horizons_us'][h],position_ratio=errors[0],velocity_ratio=errors[1]))
    result=dict(candidate=candidate,participant=participant,parameters=GRID[candidate],metrics=metrics,
                score=float(np.mean([r['position_ratio']+.25*r['velocity_ratio'] for r in metrics])),histories=len(ids))
    atomic_json(checkpoint,result);return result


def select(records):
    scores=np.array([[next(r['score'] for r in records if r['candidate']==c and r['participant']==p)
                      for p in CFG['training_participants']] for c in range(len(GRID))])
    chosen=int(np.argmin(scores.mean(1)))
    folds=[]
    for i,p in enumerate(CFG['training_participants']):
        c=int(np.argmin(np.delete(scores,i,axis=1).mean(1)))
        folds.append(dict(held_participant=p,selected_candidate=c,held_score=float(scores[c,i])))
    atomic_json(REPORT/'selection.json',dict(candidate=chosen,parameters=GRID[chosen],records=records,folds=folds,
        selection_participants=CFG['training_participants'],validation_used=False,score=float(scores[chosen].mean()),
        single_filter_ablations_not_separately_tuned=True))


def evaluate():
    data=Data();histories=json.loads((DATA/'histories.json').read_text())
    parameters=json.loads((REPORT/'selection.json').read_text())['parameters']
    ids=[i for i in data.indices('validation',usable=False) if str(i) in histories]
    outputs={m:cv(data,ids) for m in CFG['primary_methods']}
    states={}; covariances={};weights={}
    for method in ('imm','kalman_cv','kalman_stationary'):
        states[method]={};covariances[method]=[];weights[method]=[]
    for j,i in enumerate(ids):
        h=histories[str(i)]
        outputs['constant_position'][j,:,:2]=data.x[i,-1,140:142];outputs['constant_position'][j,:,2:4]=0
        outputs['nearest_observation'][j,:,:2]=h['positions'][-1];outputs['nearest_observation'][j,:,2:4]=0
        for method in states:
            state=filtered(h,parameters,method)
            states[method][str(i)]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()}
            for hi,horizon in enumerate(CFG['horizons_us']):
                mean,cov,mu=forecast(state,horizon/1e6)
                outputs[method][j,hi,:4]=mean;covariances[method].append(cov.ravel().tolist());weights[method].append(mu.tolist())
    rows=[]
    for method,out in outputs.items():
        local=prediction_rows(data,ids,out,method,0)
        for j,r in enumerate(local):
            r['coarse_squared_error']=None
            r['conditional_covariance']=covariances[method][j] if method in covariances else None
            r['mode_weights']=weights[method][j] if method in weights else None
        rows+=local
    save(REPORT/'predictions.parquet',rows);atomic_json(DATA/'validation_filter_states.json',states)
    return rows


def summarize():
    from track2_assessment import assess
    rows=pq.read_table(REPORT/'predictions.parquet').to_pylist()
    ledger=pq.read_table(REPORT/'target_ledger.parquet').to_pylist()
    paired,per,episodes=assess(rows,ledger,{(m,0) for m in CFG['primary_methods']})
    atomic_json(REPORT/'paired_audit.json',paired);save(REPORT/'episode_metrics.parquet',episodes)
    metrics=[];uncertainty=[]
    for m in CFG['primary_methods']:
        for h in CFG['horizons_us']:
            local=[r for r in rows if r['family']==m and r['horizon_us']==h and r['target_supported']]
            values={metric:float(np.mean([np.sqrt(np.mean([r[metric+'_squared_error'] for r in local if r['participant']==p]))
                        for p in CFG['validation_participants']])) for metric in ('position','velocity')}
            metrics.append(dict(method=m,horizon_us=h,targets=len(local),**values))
            if m in ('imm','kalman_cv','kalman_stationary'):
                participant=[]
                for p in CFG['validation_participants']:
                    nees=[];nll=[]
                    for r in local:
                        if r['participant']!=p:continue
                        c=np.array(r['conditional_covariance']).reshape(4,4)[:2,:2]
                        e=np.array(r['predicted_position'])-r['target_position']; n=float(e@np.linalg.solve(c,e))
                        nees.append(n);nll.append(.5*(n+np.linalg.slogdet(c)[1]+2*np.log(2*np.pi)))
                    participant.append(dict(participant=p,nees=float(np.mean(nees)),joint95=float(np.mean(np.array(nees)<=5.991464547)),nll=float(np.mean(nll))))
                uncertainty.append(dict(method=m,horizon_us=h,participants=participant))
    support=json.loads((REPORT/'support.json').read_text())
    coverage=all(r['raw_history_supported']/r['original_valid_inputs']>=.99 for r in support['records'])
    gates=[]
    for h in CFG['horizons_us']:
        candidate=next(r for r in metrics if r['method']=='imm' and r['horizon_us']==h)
        for b in ('cv','constant_position','nearest_observation'):
            base=next(r for r in metrics if r['method']==b and r['horizon_us']==h)
            gates.append(dict(horizon_us=h,baseline=b,position_ratio=candidate['position']/base['position'],
                velocity_ratio=candidate['velocity']/base['velocity'],
                passed=candidate['position']<=base['position']*(.9 if h<1000000 else 1.05) and candidate['velocity']<=1.05*base['velocity']))
    atomic_json(REPORT/'metrics.json',dict(records=metrics,aggregation='mean of participant RMSE; 2 exposed validation participants'))
    atomic_json(REPORT/'uncertainty.json',dict(records=uncertainty,conditional_moment_gaussian_approximation=True,
        calibrated=False,physical_reference=False,correlated_source_targets=True,chi_square_consistency_certified=False))
    atomic_json(REPORT/'decision.json',dict(gates=gates,input_support_gate=coverage,proxy_feasibility_passed=coverage and all(r['passed'] for r in gates),
        scientific_acceptance=False,full_track2_go=False,baseline_promoted=False,independent_references=False))
