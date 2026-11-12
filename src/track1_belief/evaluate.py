"""Matched support ledgers, analytic stress tests and source-proxy comparisons."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import time
import numpy as np

from .common import C, D, R, LOG, arrays, save, atomic_json
from .estimation import FilterConfig, process
from .memory import transition


METHODS = ('nearest', 'cv', 'independent', 'dependent', 'adaptive')


def configuration(q=10000., threshold=5., method='adaptive'):
    return FilterConfig(acceleration_spectral_density=float(q),
                        jump_spectral_density=float(q)*100,
                        shared_bias_variance=0., initial_velocity_variance=10000.,
                        maximum_gap=.15, minimum_span=.08,
                        robust_measurement=method=='adaptive',
                        process_mode='coherent_jump' if method=='adaptive' else 'gaussian',
                        jump_nis_threshold=float(threshold)**2,
                        event_speed_threshold=100., memory_timescales=(.125,.5,2.))


def estimate_episode(episode, source, method, parameters):
    t = np.asarray(episode['time'], float)
    position = np.asarray(episode[source+'_positions'], float)
    valid = np.asarray(episode['valid'] if source=='anchor' else episode['centroid_valid'], bool)
    valid &= np.isfinite(position).all(1)
    ids = np.flatnonzero(valid)
    n = len(t)
    result = dict(mean=np.zeros((n,4)), covariance=np.zeros((n,4,4)),
                  memory_mean=np.zeros((n,3,2)), joint_covariance=np.zeros((n,10,10)),
                  valid=np.zeros(n,bool), event=np.zeros(n,bool), process_jump=np.zeros(n,bool),
                  latest_evidence_time=np.full(n,np.nan), observation_valid=valid.copy())
    if not len(ids):
        return result
    local = np.asarray(episode['covariance_local'] if source=='anchor' else episode.get('centroid_covariance', np.tile(np.eye(2)*4,(n,1,1))),float)[ids]
    shared = np.asarray(episode['covariance_shared'],float)[ids]
    if source=='centroid':
        # Retrospective annotation dependence is a declared sensitivity model,
        # not estimated physical-label uncertainty.
        shared = np.tile(np.eye(2)*parameters.get('source_bias_variance',4.),(len(ids),1,1))
    if method in ('nearest','cv'):
        for j,i in enumerate(ids):
            past = ids[(t[ids]<=t[i])&(t[ids]>=t[i]-.2)]
            if len(past)<3 or t[past[-1]]-t[past[0]]<.06:
                continue
            if method=='nearest':
                result['mean'][i,:2]=position[i]
                result['covariance'][i,:2,:2]=local[j]+shared[j]
            else:
                design=np.c_[np.ones(len(past)),t[past]-t[i]]
                beta=np.linalg.lstsq(design,position[past],rcond=None)[0]
                result['mean'][i]=beta.ravel()
                sigma=np.mean(np.sum((position[past]-design@beta)**2,axis=1))/2
                result['covariance'][i]=np.kron(np.linalg.inv(design.T@design),np.eye(2)*max(sigma,1.))+np.diag([shared[j,0,0],shared[j,1,1],0,0])
            result['valid'][i]=True
            result['latest_evidence_time'][i]=t[i]
        speed=np.linalg.norm(result['mean'][:,2:4],axis=1)
        active=(speed>=100)&result['valid']
        result['event']=active&~np.r_[False,active[:-1]]
        return result
    cfg=configuration(parameters['q'],parameters.get('threshold',5.),method)
    ancestry=(np.asarray(episode.get('latest_evidence_us',episode['timestamp_us']),np.int64)[ids]-episode['timestamp_us'][0])/1e6
    if episode.get('reference_kind')=='analytic_exact':
        ancestry=t[ids].copy()
    if method=='independent':
        local=local+shared
        shared=np.zeros_like(shared)
    groups=np.asarray(episode.get('groups',np.full(n,'seed')),str)[ids]
    coherence=np.asarray(episode.get('coherence',np.zeros(n)),float)[ids]>=.7 if source=='anchor' else np.zeros(len(ids),bool)
    start=time.perf_counter()
    out=process(t[ids],position[ids],local,groups,cfg,
                ancestor_max_time=ancestry if source=='anchor' else None,
                mode='strict_causal' if source=='anchor' else 'retrospective',
                shared_covariance=shared, coherence=coherence)
    result['filter_seconds']=time.perf_counter()-start
    for key in ('mean','covariance','memory_mean','joint_covariance','valid','event','process_jump','latest_evidence_time'):
        result[key][ids]=out[key]
    if method!='adaptive':
        active=(np.linalg.norm(result['mean'][:,2:4],axis=1)>=100)&result['valid']
        result['event']=active&~np.r_[False,active[:-1]]
    return result


def query_rows(episode, states, reference_key='proxy_positions'):
    """Every proposed query is retained, including unsupported comparisons."""
    t=np.asarray(episode['time']);truth=np.asarray(episode[reference_key]);target_valid=np.asarray(episode.get('proxy_valid',np.isfinite(truth).all(1)),bool)
    cutoffs=np.arange(t[0]+1., t[-1]-.24, .25)
    indices=np.unique(np.searchsorted(t,cutoffs,side='right')-1)
    rows=[]
    for i in indices:
        for horizon in (0.,*C['horizons_seconds']):
            j=int(np.argmin(abs(t-(t[i]+horizon))))
            target_ok=bool(target_valid[j] and abs(t[j]-t[i]-horizon)<=.025 and (horizon==0 or j>i))
            actual=float(t[j]-t[i])
            for (source,method),state in states.items():
                supported=bool(state['valid'][i] and target_ok)
                row=dict(episode_id=episode['episode_id'],participant=episode['participant'],split=episode['split'],
                         source=source,method=method,cutoff_index=int(i),target_index=j,horizon=horizon,
                         actual_horizon=actual,supported=supported,target_available=target_ok,
                         input_supported=bool(state['valid'][i]),reference_kind=episode['reference_kind'],
                         reason=None if supported else 'target_unavailable' if not target_ok else 'input_unsupported',
                         latest_input_time=float(state['latest_evidence_time'][i]) if np.isfinite(state['latest_evidence_time'][i]) else None,
                         cutoff_time=float(t[i]),error_squared=None,nees=None,covered95=None,radius95=None)
                if supported:
                    h=np.c_[np.eye(2),np.eye(2)*actual]
                    prediction=h@state['mean'][i]
                    covariance=h@state['covariance'][i]@h.T+np.eye(2)*1e-8
                    if method in ('independent','dependent','adaptive'):
                        _,q=transition(actual,(),state.get('q',10000.))
                        covariance+=q[:2,:2]
                    error=prediction-truth[j]
                    nees=float(error@np.linalg.solve(covariance,error))
                    row.update(error_squared=float(error@error),prediction=prediction.tolist(),target=truth[j].tolist(),
                               nees=nees,covered95=nees<=5.991464547,radius95=float(np.sqrt(5.991464547*np.linalg.eigvalsh(covariance).max())))
                    if 'velocity' in episode and horizon==0:
                        row['velocity_error_squared']=float(np.sum((state['mean'][i,2:4]-episode['velocity'][i])**2))
                rows.append(row)
    return rows


def event_score(episode,state):
    t=episode['time'];valid=state['valid']&(t>=1.)
    speed=np.linalg.norm(episode['velocity'],axis=1)
    true_onsets=t[(speed>=100)&~np.r_[False,speed[:-1]>=100]] if episode['kind'] in ('pickup','burst') else np.array([])
    proposed=t[state['event']&valid];detected=[]
    for value in proposed:
        if not detected or value-detected[-1]>=.2:
            detected.append(value)
    used=set();delays=[]
    for onset in true_onsets:
        candidates=[(j,x) for j,x in enumerate(detected) if j not in used and onset-.02<=x<=onset+.25]
        if candidates:
            j,value=candidates[0];used.add(j);delays.append(float(value-onset))
    quiet=(speed<5)&valid;dt=np.r_[0,np.diff(t)]
    false=sum(j not in used and quiet[np.argmin(abs(t-x))] for j,x in enumerate(detected))
    return dict(events=len(true_onsets),recalled=len(delays),delays=delays,
                quiet_seconds=float(dt[quiet].sum()),false_alarms=int(false),
                position_mse=float(np.mean(np.sum((state['mean'][valid,:2]-episode['truth'][valid])**2,axis=1))) if valid.any() else None,
                velocity_mse=float(np.mean(np.sum((state['mean'][valid,2:4]-episode['velocity'][valid])**2,axis=1))) if valid.any() else None,
                supported_fraction=float(valid.sum()/max(1,(t>=1).sum())))


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['split'],row['source'],row['method'],row['horizon'])].append(row)
    result=[]
    for (part,source,method,h),local in sorted(groups.items()):
        supported=[r for r in local if r['supported']]
        by_part=defaultdict(list)
        for r in supported:by_part[r['participant']].append(r)
        result.append(dict(split=part,source=source,method=method,horizon=h,proposed=len(local),supported=len(supported),
                    participants=len(by_part),mean_participant_rmse=float(np.mean([np.sqrt(np.mean([x['error_squared'] for x in values])) for values in by_part.values()])) if by_part else None,
                    coverage95=float(np.mean([x['covered95'] for x in supported])) if supported else None,
                    mean_radius95=float(np.mean([x['radius95'] for x in supported])) if supported else None,
                    reference_kind=local[0]['reference_kind']))
    return result


def common_support(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['episode_id'],r['cutoff_index'],r['horizon'])].append(r)
    return [r for local in groups.values() if all(x['supported'] for x in local) for r in local]


def synthetic_trial(job):
    from .synthetic import episode
    seed,q,threshold=job;e=episode(seed)
    state=estimate_episode(e,'anchor','adaptive',dict(q=q,threshold=threshold));state['q']=q
    rows=query_rows(e,{('anchor','adaptive'):state},'truth')
    event=event_score(e,state)
    mse=[r['error_squared'] for r in rows if r['supported'] and r['horizon']>0]
    return dict(seed=seed,q=q,threshold=threshold,continuation_mse=float(np.mean(mse)),**event)


def select_synthetic(monitor,pool):
    trials=[(seed,q,threshold) for q in C['process_spectral_density_grid'] for threshold in C['event_threshold_grid']
            for seed in range(C['synthetic_training_episodes'])]
    LOG.info('Synthetic training-only selection: %d episode fits',len(trials))
    rows=pool(synthetic_trial,trials,monitor,workers=C['workers']);save(R/'synthetic_training.parquet',rows)
    candidates=[]
    for q in C['process_spectral_density_grid']:
        for threshold in C['event_threshold_grid']:
            local=[r for r in rows if r['q']==q and r['threshold']==threshold]
            events=sum(r['events'] for r in local);quiet=sum(r['quiet_seconds'] for r in local)
            candidates.append(dict(q=q,threshold=threshold,continuation_rmse=float(np.sqrt(np.mean([r['continuation_mse'] for r in local]))),
                              velocity_rmse=float(np.sqrt(np.mean([r['velocity_mse'] for r in local]))),
                              recall=sum(r['recalled'] for r in local)/events if events else 0.,
                              false_per_quiet_minute=60*sum(r['false_alarms'] for r in local)/quiet))
    # Explicit multi-objective score in physical units, fixed before validation.
    for r in candidates:r['objective']=r['continuation_rmse']+20*(1-r['recall'])+2*r['false_per_quiet_minute']
    selected=min(candidates,key=lambda r:(r['objective'],r['q'],r['threshold']))
    result=dict(parameters={k:selected[k] for k in ('q','threshold')},candidates=candidates,
                objective='continuation_RMSE_px + 20*(1-recall) + 2*false_onsets_per_quiet_minute',
                selected_on='synthetic training seeds only',validation_used=False,independent_real_calibration=False)
    atomic_json(R/'selection.json',result)
    return result['parameters']


def evaluate_synthetic(parameters,monitor,pool):
    jobs=[(seed,parameters) for seed in range(C['synthetic_training_episodes'],C['synthetic_training_episodes']+C['synthetic_validation_episodes'])]
    results=pool(synthetic_final,jobs,monitor,workers=C['workers'])
    rows=[r for rr,_ in results for r in rr];events=[r for _,ee in results for r in ee]
    save(R/'synthetic_predictions.parquet',rows);save(R/'synthetic_events.parquet',events)
    atomic_json(R/'synthetic.json',dict(summary=summarize(rows),matched_summary=summarize(common_support(rows)),events=aggregate_events(events),
                exact_truth=True,heldout_seed_count=len(jobs),independent_real_video_acceptance=False))


def synthetic_final(job):
    from .synthetic import episode
    seed,parameters=job;e=episode(seed);states={};events=[]
    for source in ('centroid','anchor'):
        for method in METHODS:
            state=estimate_episode(e,source,method,parameters);state['q']=parameters['q'];states[source,method]=state
            events.append(dict(seed=seed,kind=e['kind'],corruption=e['corruption'],source=source,method=method,**event_score(e,state)))
    return query_rows(e,states,'truth'),events


def aggregate_events(rows):
    output=[]
    for source in ('centroid','anchor'):
        for method in METHODS:
            local=[r for r in rows if r['source']==source and r['method']==method]
            if not local:continue
            n=sum(r['events'] for r in local);quiet=sum(r['quiet_seconds'] for r in local);delay=[d for r in local for d in r['delays']]
            output.append(dict(source=source,method=method,episodes=len(local),events=n,
                recall=sum(r['recalled'] for r in local)/n if n else None,
                false_per_quiet_minute=60*sum(r['false_alarms'] for r in local)/quiet if quiet else None,
                median_delay_seconds=float(np.median(delay)) if delay else None,
                velocity_rmse=float(np.sqrt(np.mean([r['velocity_mse'] for r in local if r['velocity_mse'] is not None])))))
    return output
