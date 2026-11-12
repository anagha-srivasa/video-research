"""Dataset evaluation, equal-input memory probes and certified geometry trials."""
import json
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT
from .common import C,D,R,arrays,save,atomic_json,LOG
from .evaluate import estimate_episode,query_rows,summarize,common_support,METHODS
from .compression import fit_distiller,fit_task_projection


def real(parameters,monitor):
    from .data import load_episode
    info=json.loads((R/'anchor_data.json').read_text());rows=[];features=[];latency=[]
    for meta in info['episodes']:
        e=load_episode(ROOT/meta['path']);e.update(meta)
        e['time']=e['times'];e['participant']=e['video_id'].split('_')[0]
        states={}
        for source in ('centroid','anchor'):
            for method in METHODS:
                state=estimate_episode(e,source,method,parameters);state['q']=parameters['q'];states[source,method]=state
                if method=='adaptive':
                    arrays(D/f"{meta['episode_id']}.{source}.state.npz",**{k:v for k,v in state.items() if isinstance(v,np.ndarray)})
                    latency.append(dict(episode_id=meta['episode_id'],source=source,observations=int(e['valid'].sum()),filter_seconds=state.get('filter_seconds',0.)))
        local=query_rows(e,states);rows.extend(local)
        # Equal cohort probes: current kinematics, three posterior memories,
        # and three recent knots (six numbers each). Targets remain source proxies.
        state=states['anchor','adaptive']
        for r in local:
            if r['source']!='anchor' or r['method']!='adaptive' or r['horizon']==0 or not r['supported']:continue
            i=r['cutoff_index'];past=np.flatnonzero(e['valid']&(e['time']<=e['time'][i]))
            knots=[]
            for age in (.125,.5,1.):
                available=past[e['time'][past]<=e['time'][i]-age]
                if not len(available):break
                knots.extend(e['anchor_positions'][available[-1]]-state['mean'][i,:2])
            if len(knots)!=6:continue
            features.append(dict(participant=e['participant'],split=e['split'],episode_id=e['episode_id'],horizon=r['horizon'],
                                 state=state['mean'][i,2:4].tolist(),memory=(state['memory_mean'][i]-state['mean'][i,:2]).ravel().tolist(),
                                 knots=knots,target=(np.array(r['target'])-state['mean'][i,:2]).tolist()))
        monitor.check();LOG.info('Evaluated %s',meta['episode_id'])
    save(R/'real_predictions.parquet',rows);save(R/'memory_features.parquet',features)
    atomic_json(R/'real.json',dict(summary=summarize(rows),matched_summary=summarize(common_support(rows)),
        independent_references=0,reference_kind='disjoint same-algorithm RGB anchors; shared seed and tracker errors remain',
        support_is_outcome_not_a_selection_filter=True,latency=latency,conditional_manual_initialization=True,
        neural_point_tracker=False,physical_motion_acceptance=False))
    return memory_probes(features)


def memory_probes(rows):
    results=[];models=[]
    for horizon in C['horizons_seconds']:
        local=[r for r in rows if r['horizon']==horizon]
        if not local:continue
        participants=np.array([r['participant'] for r in local]);split=np.array([r['split'] for r in local]);y=np.array([r['target'] for r in local])
        state=np.array([r['state'] for r in local]);mem=np.array([r['memory'] for r in local]);knots=np.array([r['knots'] for r in local])
        arms=dict(state=state,state_memory=np.c_[state,mem],state_knots=np.c_[state,knots])
        train=split=='train'
        if train.sum()<8:
            results.append(dict(horizon=horizon,status='insufficient_training_support',training_rows=int(train.sum())));continue
        weights=np.array([1/max(1,np.sum((participants==p)&train)) for p in participants])
        # Train-only task sensitivity, then rank-3 six-dimensional memory code.
        teacher=fit_distiller(mem,y,split,weights=weights,ridge=.01)
        jac=teacher.coefficients/teacher.input_scale[:,None]
        projection=fit_task_projection(mem,jac@jac.T+np.eye(6)*1e-6,3,split,weights=weights)
        arms['state_task_memory']=np.c_[state,projection.encode(mem)]
        arrays(D/f'memory_projection.{horizon}.npz',mean=projection.mean,encoder=projection.encoder,decoder=projection.decoder,metric=projection.metric)
        for arm,x in arms.items():
            model=fit_distiller(x,y,split,weights=weights,ridge=.01)
            prediction=model.predict(x)
            arrays(D/f'probe.{horizon}.{arm}.npz',input_mean=model.input_mean,input_scale=model.input_scale,target_mean=model.target_mean,coefficients=model.coefficients)
            for part in ('train','validation'):
                vals=[]
                for p in sorted(set(participants[split==part])):
                    mask=(split==part)&(participants==p)
                    vals.append(dict(participant=p,rows=int(mask.sum()),rmse=float(np.sqrt(np.mean(np.sum((prediction[mask]-y[mask])**2,axis=1))))))
                if vals:results.append(dict(horizon=horizon,arm=arm,split=part,participants=vals,mean_participant_rmse=float(np.mean([v['rmse'] for v in vals])),training_rows=model.training_rows,parameter_bytes=model.parameter_bytes))
            models.append(dict(horizon=horizon,arm=arm,training_participants=sorted(set(participants[train])),ridge=.01))
    value=dict(results=results,models=models,validation_used_for_fitting=False,
               equal_memory_feature_count=6,task_projection_rank=3,reference='correlated algorithmic RGB proxy',
               offline_linear_auxiliary_probes=True,neural_teacher_distillation=False,physical_reference_count=0)
    atomic_json(R/'memory.json',value)
    return value


def geometry(monitor):
    from .geometry import fit_adaptive_geometry,evaluate_geometry
    rows=[]
    for video in C['train']+C['validation']:
        table=pq.read_table(ROOT/f'canonical/v2/masks.{video}.parquet',columns=['mask_uid','source_kind','contours_json'])
        source=[r for r in table.to_pylist() if r['source_kind']=='manual_sparse'][:C['geometry_masks_per_video']]
        for r in source:
            rings=json.loads(r['contours_json']);points=[np.array(x['points'],float) for x in rings]
            roles=[{'inner':'hole'}.get(x['role'],x['role']) for x in rings];parents=[x.get('parent_ring',-1) for x in rings]
            for budget in C['geometry_controls']:
                record=dict(video_id=video,split='train' if video in C['train'] else 'validation',mask_uid=r['mask_uid'],budget=budget,supported=False)
                if not points or any(len(p)<12 for p in points):record['failure_code']='insufficient_ring_points_for_disjoint_holdout'
                else:
                    retained=[p[np.arange(len(p))%4!=0] for p in points];hidden=[p[np.arange(len(p))%4==0] for p in points]
                    try:
                        fit=fit_adaptive_geometry(retained,roles,parents,control_options=(8,16,24,32),max_total_controls=budget,max_certificate_attempts=2)
                        record.update(evaluate_geometry(fit,hidden),allocation=fit.get('allocation'),numeric_payload_bytes=fit.get('numeric_payload_bytes'))
                        if fit['supported']:
                            arrays(D/f"geometry.{r['mask_uid']}.{budget}.npz",coefficients=fit['coefficients'],centre=fit['centre'],scale=fit['scale'])
                            atomic_json(D/f"geometry.{r['mask_uid']}.{budget}.json",{k:fit[k] for k in ('allocation','roles','parents','certificate')})
                    except ValueError as exc:record['failure_code']=str(exc)
                rows.append(record)
            monitor.check()
        LOG.info('Geometry %s: %d source masks',video,len(source))
    save(R/'geometry.parquet',rows)
    atomic_json(R/'geometry.json',dict(proposed=len(rows),supported=sum(r['supported'] for r in rows),
        reference='withheld points from supplied source polygons, not independent physical boundaries',
        all_rings_or_explicit_failure=True,static_geometry_only=True,local_knot_refinement=False,temporal_correspondence=False))
