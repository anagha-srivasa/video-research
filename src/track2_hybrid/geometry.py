"""Optional certified anchor transport; future source polygons are evaluation only."""
import json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json
from track1_retrain.common import save
from track1_retrain.evaluation import source_rows,matched_masks
from track1_encoder_revision.runtime import CompiledGeometry
from track1_lab.motion import project
from track1.numerics import basis
from track2_feasibility.data import Data,BASE
from .experiment import REPORT,DATA,CFG
from .imm import forecast


def transport(rings,scale,position,camera):
    if not np.isfinite(scale) or scale<=0: raise ValueError('Positive finite pose scale required')
    centre=project(np.asarray(position,float),camera)
    if not np.isfinite(centre).all():raise ValueError('Nonfinite pose')
    return [np.asarray(r,float)*scale+centre for r in rings]


def evaluate():
    data=Data(); histories=json.loads((DATA/'histories.json').read_text())
    states=json.loads((DATA/'validation_filter_states.json').read_text())
    episodes={r['episode_id']:r for r in pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist()}
    ledger=[];metrics=[]
    for video in sorted({s['video_id'] for s in data.samples if s['split']=='validation'}):
        raw=pq.read_table(BASE/'spatial'/f'{video}.parquet').to_pylist()
        functional=pq.read_table(BASE/'functional'/f'{video}.m32.parquet').to_pylist()
        lookup={(r['entity_track_id'],r['window_end_us']):r for r in functional}
        a=np.load(BASE/'functional'/f'{video}.m32.geometry_coefficients.npy',mmap_mode='r')
        camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{video}.camera.json').read_text())
        targets=source_rows(video,{r['mask_uid'] for r in raw})
        for i,s in enumerate(data.samples):
            if s['split']!='validation' or s['video_id']!=video: continue
            end=s['input_times_us'][-1];anchor=lookup.get((s['entity_track_id'],end));ep=episodes[s['episode_id']]
            rings=None;scale=None;reason=None
            if str(i) not in histories:reason='unsupported_raw_history'
            elif anchor is None or not anchor['geometry_training_eligible']:reason='missing_complete_certified_M32'
            elif anchor['episode_id']!=s['episode_id']:raise ValueError('Anchor episode mismatch')
            else:
                c=np.array(a[anchor['row_index']],np.float32)
                rings=CompiledGeometry(c,anchor,CFG['geometry_samples_per_ring']).evaluate(end)
                h=basis(1.)@c;K=np.asarray(camera['intrinsic_native']);scale=float(np.sqrt(K[0,0]*K[1,1])*np.exp(h[-1]))
                if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid certified anchor pose')
            available=[r for r in raw if r['entity_track_id']==s['entity_track_id'] and end<r['timestamp_us']<=ep['end_us']]
            retained=[r for r in raw if r['entity_track_id']==s['entity_track_id'] and ep['start_us']<=r['timestamp_us']<=end]
            latest=max(retained,key=lambda r:r['timestamp_us']) if retained else None
            for horizon in CFG['geometry_horizons_us']:
                target=min(available,key=lambda r:abs(r['timestamp_us']-end-horizon)) if available else None
                target_ok=target is not None and abs(target['timestamp_us']-end-horizon)<=CFG['geometry_target_tolerance_us']
                item=dict(sequence_id=s['sequence_id'],video_id=video,participant=s['participant'],episode_id=s['episode_id'],
                          horizon_us=horizon,anchor_available=rings is not None,target_available=target_ok,
                          scored=rings is not None and target_ok,reason=reason or (None if target_ok else 'no_nearby_future_source_mask'),
                          target_mask_uid=target['mask_uid'] if target_ok else None,
                          target_topology_changed=(target['source_signature']!=latest['source_signature']) if target_ok and latest else None)
                ledger.append(item)
                if not item['scored']:continue
                dt=(target['timestamp_us']-end)/1e6
                positions={'cv':data.x[i,-1,140:142].astype(float)+dt*data.x[i,-1,142:144],
                    'constant_position':data.x[i,-1,140:142].astype(float),
                    'nearest_observation':histories[str(i)]['positions'][-1]}
                for method in states:
                    state=dict(states[method][str(i)])
                    for k in ('means','covariances','weights'):state[k]=np.asarray(state[k],float)
                    positions[method]=forecast(state,dt)[0][:2]
                predictions=[(transport(rings,scale,p,camera),anchor['roles'],anchor['parents']) for p in positions.values()]
                scores=matched_masks(targets[target['mask_uid']],predictions,np.asarray(camera['native_size']))
                for method,score in zip(positions,scores):
                    metrics.append(dict(item,method=method,actual_horizon_us=target['timestamp_us']-end,
                        iou=score['iou'],boundary_fscore_2_annotation_px=score['boundary_fscore_2px']))
        a._mmap.close()
    save(REPORT/'geometry_attempts.parquet',ledger);save(REPORT/'geometry_metrics.parquet',metrics)
    summary=[]
    for method in CFG['primary_methods']:
        for horizon in CFG['geometry_horizons_us']:
            rs=[r for r in metrics if r['method']==method and r['horizon_us']==horizon]
            summary.append(dict(method=method,horizon_us=horizon,scored=len(rs),
                mean_video_iou=float(np.mean([np.mean([r['iou'] for r in rs if r['video_id']==v]) for v in sorted({r['video_id'] for r in rs})])) if rs else None))
    atomic_json(REPORT/'geometry.json',dict(records=summary,attempts=len(ledger),scored=sum(r['scored'] for r in ledger),
        missing_geometry=sum(not r['anchor_available'] for r in ledger),
        certificate='normalized historical anchor only; positive frozen scale and translation preserve its curve topology',
        future_physical_geometry_certified=False,source_reconstruction_only=True,contact_reference=False,
        target_topology_excluded=False,boundary_units='annotation pixels',shape_dynamics_learned=False))
