"""Certified static geometry + separate CV pose, on retrospective observations.
This is Track 1 representation/continuation testing, never Track 2 training.
"""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from track1.numerics import basis
from track1_lab.causal_kinematics import estimate
from track1_lab.topology_certificate import certify,evaluate_curves,input_hashes
from track1_lab.motion import project
from .common import *
from .temporal import load,proposals,timing_eligible,prepare,fit
from .evaluation import curves,matched_masks
from .readiness import R,W


def anchor_geometry(row,h,n):
 c=np.zeros((4,2*n+3),np.float32);c[0]=h;allocation=[16] if n==16 else row['allocation'];roles=['outer'] if n==16 else row['roles'];parents=[-1] if n==16 else row['parents'];cert=certify(c,allocation,roles,parents,window_seconds=1.,max_depth=6,max_nodes=127)
 return dict(coefficients=c,allocation=allocation,roles=roles,parents=parents,certificate=cert,source_complete=(row['source_rings']==1 if n==16 else row['source_complete']),mask_uid=row['mask_uid'],timestamp_us=row['timestamp_us'])

def moved_curves(anchor,position,camera,samples=512):
 cert=anchor['certificate'];c=anchor['coefficients'];expected=input_hashes(c,anchor['allocation'],anchor['roles'],anchor['parents'],1.)
 if not cert['certified'] or any(cert.get(k)!=value for k,value in expected.items()):raise ValueError('Anchor certificate binding invalid')
 normalized=evaluate_curves(c,anchor['allocation'],anchor['roles'],1.,samples);scale=np.sqrt(camera['intrinsic_native'][0][0]*camera['intrinsic_native'][1][1])*np.exp(c[0,-1])
 if not np.isfinite(position).all() or not np.isfinite(scale) or scale<=0:raise ValueError('Invalid similarity pose')
 centre=project(np.asarray(position,float),camera)
 return [p*scale+centre for p in normalized]


def source(v):
 return {r['mask_uid']:r for r in pq.read_table(WORK/f'inputs.{v}.parquet').to_pylist()}


def worker(v):
 camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());src=source(v);names={r['entity_track_id']:r['entity_name'] for r in pq.read_table(ROOT/f'canonical/v2/spatial_metadata.{v}.parquet',columns=['entity_track_id','entity_name']).to_pylist()};rows,_=load(v,16);bytrack=collections.defaultdict(list)
 for r in rows:bytrack[r['entity_track_id']].append(r['row_index'])
 episodes=[e for e in pq.read_table(ROOT/'dataset/partition_v2/episodes.parquet').to_pylist() if e['video_id']==v];evaluation=[];attempts=[];streams=[];anchors=[]
 parameters=json.loads((folder('calibration')/'model.json').read_text())
 for n in [16,32]:
  _,data=load(v,n);cache={};anchor_indices={};options=parameters[str(n)]['options'];scale=np.array(parameters[str(n)]['feature_scale'])
  def anchor(index):
   if index not in cache:
    cache[index]=anchor_geometry(rows[index],data['h'][index],n);a=cache[index];anchor_indices[index]=len(anchors);anchors.append(dict(video_id=v,model=n,row_index=len(anchors),source_row_index=index,mask_uid=a['mask_uid'],timestamp_us=a['timestamp_us'],allocation=a['allocation'],roles=a['roles'],parents=a['parents'],source_complete=a['source_complete'],certificate_json=json.dumps(a['certificate']),coefficients_json=json.dumps(a['coefficients'].tolist())))
   return cache[index]
  # Opportunity cutoffs are expanded into the early clip prefix. A geometry/CV
  # state need not wait until a full one-second history can be fitted.
  seen=set();queries=[]
  for track,ids0 in sorted(bytrack.items()):
   ids=np.asarray(ids0);tt=np.array([rows[i]['timestamp_us'] for i in ids])
   for episode in episodes:
    if not np.any((tt>=episode['start_us'])&(tt<episode['end_us'])):continue
    for cutoff in range(((episode['start_us']+249999)//250000)*250000,episode['end_us']+1,250000):
     if (track,cutoff) in seen:continue
     seen.add((track,cutoff));past=ids[(tt>=max(episode['start_us'],cutoff-1000000))&(tt<=cutoff)];times=np.array([rows[i]['timestamp_us']/1e6 for i in past]);positions=np.asarray(data['h'][past,:2],float);var=np.asarray(data['variance'][past,:2],float);kin=estimate(times,positions,var,cutoff/1e6);reason=kin.failure_reason;selected=[i for i in past if rows[i][f'fit{n}_valid'] and cutoff-rows[i]['timestamp_us']<=100000];a=None;ai=None;ok=False
     if not kin.support_valid:reason='kinematics:'+str(reason)
     elif not selected:reason='no_recent_valid_geometry'
     else:
      ai=selected[-1];a=anchor(ai);later=[i for i in past if rows[i]['timestamp_us']>=a['timestamp_us']];mode_changed=any(rows[i]['source_signature']!=rows[ai]['source_signature'] for i in later)
      if mode_changed:reason='observed_anchor_topology_change'
      elif not a['source_complete']:reason='anchor_source_incomplete'
      elif not a['certificate']['certified']:reason='anchor_certificate_unresolved'
      else:ok=True;reason=None
     streams.append(dict(video_id=v,split=split(v),model=n,entity_track_id=track,window_end_us=cutoff,row_index=len(streams),valid=ok,epoch=rows[ai][f'epoch{n}'] if ai is not None else None,geometry_training_eligible=ok,anchor_row=anchor_indices[ai] if ai is not None else None,anchor_mask_uid=rows[ai]['mask_uid'] if ai is not None else None,reason=reason,identity_verified=False,functional_history_substituted=False,position=kin.position.tolist() if kin.support_valid else None,velocity=kin.velocity.tolist() if kin.support_valid else None,pose_covariance=kin.covariance.tolist() if kin.support_valid else None,measurement_availability='retrospective_source_masks',source_completeness_scope='anchor only; future completeness unverified'))
     # Future-target existence is accounted independently of past-state support.
     if cutoff<episode['start_us']+1000000:continue
     queries.append((track,cutoff,episode,past,kin,ai,a,ok,reason,ids,tt))
  queries=sorted(queries,key=lambda q:rank(q[0]+str(q[1])))[:160]
  for track,cutoff,episode,past,kin,ai,a,ok,reason,ids,tt in queries:
   for horizon in [.1,.25,.5]:
    target_time=cutoff+round(horizon*1e6);candidates=np.flatnonzero((tt>cutoff)&(tt<episode['end_us'])&(abs(tt-target_time)<=40000));target=ids[candidates[np.argmin(abs(tt[candidates]-target_time))]] if len(candidates) else None;ledger=dict(video_id=v,split=split(v),model=n,entity_track_id=track,cutoff_us=cutoff,horizon_seconds=horizon,target_present=target is not None,transport_supported=ok,reason=reason,anchor_mask_uid=a['mask_uid'] if a else None)
    if target is None:attempts.append(dict(ledger,reason='future_reference_unavailable'));continue
    if not ok:attempts.append(ledger);continue
    actual=(rows[target]['timestamp_us']-cutoff)/1e6;target_row=rows[target];methods={};pred_positions={};nearest_position=np.asarray(data['h'][ai,:2],float);transport_position=kin.position+actual*kin.velocity
    methods['nearest_geometry']=(moved_curves(a,nearest_position,camera),a['roles'],a['parents']);pred_positions['nearest_geometry']=nearest_position
    methods['canonical_cv_transport']=(moved_curves(a,transport_position,camera),a['roles'],a['parents']);pred_positions['canonical_cv_transport']=transport_position
    # CV raster baseline uses the identical pose estimate, retaining the full
    # source-raster contour instead of compressing it into spline controls.
    from data_v2.geometry import normalize
    native=np.array(camera['native_size']);source_scale=native/[src[a['mask_uid']]['annotation_width'],src[a['mask_uid']]['annotation_height']];raw=normalize(src[a['mask_uid']]['contours_json']);delta=project(transport_position,camera)-project(nearest_position,camera);methods['cv_source_polygon_transport']=([np.asarray(r['points'])*source_scale+delta for r in raw],[r['role'] for r in raw],[r['parent_ring'] for r in raw]);pred_positions['cv_source_polygon_transport']=transport_position
    earlier=[i for i in past if rows[i][f'fit{n}_valid'] and i!=ai and rows[ai]['timestamp_us']-rows[i]['timestamp_us']<=120000 and rows[i][f'epoch{n}']==rows[ai][f'epoch{n}']]
    if earlier:
     j=earlier[-1];dt=(rows[ai]['timestamp_us']-rows[j]['timestamp_us'])/1e6
     if dt>0:
      pred=np.asarray(data['h'][ai],float)+(rows[target]['timestamp_us']-rows[ai]['timestamp_us'])/1e6*(data['h'][ai]-data['h'][j])/dt;methods['linear_coefficients']=(curves(pred,a['allocation'],camera),a['roles'],a['parents']);pred_positions['linear_coefficients']=pred[:2]
    case=dict(indices=past.tolist(),window_end_us=cutoff)
    try:
     p=prepare(case,rows,data,n);f=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,scale);pred=basis(1+2*actual)@f['coefficients'];methods['functional_cubic_extrapolation']=(curves(pred,p['allocation'],camera),p['roles'],p['parents']);pred_positions['functional_cubic_extrapolation']=pred[:2]
    except ValueError:pass
    metrics=matched_masks(src[target_row['mask_uid']],list(methods.values()),np.array(camera['native_size']));target_position=np.asarray(data['h'][target,:2],float);target_pixel=project(target_position,camera)
    for (method,prediction),metric in zip(methods.items(),metrics):
     position_error=float(np.linalg.norm(project(pred_positions[method],camera)-target_pixel));evaluation.append(dict(**ledger,method=method,actual_horizon_seconds=actual,target_mask_uid=target_row['mask_uid'],target_source_kind=target_row['source_kind'],entity_type='hand' if 'hand' in names.get(track,'').lower() else 'other',iou=metric['iou'],boundary_fscore_2_annotation_px=metric['boundary_fscore_2px'],position_error_native_px=position_error,target_ring_mode_changed=(rows[target]['source_signature']!=rows[ai]['source_signature']),predicted_shape_topology_certificate=method in ['nearest_geometry','canonical_cv_transport'],identity_verified=False,online_rgb_pipeline=False))
    attempts.append(dict(ledger,reason=None,methods_available=list(methods)))
  monitor_guard=guard()
 save(R/f'transport.{v}.parquet',evaluation);save(R/f'transport_attempts.{v}.parquet',attempts);save(R/f'transport_stream.{v}.parquet',streams);save(R/f'transport_anchors.{v}.parquet',anchors);return dict(video_id=v,queries=len(attempts),evaluated=len(evaluation),stream_attempts=len(streams),stream_valid=sum(r['valid'] for r in streams))


def run(monitor):
 from .store import sequences
 results=pool(worker,VIDEOS,monitor,workers=5);records=[];streams=[];attempts=[]
 for v in VIDEOS:
  records+=pq.read_table(R/f'transport.{v}.parquet').to_pylist();streams+=pq.read_table(R/f'transport_stream.{v}.parquet').to_pylist();attempts+=pq.read_table(R/f'transport_attempts.{v}.parquet').to_pylist()
 save(R/'transport.parquet',records);save(R/'transport_attempts.parquet',attempts);continuity=[]
 for v in VIDEOS:
  for n in [16,32]:
   local=[r for r in streams if r['video_id']==v and r['model']==n]
   for length in [4,8,16]:
    seq=sequences(local,length);continuity.append(dict(video_id=v,split=split(v),model=n,length=length,sequences=len(seq),tracks=len({r['entity_track_id'] for r in seq}),state_kind='geometry_cv_transport',functional_core_replacement=False))
 save(R/'transport_continuity.parquet',continuity);summary=[]
 # Common query cohort for all compared methods; additional support is reported
 # independently, never silently credited as better prediction accuracy.
 groups=collections.defaultdict(list)
 for r in records:groups[(r['video_id'],r['model'],r['entity_track_id'],r['cutoff_us'],r['horizon_seconds'])].append(r)
 required={'nearest_geometry','canonical_cv_transport','cv_source_polygon_transport','linear_coefficients','functional_cubic_extrapolation'};paired=[r for rs in groups.values() if {r['method'] for r in rs}>=required for r in rs]
 for part in ['train','validation']:
  for n in [16,32]:
   for horizon in [.1,.25,.5]:
    for method in sorted(required):
     rows=[r for r in paired if (r['split'],r['model'],r['horizon_seconds'],r['method'])==(part,n,horizon,method)];byvideo=[]
     for v in sorted({r['video_id'] for r in rows}):
      local=[r for r in rows if r['video_id']==v];byvideo.append((float(np.mean([r['iou'] for r in local])),float(np.sqrt(np.mean([r['position_error_native_px']**2 for r in local])))))
     if byvideo:summary.append(dict(split=part,model=n,horizon_seconds=horizon,method=method,queries=len(rows),mean_video_iou=float(np.mean([x[0] for x in byvideo])),mean_video_position_rmse_native_px=float(np.mean([x[1] for x in byvideo]))))
 atomic_json(R/'transport.json',dict(videos=results,matched_all_method_metrics=summary,continuity=continuity,common_all_method_queries=sum({r['method'] for r in rs}>=required for rs in groups.values()),motion_method='existing 0.2-second local OLS constant velocity; no new learned motion estimator',limits=['Past-observation replay on retrospective masks, not an online RGB forecasting benchmark.','A certified anchor plus a separate finite positive similarity pose preserves modeled curve topology; future physical topology/fidelity remains unverified.','CV source-polygon baseline retains source contour detail and uses the identical motion estimate.','Functional continuation is a diagnostic; its history certificate is not extrapolated.','No motion/contact/identity independent reference or calibrated transport geometry covariance.'],track2_started=False))
