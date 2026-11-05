"""Coverage-first scoring only after independent, hash-bound adjudication."""
import collections,json
import numpy as np
from scipy.spatial import cKDTree
from . import ROOT,REF,REPORT
from .protocol import Repository,status,write,digest
from .events import event_stream,interval_scores

def dense_line(points,closed=False):
 p=np.asarray(points,float)
 if len(p)<2:return np.empty((0,2))
 if closed:p=np.r_[p,p[:1]]
 result=[]
 for a,b in zip(p[:-1],p[1:]):
  n=min(20000,max(1,int(np.ceil(np.linalg.norm(b-a)))));result.extend(a+(b-a)*np.arange(n)[:,None]/n)
 return np.asarray(result)

def boundary_quality(predicted,reference,scope):
 ref=dense_line(reference,scope=='full_visible_outline');pred=np.concatenate([dense_line(p,True) for p in predicted])
 if scope=='contact_region':
  lo=ref.min(0)-10;hi=ref.max(0)+10;pred=pred[np.all((pred>=lo)&(pred<=hi),axis=1)]
 if not len(pred):return dict(bf2=0.,mean_symmetric_distance_px=None),pred,ref
 a=cKDTree(ref).query(pred)[0];b=cKDTree(pred).query(ref)[0];precision=float(np.mean(a<=2));recall=float(np.mean(b<=2));f=2*precision*recall/(precision+recall) if precision+recall else 0.
 return dict(bf2=f,mean_symmetric_distance_px=float((a.mean()+b.mean())/2)),pred,ref

def geometry_case(repo,task,reference):
 import pyarrow.parquet as pq
 from track1_remediation.store import TieredDataset
 from track1_lab.topology_certificate import evaluate_curves
 from track1_lab.motion import project
 from track1.numerics import basis
 private=repo.private['tasks'][task['task_id']];v=private['video_id'];timestamp=private['query_timestamp_us'];frame=reference['frames'][str(task['query_index'])];camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());focal=np.sqrt(camera['intrinsic_native'][0][0]*camera['intrinsic_native'][1][1]);metadata=pq.read_table(ROOT/f'derived/track1_context_v3/functional/{v}.m16.parquet').to_pylist();records=[]
 with TieredDataset('context') as ds:
  for method in ['m16','tiered']:
   try:
    curves=[];kinds=[];complete=[]
    for track in [private.get('hand_track_id'),private.get('object_track_id')]:
     if not track:raise ValueError('source_entity_unresolved')
     # The earliest stored history ending at/after this reference time is fixed
     # before checking support. No search for an easier supported window.
     candidates=[r for r in metadata if r['entity_track_id']==track and timestamp<=r['window_end_us']<=timestamp+250000]
     if not candidates:raise ValueError('no_matching_history_opportunity')
     row=min(candidates,key=lambda r:r['window_end_us']);state=ds.get_state(v,track,row['window_end_us'])
     if method=='tiered':
      g=state['geometry32']
      if g is None:raise ValueError('missing_optional_geometry')
      ds._verify_geometry(g);c=g['coefficients'];m=g['metadata'];kind=g['kind'];complete.append(True)
     else:
      core=state['core16'];c=core['geometry_coefficients'];m=core['metadata'];kind='dominant_M16'
      if not m['certified_geometry_valid']:raise ValueError('uncertified_core_geometry')
      complete.append(m['source_complete'])
     if not m['window_start_us']<=timestamp<=m['window_end_us']:raise ValueError('reference_outside_latest_epoch')
     phase=(timestamp-m['window_start_us'])/(m['window_seconds']*1e6);norm=evaluate_curves(c,m['allocation'],m['roles'],phase,2048);h=basis(2*phase-1)@c;centre=project(h[:2],camera);scale=focal*np.exp(h[-1]);curves.append([p*scale+centre for p in norm]);kinds.append(kind)
    scope=reference.get('boundary_scope','contact_region');metrics=[];pred=[];truth=[]
    for shape,key in zip(curves,['hand_boundary','object_boundary']):
     score,p,t=boundary_quality(shape,frame[key],scope);metrics.append(score);pred.append(p);truth.append(t)
    actual_gap=float(cKDTree(truth[0]).query(truth[1])[0].min());pred_gap=float(cKDTree(pred[0]).query(pred[1])[0].min()) if all(len(p) for p in pred) else None
    records.append(dict(task_id=task['task_id'],method=method,status='scored',boundary_fscore_2_native_px=float(np.mean([m['bf2'] for m in metrics])),gap_error_native_px=abs(pred_gap-actual_gap) if pred_gap is not None else None,support_kind='latest_short_epoch' if 'latest_stable_topology_segment' in kinds else 'full_epoch',represented_source_complete=all(complete),scope=scope,reference_visual_conditions=reference.get('visual_conditions',[]),time_semantics='retrospective history reconstruction; not online geometry prediction',physical_contact_score=None))
   except (ValueError,KeyError,OSError) as e:records.append(dict(task_id=task['task_id'],method=method,status='unsupported',reason=str(e),boundary_fscore_2_native_px=0.,gap_error_native_px=None,support_kind='missing_geometry',reference_visual_conditions=reference.get('visual_conditions',[])))
 return records

def interpolate(t,p,at,max_gap=.08):
 j=np.searchsorted(t,at)
 if j<len(t) and abs(t[j]-at)<1e-8:return p[j]
 if j==0 or j==len(t) or t[j]-t[j-1]>max_gap+1e-9:return None
 return p[j-1]+(p[j]-p[j-1])*(at-t[j-1])/(t[j]-t[j-1])

def reference_velocity(t,p,at,window,degree):
 keep=abs(t-at)<=window
 if keep.sum()<4 or np.diff(t[keep]).max()>.08+1e-9:return None
 x=np.polynomial.polynomial.polyvander(t[keep]-at,degree);beta=np.linalg.lstsq(x,p[keep],rcond=None)[0];return beta[1]

def motion_case(repo,task,reference):
 import pyarrow.parquet as pq
 from track1_lab.causal_kinematics import estimate
 from track1_remediation.motion import switching,predict
 private=repo.private['tasks'][task['task_id']];frames=repo.private['clips'][task['clip_id']];anchors=[(int(i),f['landmark'][0]) for i,f in reference.get('frames',{}).items() if f.get('visibility')=='visible' and f.get('landmark')];anchors.sort()
 if len(anchors)<7:return dict(task_id=task['task_id'],status='unsupported',reason='insufficient_visible_reference_anchors')
 rt=np.array([frames[i]['timestamp_us']/1e6 for i,p in anchors]);rp=np.array([p for i,p in anchors]);rows=pq.read_table(ROOT/f'work/track1_readiness/inputs.{private["video_id"]}.parquet').to_pylist();rows=[r for r in rows if r['entity_track_id']==private['entity_track_id']];t=np.array([r['timestamp_us']/1e6 for r in rows]);p=np.array([[r['centroid_x']*r['width']/r['annotation_width'],r['centroid_y']*r['height']/r['annotation_height']] for r in rows]);seed=np.flatnonzero((t<=rt[0])&(t>=rt[0]-.04))
 if not len(seed):return dict(task_id=task['task_id'],status='unsupported',reason='no_past_seed_alignment')
 if reference.get('identity_across_occlusion')=='different':return dict(task_id=task['task_id'],status='unsupported',reason='reference_identity_transition_not_resolved_by_source_chain')
 offset=rp[0]-p[seed[-1]];p=p+offset;var=np.ones_like(p)*4;cutoffs=np.arange(np.ceil((rt[0]+1)*4)/4,rt[-1]-.5+1e-9,.25);predictions=[];prediction_ledger=[];velocity=[];attempts=0
 for cutoff in cutoffs:
  attempts+=1;cv=estimate(t,p,var,cutoff)
  try:bank=switching(t,p,var,cutoff)
  except ValueError:bank=None
  if bank is None or not cv.support_valid:
   prediction_ledger.extend(dict(cutoff=float(cutoff),horizon=h,method=m,status='unsupported',reason='common_estimator_support_missing') for h in [.25,.5] for m in ['nearest','cv','switching'])
   continue
  nearest=p[np.flatnonzero(t<=cutoff)[-1]]
  for horizon in [.25,.5]:
   truth=interpolate(rt,rp,cutoff+horizon)
   if truth is None:
    prediction_ledger.extend(dict(cutoff=float(cutoff),horizon=horizon,method=m,status='unsupported',reason='independent_reference_gap') for m in ['nearest','cv','switching'])
    continue
   prediction_ledger.extend(dict(cutoff=float(cutoff),horizon=horizon,method=m,status='scored') for m in ['nearest','cv','switching'])
   for method,point in [('nearest',nearest),('cv',cv.position+horizon*cv.velocity),('switching',predict(bank,horizon)[0])]:predictions.append(dict(method=method,cutoff=cutoff,horizon=horizon,error_native_px=float(np.linalg.norm(point-truth))))
  estimated={'nearest':np.zeros(2),'cv':cv.velocity,'switching':bank['weights']@np.array([b[1] for b,c in bank['models']])}
  for window,degree in [(.12,1),(.20,2)]:
   target=reference_velocity(rt,rp,cutoff,window,degree)
   if target is not None:
    for method,v in estimated.items():velocity.append(dict(method=method,reference_window=window,reference_degree=degree,error_native_px_s=float(np.linalg.norm(v-target))))
 truth_events=[dict(start=frames[e['start_index']]['timestamp_us']/1e6,end=frames[e['end_index']]['timestamp_us']/1e6) for e in reference.get('events',[]) if e['kind']=='motion_onset'];quiet=[(frames[e['start_index']]['timestamp_us']/1e6,frames[e['end_index']]['timestamp_us']/1e6) for e in reference.get('events',[]) if e['kind']=='quiet_interval'];detections,states=event_stream(t,p,var,cutoffs) if len(cutoffs) else ({},[]);scores={m:interval_scores(events,truth_events,quiet) for m,events in detections.items()}
 return dict(task_id=task['task_id'],status='scored' if predictions else 'unsupported',reason=None if predictions else 'reference_sampling_or_model_support',input_mode='source_centroid_with_one_past_landmark_seed_offset',cutoff_attempts=attempts,predictions=predictions,anchor_derived_velocity_sensitivity=velocity,event_scores=scores,prediction_opportunity_ledger=prediction_ledger,event_update_support=states,reference_is_2D_landmark_not_metric_motion=True,physical_contact_onset_error=None)

def run():
 repo=Repository();ledger=status(repo);geometry=[];motion=[]
 for r in ledger:
  task=repo.tasks[r['task_id']];reference=repo.accepted(r['task_id'])
  if not r['reference_usable']:
   if task['kind']=='contact':geometry.extend(dict(task_id=r['task_id'],method=m,status='pending_reference' if reference is None else 'reference_unscorable',support_kind='reference_unavailable') for m in ['m16','tiered'])
   else:motion.append(dict(task_id=r['task_id'],status='pending_reference' if reference is None else 'reference_unscorable'))
  elif task['kind']=='contact':
   scored=geometry_case(repo,task,reference);audit=repo.base/'source_audit'/f'{task["task_id"]}.json';diagnosis='not_reviewed'
   if audit.exists():
    value=json.loads(audit.read_text())
    if value.get('reference_sha256')==digest(repo.path('adjudicator',task['task_id'])):diagnosis=value['category']
   for row in scored:row['source_diagnosis']=diagnosis;row['prior_candidate_exposure']=reference.get('prior_candidate_exposure','unsure')
   for row in scored:row['video_id']=repo.private['tasks'][task['task_id']]['video_id'];row['context_id']=repo.private['tasks'][task['task_id']]['context_id']
   geometry+=scored
  else:motion.append(motion_case(repo,task,reference))
 summary=[]
 for method in ['m16','tiered']:
  rows=[r for r in geometry if r['method']==method];eligible=[r for r in rows if r['status'] in ['scored','unsupported']];scored=[r for r in rows if r['status']=='scored'];video_means=[];weighted_video=[]
  for v in sorted({r['video_id'] for r in eligible}):
   local=[r for r in eligible if r['video_id']==v];ok=[r for r in local if r['status']=='scored'];weighted_video.append(sum(r['boundary_fscore_2_native_px'] for r in local)/len(local))
   if ok:video_means.append(float(np.mean([r['boundary_fscore_2_native_px'] for r in ok])))
  summary.append(dict(conditional_boundary_fscore_mean_video=float(np.mean(video_means)) if video_means else None,coverage_weighted_boundary_fscore_mean_video=float(np.mean(weighted_video)) if weighted_video else None,method=method,planned=163,reference_eligible=len(eligible),geometry_scored=len(scored),coverage=len(scored)/len(eligible) if eligible else None,conditional_boundary_fscore=float(np.mean([r['boundary_fscore_2_native_px'] for r in scored])) if scored else None,coverage_weighted_boundary_fscore=sum(r['boundary_fscore_2_native_px'] for r in eligible)/len(eligible) if eligible else None,gap_within_2px_among_all_eligible=sum(r.get('gap_error_native_px') is not None and r['gap_error_native_px']<=2 for r in eligible)/len(eligible) if eligible else None))
 strata=[]
 for method in ['m16','tiered']:
  rows=[r for r in geometry if r['method']==method and r['status'] in ['scored','unsupported']]
  for kind in sorted({r['support_kind'] for r in rows}):
   local=[r for r in rows if r['support_kind']==kind];strata.append(dict(method=method,support_kind=kind,eligible=len(local),scored=sum(r['status']=='scored' for r in local)))
 write(REPORT/'geometry_reference_strata.json',strata)
 write(REPORT/'geometry_reference_ledger.json',geometry);write(REPORT/'motion_reference_ledger.json',motion);write(REPORT/'independent_evaluation.json',dict(geometry_summary=summary,planned_contact_tasks=163,planned_motion_tasks=20,adjudicated_usable=sum(r['reference_usable'] for r in ledger),scoring_blocked_until_adjudication=True,contact_sign_and_physical_onset_acceptance=False,identity_idf1_available=False,scientific_acceptance=False,track2_started=False));return summary
if __name__=='__main__':run();print('Reference ledger written; pending cases retained in the denominators')
