"""Source-event diagnosis and explicit variable-duration stable topology segments."""
import collections,itertools,json
import numpy as np
import pyarrow.parquet as pq
from track1.numerics import wavelet_matrix,basis
from track1_lab.topology_certificate import certify
from track1_retrain.temporal import load,proposals,fit,phase
from . import *

def match_rings(current,previous,roles,parents,previous_roles,previous_parents):
 """Shape plus relative-centre matching. No future frames or entity-ID claims."""
 if len(current)!=len(previous):return None,None
 choices=[]
 for order in itertools.permutations(range(len(current))):
  remap={j:i for i,j in enumerate(order)}
  if any(len(current[j])!=len(previous[i]) or roles[j]!=previous_roles[i] or remap.get(parents[j],-1)!=previous_parents[i] for i,j in enumerate(order)):continue
  cost=0.
  for i,j in enumerate(order):
   a=current[j];b=previous[i];aa=(a-a.mean(0))/max(np.linalg.norm(a-a.mean(0)),1e-12);bb=(b-b.mean(0))/max(np.linalg.norm(b-b.mean(0)),1e-12)
   shape=min(np.mean((np.roll(aa,-k,axis=0)-bb)**2) for k in range(len(a)))
   cost+=np.linalg.norm(a.mean(0)-b.mean(0))+.5*np.sqrt(shape)
  choices.append((float(cost),order))
 choices.sort()
 if not choices:return None,None
 margin=choices[1][0]-choices[0][0] if len(choices)>1 else None
 if margin is not None and margin<.05:return None,margin
 return choices[0][1],margin

def align(rows,data):
 previous=None;epoch=0;events=[];hs=[];variances=[];out=[]
 for r in rows:
  r=dict(r);i=r['row_index'];controls=np.asarray(data['controls'][i],float);allocation=r['allocation'];pieces=[];at=0
  for m in allocation:pieces.append((controls[at:at+m]-r['centre_native'])/r['scale_native']);at+=m
  changed=previous is None or previous['entity_track_id']!=r['entity_track_id'];gap=previous is not None and not changed and r['timestamp_us']-previous['timestamp_us']>350000;signature=previous is not None and not changed and r['source_signature']!=previous['source_signature'];allocation_change=previous is not None and not changed and allocation!=previous['allocation'];reason='new_entity' if changed else 'observation_gap' if gap else 'observed_source_topology_change' if signature else 'allocation_change' if allocation_change else None;order=tuple(range(len(pieces)));margin=None
  if reason is None:
   order,margin=match_rings(pieces,previous['pieces'],r['roles'],r['parents'],previous['roles'],previous['parents'])
   if order is None:reason='unresolved_ring_correspondence';order=tuple(range(len(pieces)))
  if reason is not None:
   epoch+=1
   if not changed:events.append(dict(video_id=r['video_id'],entity_track_id=r['entity_track_id'],timestamp_us=r['timestamp_us'],reason=reason,source_signature_before=previous['source_signature'],source_signature_after=r['source_signature'],provenance_change=previous['source_kind']!=r['source_kind'],physical_event_verified=False,identity_transition_verified=False,ring_lineage_across_event='unknown',recovery_allowed=False))
  remap={j:k for k,j in enumerate(order)};roles=[r['roles'][j] for j in order];parents=[remap.get(r['parents'][j],-1) for j in order];aligned=[phase(pieces[j],None if reason is not None else previous['pieces'][k]) for k,j in enumerate(order)];h=np.array(data['h'][i],float);h[2:-1]=np.concatenate([(wavelet_matrix(len(p))@p).ravel() for p in aligned]);starts=np.cumsum([2]+[2*m for m in allocation]);var=np.asarray(data['variance'][i],float);var=np.r_[var[:2],np.concatenate([var[starts[j]:starts[j+1]] for j in order]),var[-1]];key=f'{r["entity_track_id"]}:piecewise:{epoch}'
  r.update(candidate_epoch=key,candidate_roles=roles,candidate_parents=parents,persistent_ring_ids=[f'{key}:ring{k}' for k in range(len(pieces))],assignment_margin=margin,candidate_ambiguity=reason=='unresolved_ring_correspondence');out.append(r);hs.append(h);variances.append(var);previous=dict(r,pieces=aligned,roles=roles,parents=parents)
 return out,np.asarray(hs),np.asarray(variances),events

def diagnose(cases,rows,old,events):
 bytrack=collections.defaultdict(list)
 for e in events:bytrack[e['entity_track_id']].append(e)
 records=[]
 for case in cases:
  original=old[(case['entity_track_id'],case['window_end_us'])]
  if original['failure_reason']!='global_correspondence_epoch_change':continue
  ids=case['indices'];local=[rows[i] for i in ids if rows[i]['fit32_valid']];changes=[e for e in bytrack[case['entity_track_id']] if case['window_end_us']-1000000<e['timestamp_us']<=case['window_end_us']];source_changed=len({r['source_signature'] for r in local})>1;assignment=any(e['reason']=='unresolved_ring_correspondence' for e in changes);epochs={r['candidate_epoch'] for r in local}
  # A transient source-graph reversion is only a fragmentation/occlusion
  # hypothesis. It cannot justify interpolating unseen physical geometry.
  signatures=[r['source_signature'] for r in local];transient=source_changed and len(signatures)>2 and signatures[0]==signatures[-1]
  cls='source_topology_change_fragmentation_or_occlusion_hypothesis' if transient else 'observed_source_ring_graph_change' if source_changed else 'unresolved_ring_order_or_identity' if assignment else 'unchanged_topology_correspondence_candidate'
  records.append(dict(video_id=case['video_id'],split=split(case['video_id']),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],cause_class=cls,source_topology_changed=source_changed,transient_source_reversion=transient,provenance_change=any(e['provenance_change'] for e in changes),same_candidate_epoch=len(epochs)==1,physical_cause='unverified',identity_cause='unverified',automatically_bridgeable=False))
 return records

def segment_indices(case,rows):
 raw=case['indices'];epoch=rows[raw[-1]]['candidate_epoch'] if raw else None
 return [i for i in raw if rows[i]['fit32_valid'] and rows[i]['candidate_epoch']==epoch]

def worker(job):
 name,v=job;source=SOURCES[name];rows,data=load(v,32,source/'spatial');meta=pq.read_table(source/'functional'/f'{v}.m32.parquet').to_pylist();old={(r['entity_track_id'],r['window_end_us']):r for r in meta};episodes=pq.read_table(ROOT/f'dataset/{"context_v3" if name=="context" else "partition_v2"}/episodes.parquet').to_pylist();episodes=[e for e in episodes if e['video_id']==v];cases=proposals(v,rows,episodes);rows,h,variance,events=align(rows,data);diagnosis=diagnose(cases,rows,old,events);params=json.loads((source/'calibration/model.json').read_text())['32'];states=[];coefficients=[];fail=collections.Counter()
 for case in cases:
  ids=segment_indices(case,rows);supported=False;cert=None;c=np.zeros((4,67));start=case['window_end_us'];reason='no_usable_geometry';latest=None;epoch=None;allocation=[];roles=[];parents=[];complete=False;neff=0
  if ids:
   last=rows[ids[-1]];epoch=last['candidate_epoch'];ids=[i for i in ids if rows[i]['candidate_epoch']==epoch];start=rows[ids[0]]['timestamp_us'];latest=rows[ids[-1]]['timestamp_us'];duration=(case['window_end_us']-start)/1e6;allocation=last['allocation'];roles=last['candidate_roles'];parents=last['candidate_parents'];complete=all(rows[i]['source_complete'] for i in ids)
   if len(ids)<8:reason='too_few_epoch_observations'
   elif duration<.3 or (latest-start)/1e6<.3-1e-9:reason='short_stable_epoch'
   elif case['window_end_us']-latest>100000:reason='stale_epoch'
   elif max(np.diff([rows[i]['timestamp_us'] for i in ids]))>120000:reason='epoch_observation_gap'
   else:
    time=(np.array([rows[i]['timestamp_us'] for i in ids])-start)/(duration*1e6);f=fit(time,h[ids],variance[ids],np.array([rows[i]['prior'] for i in ids]),np.array([rows[i]['quality'] for i in ids]),params['options'],params['feature_scale']);neff=f['neff']
    if neff<6 or f['condition']>=1e10:reason='ill_conditioned_epoch'
    else:
     # Only the separate geometry view is rolled back; no covariance is moved.
     for amount in [1.,.75,.25,0.]:
      trial=f['coefficients'].copy();trial[:,2:-1]*=amount;trial[0,2:-1]+=(1-amount)*h[ids[-1],2:-1];trial=trial.astype(np.float32);test=certify(trial,allocation,roles,parents,window_seconds=duration,max_depth=8,max_nodes=255)
      if test['certified']:c=trial;cert=test;supported=complete;reason=None if complete else 'unrepresented_source_rings';break
     if cert is None:reason='certificate_unresolved'
  duration=(case['window_end_us']-start)/1e6;baseline=old[(case['entity_track_id'],case['window_end_us'])];fail[reason or 'accepted']+=1
  states.append(dict(video_id=v,split=split(v),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],window_start_us=start,window_seconds=duration,latest_input_us=latest,epoch=epoch,valid=supported,allocation=allocation,roles=roles,parents=parents,source_complete=complete,certificate_json=json.dumps(cert),failure_reason=reason,row_index=len(states),baseline_full_history_valid=baseline['valid'],baseline_complete_certified=baseline['geometry_training_eligible'],history_scope='latest stable topology epoch only; minimum 0.3 seconds',identity_verified=False,geometry_covariance_available=False,ring_ids=last['persistent_ring_ids'] if ids else [],effective_samples=float(neff)));coefficients.append(c)
  if len(states)%64==0:guard()
 dest=D/name;dest.mkdir(exist_ok=True);save(dest/f'{v}.geometry.parquet',states);array(dest/f'{v}.coefficients.npy',coefficients);save(R/f'events.{name}.{v}.parquet',events);save(R/f'epoch_failures.{name}.{v}.parquet',diagnosis)
 return dict(source=name,video_id=v,diagnosed=len(diagnosis),causes=dict(collections.Counter(r['cause_class'] for r in diagnosis)),attempts=len(states),piecewise_complete_certified=sum(r['valid'] for r in states),baseline_complete_certified=sum(r['baseline_complete_certified'] for r in states),full_window_unchanged_topology_candidates=sum(r['same_candidate_epoch'] and not r['source_topology_changed'] for r in diagnosis),reasons=dict(fail))

def run(monitor):
 result=pool(worker,[(s,v) for s in SOURCES for v in VIDEOS],monitor,workers=7);result.sort(key=lambda r:(r['source'],r['video_id']));atomic_json(R/'correspondence.json',dict(videos=result,source_graph_event_not_physical_truth=True,minimum_epoch_seconds=.3,old_full_history_requirement_unchanged=True,new_mode='optional variable-duration latest-epoch geometry',identity_verified=False,track2_started=False,finalized=False))
