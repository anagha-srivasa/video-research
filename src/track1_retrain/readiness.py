"""Cause-resolved yield/continuity audit and prospective source-context selection."""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from .common import *
from .temporal import load,proposals,prepare,timing_eligible
from .store import sequences
R=ROOT/'reports/track1_readiness';W=ROOT/'work/track1_readiness'
for p in [R,W]:p.mkdir(exist_ok=True)

def audit(monitor):
 records=[];summary=[];continuity=[];unique=set();all_states=[]
 for v in VIDEOS:
  for n in [16,32]:
   rows,data=load(v,n);meta=pq.read_table(folder('functional')/f'{v}.m{n}.parquet').to_pylist();lookup={(r['entity_track_id'],r['window_end_us']):r for r in meta};all_states+=meta
   for case in proposals(v,rows):
    unique.add((v,case['entity_track_id'],case['window_end_us']));state=lookup[(case['entity_track_id'],case['window_end_us'])];ids=case['indices'];valid_ids=[i for i in ids if rows[i][f'fit{n}_valid']];times=np.array([rows[i]['timestamp_us'] for i in valid_ids]);source_timing=timing_eligible(case,rows);epochs={rows[i][f'epoch{n}'] for i in valid_ids};sigs={rows[i]['source_signature'] for i in valid_ids};allocs={tuple(rows[i]['allocation']) for i in valid_ids};by_kind=collections.defaultdict(set)
    for i in valid_ids:by_kind[rows[i]['source_kind']].add(rows[i]['source_signature'])
    provenance_disagreement=len(sigs)>1 and len(by_kind)>1 and all(len(s)==1 for s in by_kind.values());local_ok=False;local_reason=None
    if state['failure_reason']=='global_correspondence_epoch_change':
     try:prepare(case,rows,data,n);local_ok=True
     except ValueError as e:local_reason=str(e)
    cert=json.loads(state['certificate_json']);record=dict(video_id=v,split=split(v),model=n,entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],valid=state['valid'],failure_reason=state['failure_reason'] or 'valid',source_observations=len(ids),valid_spatial_observations=len(valid_ids),source_timing_eligible=source_timing,spatial_filter_reduced_support=source_timing and not state['valid'] and state['failure_reason'] in ['too_few_retained_observations','insufficient_retained_span','retained_gap'],retained_span_us=int(times[-1]-times[0]) if len(times) else 0,global_epoch_count=len(epochs),source_signature_count=len(sigs),allocation_count=len(allocs),observed_ambiguous_matching=sum(rows[i][f'correspondence_ambiguous{n}'] for i in valid_ids),provenance_conditioned_topology_disagreement=provenance_disagreement,window_local_reencoding_supported=local_ok,window_local_failure=local_reason,kinematics_supported=state['kinematics_valid'],source_complete=state['source_complete'],curve_certified=state['certified_geometry_valid'],complete_certified=state['geometry_training_eligible'],certificate_failure=cert.get('failure_code') if cert else None,recovery_scope='local reencoding does not verify cross-window identity or permit epoch bridging')
    records.append(record)
   selected=[r for r in records if r['video_id']==v and r['model']==n];timed=sum(r['source_timing_eligible'] for r in selected);summary.append(dict(video_id=v,split=split(v),model=n,attempts=len(selected),source_timing_eligible=timed,valid=sum(r['valid'] for r in selected),valid_fraction_of_timing_eligible=sum(r['valid'] and r['source_timing_eligible'] for r in selected)/max(timed,1),failure_counts=dict(collections.Counter(r['failure_reason'] for r in selected)),local_reencoding_candidates=sum(r['window_local_reencoding_supported'] for r in selected),independent_kinematics_candidates=sum(r['kinematics_supported'] and not r['valid'] for r in selected),source_incomplete_among_valid=sum(r['valid'] and not r['source_complete'] for r in selected),certification_failures_among_valid=sum(r['valid'] and not r['curve_certified'] for r in selected)))
   for length in [4,8,16]:
    seq=sequences(meta,length);tracks={(r['entity_track_id']) for r in seq};nonoverlap=0;last={}
    for s in seq:
     key=s['entity_track_id'];start=s['end_times_us'][0]
     if key not in last or start>last[key]:nonoverlap+=1;last[key]=s['end_times_us'][-1]
    continuity.append(dict(video_id=v,split=split(v),model=n,length=length,sliding_sequences=len(seq),tracks=len(tracks),nonoverlapping_state_blocks=nonoverlap,independence_certified=False))
   monitor.check()
 save(R/'attempt_loss.parquet',records);save(R/'continuity.parquet',continuity)
 paired=[]
 for v in VIDEOS:
  a={r['window_end_us'].__str__()+r['entity_track_id']:r for r in records if r['video_id']==v and r['model']==16};b={r['window_end_us'].__str__()+r['entity_track_id']:r for r in records if r['video_id']==v and r['model']==32}
  paired.append(dict(video_id=v,unique_opportunities=len(a),both_timing_eligible=sum(a[k]['source_timing_eligible'] and b[k]['source_timing_eligible'] for k in a),both_valid=sum(a[k]['valid'] and b[k]['valid'] for k in a),only_16_valid=sum(a[k]['valid'] and not b[k]['valid'] for k in a),only_32_valid=sum(b[k]['valid'] and not a[k]['valid'] for k in a),both_complete_certified=sum(a[k]['complete_certified'] and b[k]['complete_certified'] for k in a)))
 uncertainty=json.loads((REPORT/'evaluation.json').read_text())['uncertainty'];atomic_json(R/'loss_audit.json',dict(unique_window_opportunities=len(unique),model_window_attempts=len(records),by_video_model=summary,paired=paired,continuity=continuity,uncertainty_verified=uncertainty,notes=['Geometry completeness/certification are separate from numerical support.','Local reencoding candidates are diagnostics, not promoted persistent tracks.','Timing eligibility is source-derived, not independently preregistered coverage.','Sliding sequences and nonoverlap counts are not independent annotated episodes.'],track2_started=False))
 return summary

def context_plan(monitor):
 """Choose a small, targeted six-second supplement from existing full sources.
 No metric on reserved or validation predictions selects these generic contexts.
 """
 frames=pq.read_table(ROOT/'dataset/partition_v2/frames.parquet').to_pylist();old={v:{r['native_frame_uid'] for r in frames if r['video_id']==v} for v in VIDEOS};selected=[];extra=[];budget=32000000000;spent=0
 for v in VIDEOS:
  rows=pq.read_table(ROOT/f'canonical/v2/spatial_metadata.{v}.parquet').to_pylist();groups=collections.defaultdict(list)
  for r in rows:groups[r['entity_track_id']].append(r)
  native=pq.read_table(ROOT/f'canonical/v1/frames.{v}.parquet').to_pylist();nt=np.array([r['timestamp_us'] for r in native]);candidates=[]
  for track,rs in groups.items():
   rs.sort(key=lambda r:r['timestamp_us']);times=np.array([r['timestamp_us'] for r in rs]);
   if len(rs)<120:continue
   for centre in range(int(times[0])+3000000,int(times[-1])-3000000,3000000):
    lo=np.searchsorted(times,centre-3000000);hi=np.searchsorted(times,centre+3000000);local=rs[lo:hi];tt=times[lo:hi]
    if len(tt)<100 or tt[-1]-tt[0]<5700000 or max(np.diff(tt))>120000:continue
    signatures={(r['source_component_count'],r['hole_count']) for r in local};stable=len(signatures)==1;simple=stable and next(iter(signatures))==(1,0);name=local[0].get('entity_name','');is_hand='hand' in name.lower()
    candidates.append(dict(video_id=v,split=split(v),entity_track_id=track,centre_us=centre,start_us=centre-3000000,end_us=centre+3000000,source_observations=len(local),stable_count_signature=stable,simple_count_signature=simple,entity_name=name,is_hand=is_hand,selection_rank=rank(v+track+str(centre))))
  # Prefer observable stable hands, then other stable source chains. These are
  # prospective coverage candidates, not certified physical identities/events.
  candidates.sort(key=lambda c:(not c['simple_count_signature'],not c['is_hand'],c['selection_rank']));used=set(old[v]);chosen=[]
  for c in candidates:
   if any(abs(c['centre_us']-x['centre_us'])<6000000 for x in chosen):continue
   ids=np.flatnonzero((nt>=c['start_us'])&(nt<c['end_us']));fresh=[int(i) for i in ids if native[i]['native_frame_uid'] not in used];cost=len(fresh)*1920*1080*3
   if spent+cost>budget:continue
   c.update(context_id=rank(c['video_id']+c['entity_track_id']+str(c['centre_us']))[:24],new_native_frames=len(fresh),new_rgb_bytes=cost,physical_identity_verified=False)
   chosen.append(c);spent+=cost;used.update(native[i]['native_frame_uid'] for i in fresh)
   for i in fresh:extra.append(dict(native[i],split=split(v),storage_row=sum(1 for r in extra if r['video_id']==v)))
   if len(chosen)==3:break
  selected.extend(chosen);LOG.info('Prospective context %s: %d clips / %d new frames',v,len(chosen),sum(c['new_native_frames'] for c in chosen));monitor.check()
 save(W/'context_episodes.parquet',selected);save(W/'context_frames.parquet',extra);source_bytes=json.loads((ROOT/'reports/dataset_v2/capacity.json').read_text())['total_bytes'];assert source_bytes+spent<250000000000
 atomic_json(R/'context_plan.json',dict(contexts=selected,new_frames=len(extra),new_rgb_bytes=spent,existing_dataset_bytes=source_bytes,total_dataset_estimate_bytes=source_bytes+spent,budget_bytes=250000000000,minimum_seconds_for_16_states=4.75,selected_seconds=6,scope='targeted additional native RGB from existing source videos; prospective count/timing screen, not certified sequence recovery',source_label_selection='stable component/hole counts and timing; hand preference; no validation model scores',materialized=False,track2_started=False))
 return selected

def run(monitor):
 audit(monitor);context_plan(monitor)
