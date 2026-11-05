"""Source-identity association diagnostics, contact review and bounded acceptance."""
import collections,csv,json,shutil
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from track1_lab.topology_certificate import input_hashes,evaluate_curves
from . import *


def association_worker(v):
 from .motion import switching,predict
 source=SOURCES['context'];rows=pq.read_table(source/'spatial'/f'{v}.parquet').to_pylist();meta=pq.read_table(source/'appearance'/f'{v}.parquet').to_pylist();lookup={r['mask_uid']:r['row_index'] for r in meta};appearance=np.load(source/'appearance'/f'{v}.projected.npy',mmap_mode='r');bytime=collections.defaultdict(list);bytrack=collections.defaultdict(list)
 for r in rows:bytime[r['timestamp_us']].append(r);bytrack[r['entity_track_id']].append(r)
 times=np.array(sorted(bytime));episodes=[e for e in pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist() if e['video_id']==v];contacts=pq.read_table(ROOT/f'canonical/v1/contacts.{v}.parquet').to_pylist();contacttime=collections.defaultdict(list)
 for r in contacts:
  if r['contact_status']=='resolved_target':contacttime[r['timestamp_us']].append(r)
 result=[]
 for e in episodes:
  for cutoff in range(e['start_us']+1000000,e['end_us']-500000,500000):
   past=times[(times<=cutoff)&(times>=cutoff-100000)]
   if not len(past):continue
   before=bytime[int(past[-1])]
   for gap in [.1,.25,.5]:
    candidate=times[(times>cutoff)&(times<e['end_us'])&(abs(times-cutoff-round(gap*1e6))<=40000)]
    if not len(candidate):continue
    ts=int(candidate[np.argmin(abs(candidate-cutoff-round(gap*1e6)))]);after=bytime[ts];c0=np.array([r['centre_native'] for r in before]);c1=np.array([r['centre_native'] for r in after]);scales=np.maximum(np.array([r['scale_native'] for r in before])[:,None],np.array([r['scale_native'] for r in after])[None]);distance=np.linalg.norm(c0[:,None]-c1[None],axis=-1)/np.maximum(scales,20);size=abs(np.log(np.maximum(np.array([r['scale_native'] for r in before])[:,None],1)/np.maximum(np.array([r['scale_native'] for r in after])[None],1)));geom=distance+.5*size
    a=np.array([appearance[lookup[r['mask_uid']],0] for r in before]);b=np.array([appearance[lookup[r['mask_uid']],0] for r in after]);a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-10);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-10);app=1-a@b.T;motion=np.empty_like(geom);supported=[]
    for i,r in enumerate(before):
     history=[x for x in bytrack[r['entity_track_id']] if cutoff-650000<=x['timestamp_us']<=cutoff]
     try:
      bank=switching(np.array([x['timestamp_us']/1e6 for x in history]),np.array([x['centre_native'] for x in history]),np.ones((len(history),2))*4,cutoff/1e6);point,cov=predict(bank,(ts-cutoff)/1e6);delta=c1-point;motion[i]=np.sqrt(np.einsum('ni,ij,nj->n',delta,np.linalg.inv(cov+np.eye(2)*400),delta));supported.append(True)
     except ValueError:motion[i]=distance[i];supported.append(False)
    source_ids={r['entity_track_id'] for r in before};target_ids={r['mask_uid']:r['entity_track_id'] for r in after}
    for method,cost in [('geometry',geom),('geometry_appearance',geom+app),('geometry_appearance_motion_uncertainty',.5*geom+app+motion)]:
     matrix=np.c_[cost.T,np.full((len(after),len(after)),3.)];ri,ci=linear_sum_assignment(matrix);assigned={after[i]['entity_track_id']:before[j]['entity_track_id'] if j<len(before) and cost[j,i]<3 else None for i,j in zip(ri,ci)};known=[r['entity_track_id'] for r in after if r['entity_track_id'] in source_ids];covered=sum(assigned[x] is not None for x in known);correct=sum(assigned[x]==x for x in known);unknown=[r['entity_track_id'] for r in after if r['entity_track_id'] not in source_ids];unknown_accepted=sum(assigned[x] is not None for x in unknown);contact_known=contact_covered=contact_wrong=0
     for c in contacttime[ts]:
      hand=target_ids.get(c['hand_mask_uid']);obj=target_ids.get(c['target_mask_uid'])
      if hand in known and obj in known:
       contact_known+=1
       if assigned[hand] is not None and assigned[obj] is not None:contact_covered+=1;contact_wrong+=assigned[hand]!=hand or assigned[obj]!=obj
     result.append(dict(video_id=v,split=split(v),context_id=e['context_id'],cutoff_us=cutoff,target_time_us=ts,gap_seconds=gap,method=method,known_source_entities=len(known),assigned_known=int(covered),source_chain_correct=int(correct),source_chain_wrong=int(covered-correct),source_chain_abstained=len(known)-covered,unseen_source_entities=len(unknown),unseen_wrongly_assigned=int(unknown_accepted),source_contact_pairs=contact_known,source_contact_pairs_assigned=contact_covered,source_contact_identity_mismatches=int(contact_wrong),motion_supported_sources=sum(supported),past_sources=len(before),identity_reference='provisional annotation chain only',independent_idf1=False,independent_contact_swap_reference=False))
 appearance._mmap.close();save(R/f'association.{v}.parquet',result);return result


def contact_review():
 from track1_lab.geometry_contract import shape
 from track1_retrain.context import CTX
 episodes=pq.read_table(CTX/'episodes.parquet').to_pylist();proposals=[]
 for v in VIDEOS:
  contexts=[e for e in episodes if e['video_id']==v];contacts=pq.read_table(ROOT/f'canonical/v1/contacts.{v}.parquet').to_pylist();selected=[c for c in contacts if c['timestamp_us'] is not None and any(e['start_us']<=c['timestamp_us']<e['end_us'] for e in contexts)];wanted={uid for c in selected for uid in [c['hand_mask_uid'],c['target_mask_uid']] if uid};source={r['mask_uid']:r for r in pq.read_table(ROOT/f'canonical/v2/masks.{v}.parquet',columns=['mask_uid','contours_json','annotation_width','annotation_height'],filters=[('mask_uid','in',list(wanted))]).to_pylist()} if wanted else {}
  for c in selected:
   hand=source.get(c['hand_mask_uid']);obj=source.get(c['target_mask_uid']);distance=None;overlap=None
   if hand and obj:
    h,_=shape(hand['contours_json'],[1920/hand['annotation_width'],1080/hand['annotation_height']]);o,_=shape(obj['contours_json'],[1920/obj['annotation_width'],1080/obj['annotation_height']]);distance=float(h.boundary.distance(o.boundary)) if not h.is_empty and not o.is_empty and h.boundary is not None and o.boundary is not None else None;overlap=bool(h.intersection(o).area>0) if distance is not None else None
   e=next(e for e in contexts if e['start_us']<=c['timestamp_us']<e['end_us']);proposals.append(dict(video_id=v,split=split(v),context_id=e['context_id'],timestamp_us=c['timestamp_us'],contact_uid=c['contact_uid'],hand_mask_uid=c['hand_mask_uid'],target_mask_uid=c['target_mask_uid'],source_kind=c['source_kind'],source_contact_status=c['contact_status'],source_boundary_distance_native_px=distance,source_masks_overlap=overlap,physical_signed_distance=None,physical_contact_phase=None,occlusion_label=None,high_speed_label=None,independent_reference_available=False))
 save(R/'contact_proposals.parquet',proposals);pack=R/'independent_review';pack.mkdir(exist_ok=True)
 # Copy only blank templates; preserve any existing reviewer submissions.
 for p in (ROOT/'reports/track1_readiness/independent_review').glob('*'):
  if p.is_file() and not (pack/p.name).exists():shutil.copy2(p,pack/p.name)
 fields=['contact_uid','context_id','timestamp_us','hand_mask_uid','target_mask_uid','phase','native_boundary_a_path','native_boundary_b_path','boundary_a_sha256','boundary_b_sha256','signed_distance_native_px','distance_convention','contact_sign','visibility','reviewer','exclude_reason'];template=pack/'contact_boundaries_A.csv'
 for reviewer in ['A','B']:
  path=pack/f'contact_boundaries_{reviewer}.csv'
  if not path.exists():
   with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for r in proposals:writer.writerow({k:r[k] if k in r and k in fields[:5] else '' for k in fields})
 checks=[]
 for reviewer in ['A','B']:
  anchors=list(csv.DictReader((pack/f'anchors_{reviewer}.csv').open()));completed=[r for r in anchors if r['reviewer'] and r['anchor_x_native_px'] and r['anchor_y_native_px']];valid=[]
  for r in completed:
   try:
    x=float(r['anchor_x_native_px']);y=float(r['anchor_y_native_px']);assert np.isfinite([x,y]).all() and 0<=x<1920 and 0<=y<1080 and r['visibility'];valid.append(r)
   except (ValueError,AssertionError):continue
  checks.append(dict(reviewer_slot=reviewer,submitted_nonblank=len(completed),syntactically_valid=len(valid),independently_accepted=0))
 atomic_json(R/'independent_evaluation.json',dict(source_contact_proposals=len(proposals),source_contacts_not_boundary_ground_truth=True,reviewer_checks=checks,accepted_independent_references=0,reliability_adjudication_pending=True,required_phases=['pre_contact','first_contact','sustained_contact','release','occluded_contact','high_speed_contact','small_object_contact'],signed_distance_convention_pending_preregistration=True,independent_identity_idf1=None,independent_contact_error=None,independent_velocity_coverage=None,fresh_evaluation_reserved_not_acquired=True,automatic_acceptance=False))


def run(monitor):
 results=pool(association_worker,VIDEOS,monitor,workers=5);records=sorted([r for rr in results for r in rr],key=lambda r:(r['video_id'],r['context_id'],r['cutoff_us'],r['target_time_us'],r['method']));save(R/'association.parquet',records);summary=[]
 for part in ['train','validation']:
  for method in ['geometry','geometry_appearance','geometry_appearance_motion_uncertainty']:
   rs=[r for r in records if r['split']==part and r['method']==method];byvideo=[]
   for v in sorted({r['video_id'] for r in rs}):
    local=[r for r in rs if r['video_id']==v];known=sum(r['known_source_entities'] for r in local);assigned=sum(r['assigned_known'] for r in local);correct=sum(r['source_chain_correct'] for r in local);byvideo.append(dict(video_id=v,known_entities=known,assigned_entities=assigned,source_chain_correct=correct,unconditional_correct_fraction=correct/max(known,1),abstention_fraction=1-assigned/max(known,1),source_contact_pairs=sum(r['source_contact_pairs'] for r in local),source_contact_pairs_assigned=sum(r['source_contact_pairs_assigned'] for r in local),source_contact_identity_mismatches=sum(r['source_contact_identity_mismatches'] for r in local)))
   summary.append(dict(split=part,method=method,videos=byvideo,mean_video_source_chain_correct_fraction=float(np.mean([r['unconditional_correct_fraction'] for r in byvideo])) if byvideo else None,mean_video_abstention_fraction=float(np.mean([r['abstention_fraction'] for r in byvideo])) if byvideo else None))
 atomic_json(R/'association.json',dict(summary=summary,reference='Provisional source annotation-chain agreement; not independent identity accuracy, IDF1 or HOTA',matched_queries=True,source_ids_not_used_as_association_features=True,correct_past_source_memory_assumed=True,online_tracking_error_propagation_tested=False,weights_fixed_before_validation=True,source_contact_consequential_mismatches_are_proxies=True,promoted=False,missing_reappearance_and_occlusion_truth=True));contact_review();monitor.check()


def verify(monitor):
 from .store import TieredDataset
 checked=0;reader_checks=0
 for source in SOURCES:
  with TieredDataset(source) as ds:
   for v in VIDEOS:
    rows=pq.read_table(D/source/f'{v}.geometry.parquet').to_pylist();a=np.load(D/source/f'{v}.coefficients.npy',mmap_mode='r');assert np.isfinite(a).all()
    for r in rows:
     if not r['valid']:continue
     assert r['source_complete'] and r['latest_input_us']<=r['window_end_us'] and .3<=r['window_seconds']<=1.+1e-9;cert=json.loads(r['certificate_json']);expected=input_hashes(a[r['row_index']],r['allocation'],r['roles'],r['parents'],r['window_seconds']);assert cert['certified'] and all(cert.get(k)==value for k,value in expected.items());checked+=1
    a._mmap.close()
    tiered=[r for r in pq.read_table(D/source/'tiered_states.parquet').to_pylist() if r['video_id']==v and r['valid'] and r['m32_available']]
    for r in tiered[:3]:state=ds.get_state(v,r['entity_track_id'],r['window_end_us']);ds.geometry(state,r['window_end_us']);reader_checks+=1
    monitor.check()
  for part in ['train','validation']:
   for length in [4,8,16]:
    table=pq.read_table(D/source/'indices'/f'{part}.length{length}.parquet');assert {'geometry32_mask','geometry32_history_seconds','all_core16_geometry_complete_certified'}<=set(table.column_names);rows=table.to_pylist()
    for r in rows:
     assert len(r['end_times_us'])==length and np.all(np.diff(r['end_times_us'])==250000);assert len(r['geometry32_mask'])==length;assert not r['identity_verified'];assert r['single_compound_epoch']==(r['all_compound_geometry_available'] and all(r['geometry32_same_epoch_pairs']))
 camera=[];motion=[]
 for v in VIDEOS:
  camera+=pq.read_table(D/f'camera.{v}.parquet').to_pylist();motion+=pq.read_table(D/f'motion_states.{v}.parquet').to_pylist()
 for r in motion:
  assert r['latest_input_us']<=r['cutoff_us'];w=np.array(r['hypothesis_weights']);assert np.isfinite(w).all() and np.all(w>=0) and abs(w.sum()-1)<1e-9
 atomic_json(R/'verification.json',dict(certificate_states_checked=checked,tiered_reader_checks=reader_checks,motion_states_checked=len(motion),background_frames=len(camera),numerical_checks_passed=True,independent_scientific_acceptance=False,clean_full_rebuild_performed=False,track2_started=False,finalized=False));report();monitor.check()


def report():
 from html import escape
 def table(rows):
  if not rows:return '<p>No measured support.</p>'
  return '<table><tr>'+''.join('<th>'+escape(k)+'</th>' for k in rows[0])+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r.values())+'</tr>' for r in rows)+'</table>'
 def read(name):return json.loads((R/name).read_text())
 a=read('correspondence.json');b=read('motion.json');c=read('tiered.json');d=read('independent_evaluation.json');association=read('association.json');syn=read('synthetic_motion.json');counts=[];causes=[]
 for source in SOURCES:
  local=[r for r in a['videos'] if r['source']==source];counter=collections.Counter()
  for r in local:counter.update(r['causes'])
  causes.extend(dict(source=source,cause=k,windows=value) for k,value in counter.items())
  for part in ['train','validation']:
   rows=[r for r in c['continuity'] if r['source']==source and r['split']==part and r['length']==16];counts.append(dict(source=source,split=part,core_length16_sequences=sum(r['core_sequences'] for r in rows),all_optional_compound_geometry=sum(r['all_compound_geometry_available'] for r in rows),single_compound_epoch=sum(r['single_compound_epoch_sequences'] for r in rows)))
 assoc=[{k:r[k] for k in ['split','method','mean_video_source_chain_correct_fraction','mean_video_abstention_fraction']} for r in association['summary']];metrics=[r for r in b['matched_metrics'] if r['split']=='validation'];stress=[r for r in syn['summary'] if r['split']=='synthetic_validation'];notes=['Open bounded Track 1 candidates; no finalization, baseline replacement or Track 2 training.','A: Source-mask ring changes are observed annotation events. Physical birth/death, occlusion and identity causes cannot be resolved from these labels. Variable-duration geometry is a separate latest-epoch view, not a recovered one-second history.','C: M16 carries continuity. M32 is optional, certificate-bound, and its duration, availability and discrete epoch boundaries remain explicit. Missing geometry is None; no joint calibrated geometry covariance is invented.','B: Fixed causal hypothesis bank and confidence-gated background alignment are unpromoted ablations. Hypothesis weights are uncalibrated model weights. The camera-relative view is 2D alignment, not metric object motion.','D: Association scores use provisional annotation-chain identities. Contact relations only select review cases; they cannot certify boundary distances, physical events or consequential identity swaps.','All sequence counts overlap. Participant/video/episode are the uncertainty units. Two validation participants are insufficient for strong independent claims.','95% simultaneous source-trajectory split-conformal calibration has only five training participant units. The finite-sample order statistic is out of range, so the honest bound is unbounded, not a finite 95% guarantee.','Genuinely fresh participants remain reserved but unacquired; no fresh labels are used for development. Independent motion/contact/identity review remains pending.']
 body='<h1>Bounded Track 1 remediation</h1>'+''.join('<p>'+escape(n)+'</p>' for n in notes)+'<h2>A: all original compound epoch failures</h2>'+table(causes)+'<h2>C: tiered length-16 availability</h2>'+table(counts)+'<p>All optional-geometry coverage may contain explicit topology resets. Only the last column retains one continuous compound epoch.</p><h2>B: matched source-centroid motion</h2>'+table(metrics)+'<h2>B: exact synthetic motion stress</h2>'+table(stress)+'<h2>D: source-identity proxy association</h2>'+table(assoc)+'<p>Accepted independent references: '+str(d['accepted_independent_references'])+'. Source contact proposals: '+str(d['source_contact_proposals'])+'.</p><h2>Evidence</h2><p>'+' · '.join(f'<a href="{name}">{name}</a>' for name in ['correspondence.json','geometry_quality.json','camera.json','motion.json','synthetic_motion.json','tiered.json','association.json','independent_evaluation.json','verification.json','independent_review/README.md'])+'</p>'
 (R/'dashboard.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bounded Track 1 remediation</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#111820;color:#e3eaf0}a{color:#8ed6ff}th,td{padding:9px;text-align:left;border-bottom:1px solid #465360}table{border-collapse:collapse}p{line-height:1.5}</style>'+body+'</html>')
 (R/'REMEDIATION.md').write_text('# Bounded Track 1 remediation\n\n'+'\n\n'.join(notes)+'\n\n## Original epoch-failure diagnosis\n\n'+json.dumps(causes,indent=2)+'\n\n## Tiered sequence availability\n\n'+json.dumps(counts,indent=2)+'\n\n[Dashboard](dashboard.html) contains the measured motion/association comparisons and scientific limits.\n')
 atomic_json(D/'status.json',dict(ready=True,ready_scope='numerically checked experimental tiered Track 1 reader; candidates not promoted',finalized=False,scientific_acceptance=False,track2_started=False,independent_references_ready=False,kinematics_calibrated=False,geometry_covariance_available=False,continuous_geometry_epoch_bridging_allowed=False))


def geometry_quality(monitor):
 from .store import TieredDataset
 from track1.numerics import basis
 from track1_lab.motion import project
 from track1_retrain.evaluation import matched_masks
 from track1_retrain.common import rank
 results=[];attempts=[]
 for v in VIDEOS:
  meta=pq.read_table(D/'context/tiered_states.parquet').to_pylist();candidates=[r for r in meta if r['video_id']==v and r['valid']];candidates=sorted(candidates,key=lambda r:rank(r['entity_track_id']+str(r['window_end_us'])))[:64];source=pq.read_table(ROOT/f'work/track1_readiness/inputs.{v}.parquet').to_pylist();bytrack=collections.defaultdict(list)
  for r in source:bytrack[r['entity_track_id']].append(r)
  camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());focal=np.sqrt(camera['intrinsic_native'][0][0]*camera['intrinsic_native'][1][1])
  with TieredDataset('context') as ds:
   for r in candidates:
    targets=[t for t in bytrack[r['entity_track_id']] if r['window_end_us']-40000<=t['timestamp_us']<=r['window_end_us']]
    if not targets:attempts.append(dict(video_id=v,entity_track_id=r['entity_track_id'],window_end_us=r['window_end_us'],reason='no_recent_source_mask'));continue
    target=targets[-1];state=ds.get_state(v,r['entity_track_id'],r['window_end_us']);methods=[];names=[]
    def native(coeff,metadata,timestamp):
     phase=(timestamp-metadata['window_start_us'])/(metadata['window_seconds']*1e6);normalized=evaluate_curves(coeff,metadata['allocation'],metadata['roles'],phase,512);h=basis(2*phase-1)@coeff;centre=project(h[:2],camera);scale=focal*np.exp(h[-1]);return [p*scale+centre for p in normalized]
    core=state['core16'];m=core['metadata']
    if m['certified_geometry_valid']:
     names.append('core16_canonical');methods.append((native(core['geometry_coefficients'],m,target['timestamp_us']),m['roles'],m['parents']))
    if state['m32_available']:
     g=state['geometry32'];m=g['metadata']
     if target['timestamp_us']>=m['window_start_us']:
      names.append('tiered32_canonical');methods.append((native(g['coefficients'],m,target['timestamp_us']),m['roles'],m['parents']))
    full=ds.base.get_state(v,r['entity_track_id'],r['window_end_us'],model=32,require_functional=False);m=full['metadata']
    if m['geometry_training_eligible']:
     names.append('full32_canonical');methods.append((native(full['geometry_coefficients'],m,target['timestamp_us']),m['roles'],m['parents']))
    metrics=matched_masks(target,methods,np.array(camera['native_size'])) if methods else []
    for name,metric in zip(names,metrics):results.append(dict(video_id=v,split=split(v),entity_track_id=r['entity_track_id'],window_end_us=r['window_end_us'],target_mask_uid=target['mask_uid'],method=name,iou=metric['iou'],boundary_fscore_2px=metric['boundary_fscore_2px'],target_source_kind=target['source_kind'],target_used_as_input=True,optional_geometry_kind=state['geometry32']['kind'] if state['m32_available'] else None,independent_ground_truth=False))
    attempts.append(dict(video_id=v,entity_track_id=r['entity_track_id'],window_end_us=r['window_end_us'],reason=None,methods_available=names))
  monitor.check()
 save(R/'geometry_quality.parquet',results);save(R/'geometry_quality_attempts.parquet',attempts);group=collections.defaultdict(list)
 for r in results:group[(r['video_id'],r['entity_track_id'],r['window_end_us'])].append(r)
 summary=[]
 for cohort,required in [('core_tiered',{'core16_canonical','tiered32_canonical'}),('all_three',{'core16_canonical','tiered32_canonical','full32_canonical'}),('new_segment_core',{'core16_canonical','tiered32_canonical'})]:
  selected=[r for rows in group.values() if {r['method'] for r in rows}>=required and (cohort!='new_segment_core' or rows[0]['optional_geometry_kind']=='latest_stable_topology_segment') for r in rows if r['method'] in required]
  for part in ['train','validation']:
   for name in sorted(required):
    rows=[r for r in selected if r['method']==name and r['split']==part];video=[]
    for v in sorted({r['video_id'] for r in rows}):
     local=[r for r in rows if r['video_id']==v];video.append((np.mean([r['iou'] for r in local]),np.mean([r['boundary_fscore_2px'] for r in local])))
    if video:summary.append(dict(cohort=cohort,split=part,method=name,probes=len(rows),mean_video_iou=float(np.mean([x[0] for x in video])),mean_video_bf2=float(np.mean([x[1] for x in video]))))
 atomic_json(R/'geometry_quality.json',dict(matched_metrics=summary,scope='Canonical decoder source-mask reconstruction at retained input timestamps; in-sample, not held-out segmentation or forecasting',selected_without_target_score=True,availability_reported_separately=True,independent_quality_acceptance=False))
