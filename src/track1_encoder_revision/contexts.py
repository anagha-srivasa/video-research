"""Source-backed stable-context expansion; never bridges an observed graph change."""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from data_v2.geometry import normalize
from track1_retrain.temporal import fit,prepare
from track1_retrain.spatial import align as align_spatial
from track1_retrain.store import sequences
from track1_lab.variants import collocation
from track1_lab.topology_certificate import certify
from track1_lab.geometry_contract import topology_signature
from .experiments import ROOT,D,R,W,CFG,VIDEOS,save,array,rank


def plan_worker(video):
 rows=pq.read_table(ROOT/f'canonical/v2/spatial_metadata.{video}.parquet',columns=['video_id','mask_uid','entity_track_id','timestamp_us','source_kind','identity_evidence','row_index','source_component_count','hole_count','width','height']).to_pylist();groups=collections.defaultdict(dict)
 prior={}
 for directory in ['partition_v2','context_v3']:
  for e in pq.read_table(ROOT/f'dataset/{directory}/episodes.parquet').to_pylist():
   if e['video_id']==video:prior[(e['start_us'],e['end_us'])]=True
 preference={'manual_sparse':0,'dense_gt_endpoint':1,'dense_interpolated':2}
 for r in rows:
  t=r['timestamp_us'];track=r['entity_track_id']
  if t is None:continue
  old=groups[track].get(t)
  if old is None or preference.get(r['source_kind'],3)<preference.get(old['source_kind'],3):groups[track][t]=r
 candidates=[];duration=round(CFG['context_seconds']*1e6)
 for track,bytime in groups.items():
  local=sorted(bytime.values(),key=lambda r:r['timestamp_us']);times=np.array([r['timestamp_us'] for r in local])
  if len(local)<100:continue
  for start in range(int(times[0]),int(times[-1])-duration+1,duration):
   if any(start<b and start+duration>a for a,b in prior):continue
   rs=[r for r in local if start<=r['timestamp_us']<start+duration]
   if len(rs)<100:continue
   tt=np.array([r['timestamp_us'] for r in rs]);counts={(r['source_component_count'],r['hole_count']) for r in rs}
   if tt[-1]-tt[0]<duration-200000 or max(np.diff(tt))>120000 or len(counts)!=1:continue
   candidates.append(dict(video_id=video,entity_track_id=track,start_us=start,end_us=start+duration,mask_uids=[r['mask_uid'] for r in rs],source_observations=len(rs),records=rs,rank=rank((video,track,start))))
 candidates=sorted(candidates,key=lambda r:r['rank'])[:CFG['context_candidates_per_video']];wanted={u for c in candidates for u in c['mask_uids']};masks={}
 columns=['mask_uid','geometry_complete','contours_json','segments_raw','area_px','centroid_x','centroid_y','annotation_width','annotation_height','geometry_status','component_count','hole_count','source_kind','dependency_group']
 for batch in pq.ParquetFile(ROOT/f'canonical/v2/masks.{video}.parquet').iter_batches(batch_size=256,columns=columns):
  for r in batch.to_pylist():
   if r['mask_uid'] in wanted:masks[r['mask_uid']]=r
 ledger=[];chosen=None
 for candidate in candidates:
  rs=[masks[u] for u in candidate['mask_uids'] if u in masks];signatures={json.dumps(topology_signature(normalize(r['contours_json'])),sort_keys=True) for r in rs};complete=len(rs)==candidate['source_observations'] and all(r['geometry_complete'] for r in rs);valid=complete and len(signatures)==1
  ledger.append({k:v for k,v in candidate.items() if k not in ['records','mask_uids']}|dict(graph_stable=valid,complete=complete,source_graph_signatures=len(signatures),physical_identity_verified=False))
  if valid and chosen is None:chosen=candidate
 save(R/f'context_candidates.{video}.parquet',ledger)
 if chosen is None:return dict(video_id=video,candidates=len(candidates),selected=False,reason='no_complete_source_graph_stable_context')
 inputs=[]
 for r in chosen['records']:
  m=masks[r['mask_uid']];inputs.append(dict(r,**{k:v for k,v in m.items() if k not in ['source_kind','mask_uid']},baseline_row_index=r['row_index'],split='train' if video in __import__('track1_retrain.common',fromlist=['CFG']).CFG['train'] else 'validation'))
 work=W/'contexts';work.mkdir(exist_ok=True);save(work/f'{video}.inputs.parquet',inputs)
 frames=[]
 for batch in pq.ParquetFile(ROOT/f'canonical/v1/frames.{video}.parquet').iter_batches(batch_size=4096,columns=['timestamp_us','pts_raw','native_frame_uid']):
  frames.extend(dict(r,video_id=video,source_path=f'dataset/videos/{video}.MP4') for r in batch.to_pylist() if chosen['start_us']<=r['timestamp_us']<chosen['end_us'])
 save(D/f'{video}.context_frames.parquet',frames)
 return {k:v for k,v in chosen.items() if k not in ['records','mask_uids','rank']}|dict(selected=True,candidates=len(candidates),indexed_native_frames=len(frames),new_rgb_bytes=0,source_graph_verified=True,physical_identity_verified=False)


def refit_worker(context):
 if not context['selected']:return context
 video=context['video_id'];work=W/'contexts';inputs=pq.read_table(work/f'{video}.inputs.parquet').to_pylist()
 from track1_retrain import spatial
 spatial.WORK=work
 result=spatial.worker((video,0,inputs,dict(scope='source_backed_stable_context',mask_uids=[r['mask_uid'] for r in inputs])))
 rows=align_spatial(pq.read_table(work/'spatial_chunks'/f'{video}.00000.parquet').to_pylist());data={}
 for key in ['h','variance','controls']:
  for n in [16,32]:data[key+str(n)]=np.array([r.pop(key+str(n)) for r in rows])
 for i,r in enumerate(rows):r['row_index']=i
 model=json.loads((ROOT/'derived/track1_finetuned/v2/calibration/model.json').read_text());states=[];coefficients=[];geometry=[];support=[];xi_values=[];kin_position=[];kin_velocity=[];kin_cov=[]
 start=((context['start_us']+1000000+249999)//250000)*250000
 for end in range(start,context['end_us']+1,250000):
  local=[i for i,r in enumerate(rows) if end-1000000<=r['timestamp_us']<=end]
  for n in [16,32]:
   case=dict(indices=local,window_end_us=end);valid=False;reason=None;cert=None;p=None;c=np.zeros((4,2*n+3),np.float32);gc=c.copy();epoch=None;roles=[];parents=[];allocation=[n]
   try:
    p=prepare(case,rows,{key:data[key+str(n)] for key in ['h','variance','controls']},n,global_gauge=True);f=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],model[str(n)]['options'],np.array(model[str(n)]['feature_scale']))
    if f['neff']<6 or f['condition']>=1e10:raise ValueError('ill_conditioned_or_low_effective_support')
    c=f['coefficients'].astype(np.float32);epoch=rows[p['ids'][0]][f'epoch{n}'];allocation=p['allocation'];roles=p['roles'];parents=p['parents'];valid=True
    from .geometry import repair
    repaired=repair(c,p['h'][-1],allocation,roles,parents,p['time'],p['h']);gc=repaired['coefficients'] if repaired['coefficients'] is not None else c;cert=repaired['certificate']
   except (ValueError,np.linalg.LinAlgError) as e:reason=str(e)
   from track1_lab.causal_kinematics import estimate
   times=(np.array([rows[i]['timestamp_us'] for i in local])-end+1000000)/1e6;kin=estimate(times,data['h'+str(n)][local,:2],data['variance'+str(n)][local,:2],1.)
   kin_position.append(kin.position if kin.support_valid else np.zeros(2));kin_velocity.append(kin.velocity if kin.support_valid else np.zeros(2));kin_cov.append(kin.covariance if kin.support_valid else np.zeros((4,4)));nodes=collocation(c.astype(float),1.);xi_values.append(np.pad(nodes['xi'],((0,0),(0,132-nodes['xi'].shape[1]))))
   r=dict(video_id=video,split=inputs[0]['split'],entity_track_id=context['entity_track_id'],model=n,window_end_us=end,window_start_us=end-1000000,window_seconds=1.,valid=valid,epoch=epoch,row_index=len(states),source_complete=bool(p is not None and all((rows[i]['source_rings']==1 if n==16 else rows[i]['source_complete']) for i in p['ids'])),retained_mask_uids=[rows[i]['mask_uid'] for i in p['ids']] if p is not None else [],geometry_training_eligible=bool(valid and cert and cert['certified'] and p is not None and all((rows[i]['source_rings']==1 if n==16 else rows[i]['source_complete']) for i in p['ids'])),certificate_json=json.dumps(cert),failure_reason=reason,allocation=allocation,roles=roles,parents=parents,appearance_valid=False,kinematics_valid=kin.support_valid,latest_input_us=max([rows[i]['timestamp_us'] for i in p['ids']]) if p is not None else None,coordinate_mode='intrinsic_normalized_2D',identity_verified=False,independent_coverage_eligible=False,episode_id=rank((video,context['start_us']))[:20]);states.append(r);coefficients.append(np.pad(c,((0,0),(0,67-c.shape[1]))));geometry.append(np.pad(gc,((0,0),(0,67-gc.shape[1]))))
 save(D/f'{video}.context_states.parquet',states);array(D/f'{video}.context_coefficients.npy',coefficients);array(D/f'{video}.context_geometry.npy',geometry);array(D/f'{video}.context_xi.npy',xi_values);array(D/f'{video}.context_kinematic_position.npy',kin_position);array(D/f'{video}.context_kinematic_velocity.npy',kin_velocity);array(D/f'{video}.context_kinematic_cov.npy',kin_cov)
 counts=[]
 for n in [16,32]:
  rs=[r for r in states if r['model']==n]
  for length in [4,8,16]:
   seq=sequences(rs,length);save(R/f'context_sequences.{video}.m{n}.length{length}.parquet',seq) if seq else None;counts.append(dict(model=n,length=length,sequences=len(seq),all_certified=sum(r['all_geometry_valid'] for r in seq)))
 return dict(context,refitted_source_observations=len(rows),windows=len(states),valid=sum(r['valid'] for r in states),continuity=counts,appearance_extraction_pending=True,new_rgb_bytes=0)
