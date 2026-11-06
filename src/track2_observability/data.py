import collections,json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json
from track1_retrain.common import save
from track1.camera import undistort
from track2_feasibility.data import Data,BASE
from .common import C,D,R,OLD


def candidate_audit(records,cutoff,target,tolerance):
    future=[r for r in records if r['timestamp_us']>cutoff]
    nearest=min(future,key=lambda r:(abs(r['timestamp_us']-target),r['mask_uid'])) if future else None
    inside=[r for r in future if abs(r['timestamp_us']-target)<=tolerance]
    usable=[r for r in inside if r['complete'] and np.isfinite(r['position']).all()]
    chosen=min(usable,key=lambda r:(abs(r['timestamp_us']-target),r['mask_uid'])) if usable else None
    nearest_ok=nearest is not None and any(r['mask_uid']==nearest['mask_uid'] for r in usable)
    distances=[float(np.linalg.norm(np.array(a['position'])-b['position'])) for i,a in enumerate(usable) for b in usable[i+1:]]
    return dict(nearest_usable=nearest_ok,any_usable=bool(usable),candidate_count=len(inside),complete_candidates=len(usable),
        nearest_mask_uid=nearest['mask_uid'] if nearest else None,nearest_complete_mask_uid=chosen['mask_uid'] if chosen else None,
        maximum_complete_candidate_disagreement=max(distances) if distances else None,
        alternative_position=chosen['position'] if chosen else None,alternative_timestamp_us=chosen['timestamp_us'] if chosen else None)


def prepare():
    data=Data();raw=np.load(OLD/'recent_observation.npy');features=np.zeros((len(data.samples),4,11));audit=[];relations=[]
    features[:,:,:2]=raw[:,:,:2]-raw[:,-1:,:2];features[:,:,2:4]=raw[:,:,2:]
    old=pq.read_table(ROOT/'reports/track2_measurement/target_audit.parquet').to_pylist()
    episodes={r['episode_id']:r for r in pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist()}
    for video in sorted({s['video_id'] for s in data.samples}):
        rows=pq.read_table(BASE/'spatial'/f'{video}.parquet').to_pylist();wanted={r['mask_uid'] for r in rows};sources={}
        columns=['mask_uid','geometry_complete','centroid_x','centroid_y','annotation_width','annotation_height','entity_name_open']
        for batch in pq.ParquetFile(ROOT/f'canonical/v2/masks.{video}.parquet').iter_batches(columns=columns,batch_size=2048):
            for r in batch.to_pylist():
                if r['mask_uid'] in wanted:sources[r['mask_uid']]=r
        camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{video}.camera.json').read_text());k=np.array(camera['intrinsic_native']);dist=np.array(camera['distortion']);tracks=collections.defaultdict(list)
        for r in rows:
            source=sources[r['mask_uid']];r=dict(r)
            centroid=np.array([source['centroid_x'],source['centroid_y']],float)*np.array(camera['native_size'])/np.array([source['annotation_width'],source['annotation_height']])
            r['position']=undistort(centroid[None],k,dist)[0].tolist();r['complete']=source['geometry_complete'];r['name']=source['entity_name_open'];tracks[r['entity_track_id']].append(r)
        for rs in tracks.values():rs.sort(key=lambda r:r['timestamp_us'])
        hands={track:rs for track,rs in tracks.items() if any('hand' in str(r['name']).lower() for r in rs)}
        for r in [r for r in old if r['video_id']==video]:
            ep=episodes[r['episode_id']];records=[x for x in tracks[r['entity_track_id']] if ep['start_us']<=x['timestamp_us']<=ep['end_us']]
            audit.append(dict(r,**candidate_audit(records,r['input_cutoff_us'],r['input_cutoff_us']+r['horizon_us'],C['observation_tolerance_us'])))
        for s in [s for s in data.samples if s['video_id']==video]:
            i=s['row_index'];ep=episodes[s['episode_id']]
            for j,t in enumerate(s['input_times_us']):
                past=[r for r in tracks[s['entity_track_id']] if ep['start_us']<=r['timestamp_us']<=t and r['fit16_valid']]
                if past:features[i,j,4]=past[-1]['quality']
                candidates=[]
                for track,rs in hands.items():
                    if track==s['entity_track_id']:continue
                    recent=[r for r in rs if max(ep['start_us'],t-200000)<=r['timestamp_us']<=t and r['fit16_valid'] and 'hand' in str(r['name']).lower()]
                    if not recent or t-recent[-1]['timestamp_us']>100000:continue
                    last=recent[-1];position=np.array(last['position']);prior=[r for r in recent[:-1] if r['timestamp_us']<last['timestamp_us'] and r['epoch16']==last['epoch16']]
                    velocity=(position-prior[-1]['position'])/((last['timestamp_us']-prior[-1]['timestamp_us'])/1e6) if prior else np.zeros(2)
                    candidates.append((float(np.linalg.norm(position-raw[i,j,:2])),track,position,velocity,bool(prior),last['timestamp_us']))
                if candidates:
                    _,track,position,velocity,vel_valid,latest=min(candidates,key=lambda r:(r[0],r[1]))
                    object_v=(raw[i,j,:2]-raw[i,j-1,:2])/.25 if j else np.zeros(2)
                    features[i,j,5:7]=position-raw[i,j,:2];features[i,j,7:9]=velocity-object_v if vel_valid and j else 0
                    features[i,j,9:]=[1,int(vel_valid and j>0)]
                    relations.append(dict(sequence_id=s['sequence_id'],step=j,cutoff_us=t,latest_hand_source_us=latest,hand_track_id=track,available=True,physical_contact=False,identity_verified=False))
                else:relations.append(dict(sequence_id=s['sequence_id'],step=j,cutoff_us=t,latest_hand_source_us=None,hand_track_id=None,available=False,physical_contact=False,identity_verified=False))
    np.save(D/'information.npy',features,allow_pickle=False);save(R/'observation_candidates.parquet',audit);save(R/'relation_ledger.parquet',relations)
    counts=[]
    for split in ['train','validation']:
        for h in [250000,500000,1000000]:
            rs=[r for r in audit if r['split']==split and r['horizon_us']==h]
            counts.append(dict(split=split,horizon_us=h,proposed=len(rs),nearest_usable=sum(r['nearest_usable'] for r in rs),any_usable=sum(r['any_usable'] for r in rs),
                rejected_with_nearest=sum(not r['target_supported'] and r['nearest_usable'] for r in rs),rejected_with_any=sum(not r['target_supported'] and r['any_usable'] for r in rs)))
    atomic_json(R/'candidate_audit.json',dict(counts=counts,primary_target_policy_changed=False,independent_references=0,positions='intrinsic-normalized source centroids',disagreement_measured_among='complete candidates inside tolerance'))
