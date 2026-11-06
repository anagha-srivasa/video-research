"""Explicit noun mapping, same-cutoff source slopes and auditable hand assignments."""
import collections,csv,json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json
from track1_retrain.common import save
from track1.camera import undistort
from track2_feasibility.data import Data,BASE
from .common import C,D,R,OLD

NOUNS={int(r['id']):r['key'] for r in csv.DictReader((ROOT/'dataset/actions/EPIC_100_noun_classes_v2.csv').open())}

def category(class_id):
    key=NOUNS.get(class_id)
    if key in C['hand_noun_keys']:return 'hand'
    if key in C['glove_noun_keys']:return 'glove'
    if key is None:return 'unknown'
    return 'object'

def source_slope(records,cutoff):
    rows=[r for r in records if cutoff-C['slope_window_us']<=r['timestamp_us']<=cutoff and r['fit16_valid']]
    if not rows:return None
    last=max(rows,key=lambda r:(r['timestamp_us'],r['mask_uid']));rows=[r for r in rows if r['epoch16']==last['epoch16']]
    unique=collections.defaultdict(list)
    for r in rows:unique[r['timestamp_us']].append(r['position'])
    times=np.array(sorted(unique));positions=np.array([np.median(unique[t],axis=0) for t in times]);age=cutoff-int(times[-1]);span=int(times[-1]-times[0])
    valid=len(times)>=C['minimum_slope_observations'] and span>=C['minimum_slope_span_us'] and age<=C['maximum_observation_age_us']
    t=(times-times[-1])/1e6;velocity=np.zeros(2);two=np.zeros(2);rmse=0.
    if len(times)>1:
        two=(positions[-1]-positions[-2])/((times[-1]-times[-2])/1e6)
    if valid:
        design=np.c_[np.ones(len(t)),t];coef=np.linalg.lstsq(design,positions,rcond=None)[0];velocity=coef[1];rmse=float(np.sqrt(np.mean((design@coef-positions)**2)))
    return dict(position=positions[-1],velocity=velocity,two_point_velocity=two,velocity_valid=valid,age_us=age,count=len(times),span_us=span,rmse=rmse,quality=float(last['quality']),latest_us=int(times[-1]),class_id=last['class_id'],name=last['name'])

def relation_vector(object_state,hand):
    r=hand['position']-object_state['position'];valid=bool(hand['velocity_valid'] and object_state['velocity_valid']);u=hand['velocity']-object_state['velocity'] if valid else np.zeros(2);distance=float(np.linalg.norm(r))
    return np.r_[r,u,distance,float(r@u/(distance+1e-8)) if valid else 0.,float(np.linalg.norm(u)),1.,int(valid),hand['age_us']/1e6,hand['count'],hand['rmse']]


def prepare():
    data=Data();features=np.zeros((len(data.samples),4,21));pair=features.copy();cohorts=np.full(len(data.samples),'unknown',dtype='<U8');ledger=[];names=collections.Counter();pair_effective=np.zeros(len(data.samples),bool)
    episodes={r['episode_id']:r for r in pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist()}
    for video in sorted({s['video_id'] for s in data.samples}):
        spatial=pq.read_table(BASE/'spatial'/f'{video}.parquet').to_pylist();wanted={r['mask_uid'] for r in spatial};source={}
        for batch in pq.ParquetFile(ROOT/f'canonical/v2/masks.{video}.parquet').iter_batches(columns=['mask_uid','class_id','entity_name_open'],batch_size=2048):
            for r in batch.to_pylist():
                if r['mask_uid'] in wanted:source[r['mask_uid']]=r
        camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{video}.camera.json').read_text());k=np.array(camera['intrinsic_native']);dist=np.array(camera['distortion']);tracks=collections.defaultdict(list)
        for r in spatial:
            meta=source[r['mask_uid']];row=dict(r,class_id=meta['class_id'],name=meta['entity_name_open']);row['position']=undistort(np.array(row['centre_native'])[None],k,dist)[0];tracks[row['entity_track_id']].append(row)
            if 'hand' in str(row['name']).lower() or category(row['class_id']) in ('hand','glove'):names[(row['class_id'],str(row['name']),category(row['class_id']))]+=1
        for values in tracks.values():values.sort(key=lambda r:r['timestamp_us'])
        for s in [s for s in data.samples if s['video_id']==video]:
            i=s['row_index'];ep=episodes[s['episode_id']];hand_states={};own=[]
            for j,t in enumerate(s['input_times_us']):
                state=source_slope([r for r in tracks[s['entity_track_id']] if ep['start_us']<=r['timestamp_us'] and r['epoch16']==s['epoch']],t);own.append(state)
                if state is None:continue
                features[i,j,:8]=np.r_[state['position'],state['velocity'],state['age_us']/1e6,state['count'],state['quality'],state['velocity_valid']];features[i,j,20]=state['rmse']
                candidates=[]
                for track,rows in tracks.items():
                    if track==s['entity_track_id']:continue
                    recent=[r for r in rows if ep['start_us']<=r['timestamp_us']<=t and category(r['class_id']) in ('hand','glove')]
                    hs=source_slope(recent,t)
                    if hs is None or hs['age_us']>C['maximum_observation_age_us']:continue
                    hand_states[track,j]=hs;distance=float(np.linalg.norm(hs['position']-state['position']));candidates.append((distance,track,hs))
                if candidates:
                    _,track,hs=min(candidates,key=lambda r:(r[0],r[1]));features[i,j,8:20]=relation_vector(state,hs)
                    ledger.append(dict(sequence_id=s['sequence_id'],step=j,cutoff_us=t,object_latest_us=state['latest_us'],hand_latest_us=hs['latest_us'],object_class_id=state['class_id'],hand_class_id=hs['class_id'],hand_name=hs['name'],hand_track_id=track,
                        object_age_us=state['age_us'],hand_age_us=hs['age_us'],object_span_us=state['span_us'],hand_span_us=hs['span_us'],object_count=state['count'],hand_count=hs['count'],relative_velocity_valid=bool(state['velocity_valid'] and hs['velocity_valid']),
                        object_velocity=state['velocity'].tolist(),hand_velocity=hs['velocity'].tolist(),object_two_point_velocity=state['two_point_velocity'].tolist(),hand_two_point_velocity=hs['two_point_velocity'].tolist(),physical_association_verified=False))
            if own[-1] is not None:
                cat=category(own[-1]['class_id']);cohorts[i]='hand' if cat in ('hand','glove') else cat
                features[i,:,:2]-=own[-1]['position']
            pair[i]=features[i]
            # Assign a different observed hand ID, with the same four-hop availability/velocity support.
            selected={r['hand_track_id'] for r in ledger if r['sequence_id']==s['sequence_id']}
            alternatives=[]
            for track in sorted({track for track,j in hand_states}-selected):
                candidate=features[i].copy();compatible=True
                for j in range(4):
                    hs=hand_states.get((track,j));present=features[i,j,15]>0
                    if bool(hs is not None)!=present:compatible=False;break
                    if present:
                        if own[j] is None or bool(hs['velocity_valid'] and own[j]['velocity_valid'])!=bool(features[i,j,16]) or abs(hs['age_us']/1e6-features[i,j,17])>.05:compatible=False;break
                        candidate[j,8:20]=relation_vector(own[j],hs)
                if compatible:alternatives.append(candidate)
            if alternatives:pair[i]=alternatives[i%len(alternatives)];pair_effective[i]=True
    np.save(D/'features.npy',features,allow_pickle=False);np.save(D/'paired_features.npy',pair,allow_pickle=False);np.save(D/'cohorts.npy',cohorts,allow_pickle=False);np.save(D/'pair_effective.npy',pair_effective,allow_pickle=False);save(R/'selected_pairs.parquet',ledger)
    atomic_json(R/'hand_policy.json',dict(noun_mapping='dataset/actions/EPIC_100_noun_classes_v2.csv',accepted_hand_keys=C['hand_noun_keys'],accepted_glove_keys=C['glove_noun_keys'],generic_glove_policy=C['generic_glove_policy'],
        candidate_names=[dict(class_id=c,name=n,policy_category=cat,source_observations=count) for (c,n,cat),count in sorted(names.items(),key=str)],
        selected_pairs=len(ledger),class_policy='Noun class is authoritative; substring never establishes a hand.',provisional_identity=True))
    eligible=np.load(OLD/'input_mask.npy')
    counts=[dict(split=split,cohort=cohort,histories=int(np.sum((data.split==split)&(cohorts==cohort))),input_eligible_histories=int(np.sum((data.split==split)&(cohorts==cohort)&eligible)),pair_shuffle_effective=int(np.sum((data.split==split)&(cohorts==cohort)&pair_effective)),eligible_pair_shuffle_effective=int(np.sum((data.split==split)&(cohorts==cohort)&pair_effective&eligible))) for split in ['train','validation'] for cohort in ['object','hand','unknown']]
    atomic_json(R/'feature_support.json',dict(cohorts=counts,feature_channels=21,raw_query_support_reused=True,frame_availability='Retrospective annotations; timestamps past cutoff do not prove online availability.',pair_shuffle='Alternate observed hand with matched four-hop presence, velocity support and age within 50ms; unchanged when unavailable; not a certified physical association.'))
