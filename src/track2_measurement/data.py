import collections,hashlib,json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json
from track1_retrain.common import save
from track2_feasibility.data import Data,BASE,inventory,stable_id
from track1.camera import undistort

C=json.loads((ROOT/'configs/track2_measurement.json').read_text())
D=ROOT/'derived/track2_measurement';R=ROOT/'reports/track2_measurement';W=ROOT/'work/track2_measurement'


def evidence_allowed(row,cutoff):
    # Timestamp alone cannot certify retrospective annotation availability.
    latest=row.get('latest_evidence_us');available=row.get('available_at_us')
    return bool(row.get('online_observation_allowed') and latest is not None and available is not None
                and latest<=cutoff and available<=cutoff and row['timestamp_us']<=cutoff)


def prepare():
    data=Data();functional,indices=inventory(BASE)
    origins={(r['video_id'],r['entity_track_id'],r['window_end_us']):r for r in functional}
    seqs={stable_id(s):s for s in indices};samples={s['sequence_id']:i for i,s in enumerate(data.samples)}
    episodes={r['episode_id']:r for r in pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist()}
    ledger=pq.read_table(ROOT/'reports/track2_feasibility/target_ledger.parquet').to_pylist()
    raw=np.zeros((len(data.samples),4,4),float);input_ok=np.zeros((len(data.samples),4),bool)
    observations=[];ancestry=[];reviews=[];private=[]
    for video in sorted({s['video_id'] for s in data.samples}):
        spatial=pq.read_table(BASE/'spatial'/f'{video}.parquet').to_pylist();tracks=collections.defaultdict(list)
        camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{video}.camera.json').read_text());k=np.array(camera['intrinsic_native']);dist=np.array(camera['distortion'])
        wanted={r['mask_uid'] for r in spatial};source={};group_latest={}
        columns=['mask_uid','timestamp_us','source_kind','available_at_us','online_observation_allowed','dependency_group','geometry_complete','centroid_x','centroid_y','annotation_width','annotation_height']
        for batch in pq.ParquetFile(ROOT/f'canonical/v2/masks.{video}.parquet').iter_batches(columns=columns,batch_size=2048):
            for r in batch.to_pylist():
                if r['timestamp_us'] is not None:group_latest[r['dependency_group']]=max(r['timestamp_us'],group_latest.get(r['dependency_group'],-1))
                if r['mask_uid'] in wanted:source[r['mask_uid']]=r
        for row in spatial:tracks[row['entity_track_id']].append(row)
        for values in tracks.values():values.sort(key=lambda r:r['timestamp_us'])
        for s in data.samples:
            if s['video_id']!=video:continue
            i=s['row_index'];ep=episodes[s['episode_id']]
            for j,t in enumerate(s['input_times_us']):
                rs=[r for r in tracks[s['entity_track_id']] if max(ep['start_us'],t-1000000)<=r['timestamp_us']<=t and r['fit16_valid'] and r['epoch16']==s['epoch']]
                if not rs:continue
                last=rs[-1];age=(t-last['timestamp_us'])/1e6;count=len({r['timestamp_us'] for r in rs if r['timestamp_us']>=t-200000})
                raw[i,j,:2]=undistort(np.array(last['centre_native'])[None],k,dist)[0];raw[i,j,2:]=[age,count]
                input_ok[i,j]=age<=.1+1e-9 and count>=3
                # Include every retained source mask in the functional window, not merely the last centroid.
                state=origins[video,s['entity_track_id'],t]
                deps=[source[uid] for uid in state['retained_mask_uids']]
                ancestry.append(dict(sequence_id=s['sequence_id'],step=j,video_id=video,split=s['split'],cutoff_us=t,
                    source_masks=len(deps),latest_evidence_us=None,available_at_us=None,
                    ancestry_established=False,online_eligible=all(evidence_allowed(x,t) for x in deps) if deps else False,
                    dependency_group_contains_later_labels=any(group_latest.get(x['dependency_group'],-1)>t for x in deps),
                    reason='source_annotation_availability_and_full_ancestry_unknown',group_span_is_not_proof_of_dependency=True))
        for item in [r for r in ledger if r['video_id']==video]:
            s=seqs[item['sequence_id']];origin=origins[video,s['entity_track_id'],s['end_times_us'][-1]];ep=episodes[origin['episode_id']]
            t=item['input_cutoff_us'];end=t+item['horizon_us'];rs=tracks[s['entity_track_id']]
            future=[r for r in rs if t<r['timestamp_us']<=ep['end_us']]
            target=min(future,key=lambda r:abs(r['timestamp_us']-end)) if future else None
            timing=target is not None and abs(target['timestamp_us']-end)<=C['observation_target_tolerance_us']
            sr=source[target['mask_uid']] if timing else None
            usable=bool(timing and sr['geometry_complete'] and np.isfinite([sr['centroid_x'],sr['centroid_y']]).all())
            past=[r for r in rs if ep['start_us']<=r['timestamp_us']<=t];latest=past[-1] if past else None
            changes=bool(latest and any(r['source_signature']!=latest['source_signature'] for r in future if r['timestamp_us']<=end))
            observations.append(dict(item,entity_track_id=s['entity_track_id'],observation_target_available=usable,
                observation_mask_uid=target['mask_uid'] if usable else None,actual_target_us=target['timestamp_us'] if usable else None,
                observation_source_kind=sr['source_kind'] if usable else None,
                observation_position=undistort((np.array([sr['centroid_x'],sr['centroid_y']])*np.array(camera['native_size'])/np.array([sr['annotation_width'],sr['annotation_height']]))[None],k,dist)[0].tolist() if usable else None,
                source_topology_transition=changes,independent_position_available=False,physical_reference_position=None,
                identity_verified=False,observation_covariance_calibrated=False,
                input_eligible=item['input_supported'] and item['sequence_id'] in samples and bool(input_ok[samples[item['sequence_id']]].all())))
    np.save(D/'recent_observation.npy',raw,allow_pickle=False);np.save(D/'input_mask.npy',input_ok.all(1),allow_pickle=False)
    save(R/'target_audit.parquet',observations);save(R/'ancestry.parquet',ancestry)
    counts=[]
    for split in ('train','validation'):
        for h in C['horizons_us']:
            rs=[r for r in observations if r['split']==split and r['horizon_us']==h]
            counts.append(dict(split=split,horizon_us=h,proposed=len(rs),source_state_supported=sum(r['target_supported'] for r in rs),
                exact_rejection_reasons=dict(collections.Counter(r['reason'] for r in rs if not r['target_supported'])),
                observation_available=sum(r['observation_target_available'] for r in rs),
                rejected_state_with_observation=sum(not r['target_supported'] and r['observation_target_available'] for r in rs),
                transition_strata=[dict(source_topology_transition=change,proposed=sum(r['source_topology_transition']==change for r in rs),
                    rejected=sum(r['source_topology_transition']==change and not r['target_supported'] for r in rs)) for change in (False,True)]))
            if split=='validation':
                for retained in (False,True):
                    chosen=sorted([r for r in rs if r['target_supported']==retained],key=lambda r:hashlib.sha256((r['sequence_id']+str(h)).encode()).hexdigest())[:8]
                    for r in chosen:
                        uid=hashlib.sha256((r['sequence_id']+str(h)).encode()).hexdigest()[:20]
                        reviews.append(dict(task_id=uid,video_id=r['video_id'],entity_track_id=r['entity_track_id'],start_us=r['input_cutoff_us']-500000,end_us=r['input_cutoff_us']+h,
                            instructions='Review visible motion/occlusion and source failure without predictions. Unknown is allowed. This is not a reference label.'))
                        private.append(dict(task_id=uid,retained=retained,reason=r['reason']))
    atomic_json(R/'support.json',dict(records=counts,input_histories=len(data.samples),new_input_supported=int(input_ok.all(1).sum()),
        common_all_horizon_validation=int((data.mask.all(1)&(data.split=='validation')&input_ok.all(1)).sum()),
        online_eligible_histories=0 if not any(r['online_eligible'] for r in ancestry) else None,
        ancestry_unknown=True,independent_reference_count=0,topology_is_not_physical_event_label=True))
    atomic_json(R/'blinded_review_queue.json',dict(tasks=reviews,reviewed=False,contains_candidate_predictions=False))
    atomic_json(W/'review_sampling_key.json',private)
