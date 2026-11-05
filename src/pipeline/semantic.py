from __future__ import annotations
import ast
from bisect import bisect_left, bisect_right
import collections
import csv
import json
import numpy as np
import duckdb
from scipy.spatial.transform import Rotation
from .common import ROOT,LOG,TableWriter,read,write_rows,uid,dumps,seconds_us,stage_manifest
from .video import load_timeline
from .visor import number

def normalize_actions(cfg,choice):
    v=choice['video_id'];raw=ROOT/cfg['raw'];out=ROOT/cfg['output'];duration=read(out,'videos',v)[0]['duration_us']
    rows=[];inputs=[]
    for split in ['train','validation']:
        path=raw/f'actions/EPIC_100_{split}.csv';inputs.append(path)
        for r in csv.DictReader(path.open()):
            if r['video_id']!=v:continue
            start=seconds_us(r['start_timestamp']);stop=seconds_us(r['stop_timestamp'])
            flags=[]
            if not 0<=start<=stop<=duration+500000:flags.append('invalid_or_out_of_video_interval')
            rows.append(dict(video_id=v,action_uid=uid(v,'action',r['narration_id']),narration_id=r['narration_id'],participant_id=r['participant_id'],
                action_split=split,visor_split=choice['visor_split'],start_us=start,stop_us=stop,narration_timestamp_us=seconds_us(r['narration_timestamp']),
                start_frame_source=int(r['start_frame']),stop_frame_source=int(r['stop_frame']),narration=r['narration'],verb=r['verb'],verb_class=int(r['verb_class']),
                noun=r['noun'],noun_class=int(r['noun_class']),all_nouns=ast.literal_eval(r['all_nouns']),all_noun_classes=ast.literal_eval(r['all_noun_classes']),
                source_file=str(path.relative_to(ROOT)),quality_flags=flags))
    if not rows or any(r['action_split']!=choice['action_split'] for r in rows):raise ValueError('Action split mismatch '+v)
    p=write_rows(out,'actions',v,rows);stage_manifest(cfg,v,'normalize_actions',inputs,[p])

def normalize_camera(cfg,choice):
    v=choice['video_id'];raw=ROOT/cfg['raw'];out=ROOT/cfg['output'];path=raw/f'camera/{v}.json'
    source=json.loads(path.read_text());cam=source['camera'];native=read(out,'videos',v)[0];tl=load_timeline(cfg,v)
    sparse=read(out,'annotation_frames',v)
    anchors={a['epic_mapped_frame_index']:a['timestamp_us'] for a in sparse
             if a['source_kind']=='manual_sparse' and a['alignment_status']=='aligned_official_sparse' and a['epic_mapped_frame_index'] is not None}
    indices=sorted(anchors);times=[anchors[i] for i in indices]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Nonmonotonic EPIC camera timestamp anchors '+v)
    sw=cam['width']/native['native_width'];sh=cam['height']/native['native_height']
    candidate=[sw,0.,0.,0.,sh,0.,0.,0.,1.]
    verified=bool(cfg.get('camera_resize_verified',False))
    with TableWriter(out,'camera_frames',v) as w:
        for name,pose in sorted(source['images'].items()):
            index=number(name);flags=[];target=None;status='unresolved'
            if index in anchors:target=anchors[index];status='official_sparse_bridge'
            else:
                j=bisect_left(indices,index)
                if 0<j<len(indices):
                    # Interpolate correspondence clock, never the camera pose.
                    a,b=indices[j-1],indices[j]
                    target=round(anchors[a]+(index-a)*(anchors[b]-anchors[a])/(b-a))
                    status='camera_time_inferred_between_sparse_anchors';flags.append(status)
            frame,error,limit=tl.nearest(target)
            aligned=frame is not None and error<=limit
            values=np.asarray(pose,dtype=np.float64);valid=values.shape==(7,) and np.isfinite(values).all() and np.linalg.norm(values[:4])>1e-12
            tcw=twc=None
            if valid:
                qw,qx,qy,qz,tx,ty,tz=values
                rot=Rotation.from_quat([qx,qy,qz,qw]).as_matrix();t=np.array([tx,ty,tz])
                a=np.eye(4);a[:3,:3]=rot;a[:3,3]=t
                b=np.eye(4);b[:3,:3]=rot.T;b[:3,3]=-rot.T@t
                tcw=a.ravel().tolist();twc=b.ravel().tolist()
                if abs(np.linalg.norm(values[:4])-1)>1e-4:flags.append('quaternion_normalized_for_matrix')
            else:flags.append('invalid_source_pose')
            if not aligned:flags.append('camera_time_alignment_unresolved')
            if not verified:flags.append('camera_resize_and_crop_not_verified')
            w.add(dict(video_id=v,camera_uid=uid(v,'camera',name),source_frame_name=name,
                native_frame_uid=frame['native_frame_uid'] if aligned else None,timestamp_us=frame['timestamp_us'] if aligned else None,
                source_timestamp_estimate_us=target,alignment_error_us=error,alignment_limit_us=limit,
                alignment_status=status if aligned else 'unresolved',pose_valid=bool(valid and aligned),camera_model=cam['model'],
                camera_width=cam['width'],camera_height=cam['height'],camera_params=cam['params'],T_world_to_camera=tcw,T_camera_to_world=twc,
                T_native_to_camera=candidate,transform_valid=verified,transform_reason='configured_verified_pure_resize' if verified else 'candidate_scale_only_crop_and_field_of_view_unverified',
                geometry_scale='reconstruction_units',source_file=str(path.relative_to(ROOT)),quality_flags=flags))
    stage_manifest(cfg,v,'normalize_camera',[path,out/f'annotation_frames.{v}.parquet'],[w.path],dict(pose_convention='world_to_camera',clock='sparse-anchor correspondence interpolation; no frame/fps clock',resize_verified=verified))

def build_events(cfg,choice):
    v=choice['video_id'];out=ROOT/cfg['output'];events=[]
    def add(t,kind,source,previous=None,confidence='source_annotation',flags=None):
        if t is not None:events.append(dict(video_id=v,evidence_uid=uid(v,kind,source,t),timestamp_us=t,evidence_type=kind,source_uid=source,
            previous_observed_us=previous,confidence_class=confidence,quality_flags=flags or []))
    for action in read(out,'actions',v):
        add(action['start_us'],'action_start',action['action_uid']);add(action['stop_us'],'action_end',action['action_uid'])
    frames=read(out,'annotation_frames',v);manual=sorted([r for r in frames if r['source_kind']=='manual_sparse' and r['timestamp_us'] is not None],key=lambda r:(r['timestamp_us'],r['annotation_frame_uid']))
    previous_seq=None
    for frame in manual:
        if frame['subsequence']!=previous_seq:
            add(frame['timestamp_us'],'subsequence_boundary',frame['annotation_frame_uid']);previous_seq=frame['subsequence']
    contacts=sorted([r for r in read(out,'contacts',v) if r['timestamp_us'] is not None and r['source_kind']=='manual_sparse'],key=lambda r:(r['timestamp_us'],r['contact_uid']))
    mask_path=out/f'masks.{v}.parquet'
    with duckdb.connect() as db:
        db.from_parquet(str(mask_path)).create_view('masks')
        hand_entities=dict(db.execute('SELECT mask_uid, entity_uid FROM masks WHERE mask_uid IN (SELECT unnest(?))',[[c['hand_mask_uid'] for c in contacts]]).fetchall()) if contacts else {}
        presence=db.execute("SELECT entity_uid,min(timestamp_us),max(timestamp_us) FROM masks WHERE timestamp_us IS NOT NULL AND source_kind='manual_sparse' GROUP BY entity_uid ORDER BY entity_uid").fetchall()
    for e,start,end in presence:
        add(start,'entity_first_observed',e,confidence='observation_extent',flags=['absence_not_inferred'])
        add(end,'entity_last_observed',e,confidence='observation_extent',flags=['absence_not_inferred'])
    prev={}
    for c in contacts:
        e=hand_entities.get(c['hand_mask_uid'])
        if e is None:continue
        state=(c['contact_status'],c['target_entity_uid']);old=prev.get(e)
        if old and old[0]!=state:
            if state[0] in ['resolved_target','unsegmented_contact_target'] and old[0][0]=='hand_not_in_contact':kind='contact_started_observed'
            elif state[0]=='hand_not_in_contact' and old[0][0] in ['resolved_target','unsegmented_contact_target']:kind='contact_ended_observed'
            elif state[0]=='resolved_target' and old[0][0]=='resolved_target':kind='contact_target_changed_observed'
            else:kind='contact_status_changed_observed'
            add(c['timestamp_us'],kind,c['contact_uid'],old[1],confidence='interval_censored_observation_change',flags=['transition_time_unknown_between_observations','provisional_identity_scope'])
        prev[e]=(state,c['timestamp_us'])
    events.sort(key=lambda r:(r['timestamp_us'],r['evidence_type'],r['evidence_uid']))
    path=write_rows(out,'event_evidence',v,events)
    stage_manifest(cfg,v,'build_event_evidence',[out/f'{n}.{v}.parquet' for n in ['actions','annotation_frames','masks','contacts']],[path],dict(not_ground_truth_events=True))

def build_windows(cfg,choice):
    v=choice['video_id'];out=ROOT/cfg['output'];tl=load_timeline(cfg,v);duration=read(out,'videos',v)[0]['duration_us']
    frames=tl.ordered;times=tl.times
    ann=sorted([a for a in read(out,'annotation_frames',v) if a['timestamp_us'] is not None],key=lambda r:(r['timestamp_us'],r['annotation_frame_uid']))
    ats=[a['timestamp_us'] for a in ann];actions=read(out,'actions',v)
    cameras={r['native_frame_uid'] for r in read(out,'camera_frames',v) if r['pose_valid']}
    with duckdb.connect() as db:
        counts={a:(m,i) for a,m,i in db.execute("SELECT annotation_frame_uid,count(*) FILTER(WHERE source_kind='manual_sparse'),count(*) FILTER(WHERE source_kind='dense_interpolated') FROM read_parquet(?) GROUP BY annotation_frame_uid",[str(out/f'masks.{v}.parquet')]).fetchall()}
        contacts=dict(db.execute('SELECT annotation_frame_uid,count(*) FROM read_parquet(?) GROUP BY annotation_frame_uid',[str(out/f'contacts.{v}.parquet')]).fetchall())
    with TableWriter(out,'windows_1s_hop250ms',v) as w:
        for start in range(0,duration,cfg['hop_us']):
            end=start+cfg['window_us'];lo=bisect_left(times,start);hi=bisect_left(times,end)
            a=bisect_left(ats,start);b=bisect_left(ats,end);selection=ann[a:b]
            masks=sum(counts.get(x['annotation_frame_uid'],(0,0))[0] for x in selection)
            dense=sum(counts.get(x['annotation_frame_uid'],(0,0))[1] for x in selection)
            covered=0;last_end=start
            for f in frames[max(0,lo-1):hi]:
                if f['duration_us'] and f['duration_us']>0:
                    begin=max(start,f['timestamp_us'],last_end);finish=min(end,f['timestamp_us']+f['duration_us'])
                    covered+=max(0,finish-begin);last_end=max(last_end,finish)
            flags=[]
            if end>duration:flags.append('window_extends_beyond_video')
            if not selection:flags.append('no_aligned_mask_annotation')
            w.add(dict(video_id=v,window_uid=uid(v,'physical_window',start,end),start_us=start,end_us=end,
                native_frame_begin=frames[lo]['decode_index'] if lo<hi else None,native_frame_end=frames[hi-1]['decode_index']+1 if lo<hi else None,
                annotation_frame_uids=[x['annotation_frame_uid'] for x in selection],action_uids=[r['action_uid'] for r in actions if r['start_us']<end and r['stop_us']>start],
                camera_pose_coverage=sum(f['native_frame_uid'] in cameras for f in frames[lo:hi])/(hi-lo) if hi>lo else 0.,
                manual_mask_count=masks,interpolated_mask_count=dense,contact_count=sum(contacts.get(x['annotation_frame_uid'],0) for x in selection),
                coverage_fraction=min(1.,covered/cfg['window_us']),quality_flags=flags))
    stage_manifest(cfg,v,'build_window_index',[out/f'{n}.{v}.parquet' for n in ['videos','frames','annotation_frames','actions','masks','contacts','camera_frames']],[w.path])
