from __future__ import annotations
import collections
import json
from pathlib import Path
import duckdb
import numpy as np
from PIL import Image,ImageDraw
import cv2
from .common import ROOT,LOG,read,write_rows,uid,dumps,atomic_json,stage_manifest,code_hash,config_hash
from .store import Dataset

def audit_and_render(cfg,choice,render=True,root=None):
    v=choice['video_id'];out=ROOT/cfg['output'];flags=[];gates={}
    def flag(record,stage,code,severity='WARN',details=''):
        flags.append(dict(video_id=v,record_uid=record,stage=stage,severity=severity,code=code,details=details))
    with Dataset(out,require_ready=False,root=root) as ds:
        rows=lambda sql,p=None:ds.rows(sql,p or [v])
        video=rows('SELECT * FROM videos WHERE video_id=?')[0]
        frames=rows('SELECT * FROM frames WHERE video_id=? ORDER BY decode_index')
        invalid_pts=sum(f['timestamp_us'] is None or 'non_increasing_presentation_time' in f['quality_flags'] for f in frames)
        if invalid_pts:flag(v,'timeline','pts_exceptions',details=str(invalid_pts))
        gates['timeline_exceptions_explicit']=all(f['quality_flags'] for f in frames if f['timestamp_us'] is None)
        summary=rows("SELECT count(*) n,count(*) FILTER(WHERE geometry_status='quarantined') quarantined,count(*) FILTER(WHERE geometry_status='repaired_for_rasterisation') repaired,count(*) FILTER(WHERE source_kind='manual_sparse') manual,count(*) FILTER(WHERE source_kind='dense_interpolated') interpolated,count(*) FILTER(WHERE source_kind='unknown_dense_type') unknown_provenance FROM masks WHERE video_id=?")[0]
        if summary['quarantined']:flag(v,'geometry','quarantined_masks_retained',details=str(summary['quarantined']))
        if summary['repaired']:flag(v,'geometry','derived_raster_repairs_recorded',details=str(summary['repaired']))
        gates['known_mask_provenance']=summary['unknown_provenance']==0
        ann=rows('SELECT * FROM annotation_frames WHERE video_id=? ORDER BY source_file,source_record_offset')
        unresolved=sum(a['alignment_status']=='unresolved' for a in ann)
        if unresolved:flag(v,'alignment','unresolved_annotation_times_retained',details=str(unresolved))
        gates['annotation_correspondence']=all(a['native_frame_uid'] is not None or a['alignment_status']=='unresolved' for a in ann)
        gates['annotation_dimensions']=all(a['annotation_width']>0 and a['annotation_height']>0 for a in ann)
        orphans=rows('SELECT count(*) n FROM masks m LEFT JOIN annotation_frames a USING(annotation_frame_uid) WHERE m.video_id=? AND a.annotation_frame_uid IS NULL')[0]['n']
        gates['no_orphan_masks']=orphans==0
        orphans=rows("SELECT count(*) n FROM contacts c LEFT JOIN masks m ON c.target_mask_uid=m.mask_uid WHERE c.video_id=? AND c.contact_status='resolved_target' AND m.mask_uid IS NULL")[0]['n']
        gates['contact_target_references']=orphans==0
        source_contacts=rows('SELECT contact_status,count(*) n FROM contacts WHERE video_id=? GROUP BY contact_status ORDER BY contact_status')
        expected_contacts=0;expected_masks=0
        for path in [ROOT/cfg['raw']/f'visor/{v}.json',ROOT/cfg['staging']/f'visor_corrected/{v}_interpolations.json']:
            source=json.loads(path.read_text())
            for a in source['video_annotations']:
                expected_masks+=len(a['annotations'])
                expected_contacts+=sum('in_contact_object' in obj for obj in a['annotations'])
            del source
        gates['every_source_mask_retained']=summary['n']==expected_masks
        gates['every_source_contact_retained']=sum(r['n'] for r in source_contacts)==expected_contacts
        matrices=rows('SELECT T_world_to_camera a,T_camera_to_world b FROM camera_frames WHERE video_id=? AND pose_valid')
        gates['camera_pose_inverse']=all(np.allclose(np.array(r['a']).reshape(4,4)@np.array(r['b']).reshape(4,4),np.eye(4),atol=1e-8) for r in matrices)
        missing_pose=rows('SELECT count(*) n FROM camera_frames WHERE video_id=? AND NOT pose_valid')[0]['n']
        if missing_pose:flag(v,'camera','source_camera_pose_unaligned_or_invalid',details=str(missing_pose))
        if not cfg.get('camera_resize_verified'):flag(v,'camera','resize_crop_transform_unverified',details='Candidate transform retained; geometry must not be used as validated calibration.')
        gates['split_preservation']=video['action_split']==choice['action_split'] and video['visor_split']==choice['visor_split']
        duplicate_uids={}
        for table,key in [('frames','native_frame_uid'),('annotation_frames','annotation_frame_uid'),('masks','mask_uid'),('contacts','contact_uid'),('actions','action_uid')]:
            n=rows(f'SELECT count(*)-count(DISTINCT {key}) n FROM {table} WHERE video_id=?')[0]['n']
            duplicate_uids[table]=n
        gates['unique_observation_ids']=not any(duplicate_uids.values())
        # Exercise source-independent windows at beginning, middle, end and a
        # non-grid boundary. This catches joins beyond a single component test.
        ends=sorted({min(video['duration_us'],x) for x in [1000000,1225000,video['duration_us']//2,video['duration_us']] if min(video['duration_us'],x)>=1000000})
        ingestion=[]
        for end in ends:
            sample=ds.get_window(v,end_time_us=end,duration_us=1000000)
            ok=all(sample.start_us<=int(t)<sample.end_us for t in sample.frame_timestamps_us)
            ok &= all(a['start_us']<sample.end_us and a['stop_us']>sample.start_us for a in sample.action_intervals)
            ok &= all(m['source_kind'] in ['manual_sparse','dense_gt_endpoint','dense_interpolated'] for m in sample.mask_observations)
            ingestion.append(ok)
        gates['window_query_contract']=bool(ingestion) and all(ingestion)
        sample_times=[frames[i]['timestamp_us'] for i in sorted({0,len(frames)//2,len(frames)-1}) if frames[i]['timestamp_us'] is not None]
        for t in sample_times:
            rgb=ds.get_frame(v,t)
            if rgb.shape[:2]!=(video['native_height'],video['native_width']):raise ValueError('Lazy decode dimensions mismatch')
        gates['lazy_random_decode']=True
        output_images=[]
        if render:
            candidates=[]
            for kind in ['manual_sparse','dense']:
                subset=[a for a in ann if a['source_kind']==kind and a['timestamp_us'] is not None]
                if subset:candidates.extend([subset[0],subset[len(subset)//2]])
            by_error=sorted([a for a in ann if a['timestamp_us'] is not None],key=lambda a:(-(a['alignment_error_us'] or 0),a['annotation_frame_uid']))
            if by_error:candidates.append(by_error[0])
            contact_frames=rows("SELECT DISTINCT a.* FROM annotation_frames a JOIN contacts c USING(annotation_frame_uid) WHERE a.video_id=? AND a.timestamp_us IS NOT NULL AND c.contact_status='resolved_target' ORDER BY a.timestamp_us LIMIT 1")
            candidates.extend(contact_frames)
            chosen=[];seen=set()
            for a in candidates:
                if a['annotation_frame_uid'] not in seen:chosen.append(a);seen.add(a['annotation_frame_uid'])
            for a in by_error:
                if len(chosen)>=cfg['qa_samples_per_video']:break
                if a['annotation_frame_uid'] not in seen:chosen.append(a);seen.add(a['annotation_frame_uid'])
            chosen=chosen[:cfg['qa_samples_per_video']]
            sheet=Image.new('RGB',(1920,2*520),(20,20,20));draw=ImageDraw.Draw(sheet)
            for i,a in enumerate(chosen):
                rgb=ds.get_frame(v,a['timestamp_us']);image=Image.fromarray(rgb).resize((640,360))
                base=np.asarray(image).copy();overlay=base.copy()
                masks=ds.rows('SELECT * FROM masks WHERE annotation_frame_uid=? ORDER BY mask_uid',[a['annotation_frame_uid']])
                for j,m in enumerate(masks):
                    for ring in json.loads(m['segments_raw']):
                        poly=np.asarray(ring,dtype=np.float64)
                        if poly.ndim!=2 or poly.shape[1]!=2 or not np.isfinite(poly).all():continue
                        poly=np.rint(poly*[640/a['annotation_width'],360/a['annotation_height']]).astype(np.int32)
                        color=[(40,220,100),(255,90,90),(90,160,255),(255,220,60),(220,100,240)][j%5]
                        cv2.fillPoly(overlay,[poly],color);cv2.polylines(base,[poly],True,color,1)
                blend=cv2.addWeighted(base,.65,overlay,.35,0)
                x=i%3*640;y=i//3*520;sheet.paste(Image.fromarray(blend),(x,y))
                c=ds.rows('SELECT contact_status,target_source_mask_id FROM contacts WHERE annotation_frame_uid=?',[a['annotation_frame_uid']])
                act=ds.rows('SELECT narration FROM actions WHERE video_id=? AND start_us<=? AND stop_us>? ORDER BY start_us',[v,a['timestamp_us'],a['timestamp_us']])
                pose=ds.rows('SELECT pose_valid FROM camera_frames WHERE native_frame_uid=?',[a['native_frame_uid']])
                lines=[f"{v} | native t={a['timestamp_us']/1e6:.6f}s | decode={a['native_decode_index']}",
                    f"VISOR {a['visor_frame_name']} | EPIC {a['epic_mapped_frame_index']}",
                    f"{','.join(sorted({m['source_kind'] for m in masks}))} | {a['alignment_status']} | error_us={a['alignment_error_us']}",
                    'Masks: '+', '.join(f"{m['entity_name_open']}[{m['source_mask_id'][:7]}]" for m in masks),
                    'Contact: '+', '.join(f"{r['contact_status']}->{r['target_source_mask_id']}" for r in c),
                    'Actions: '+', '.join(r['narration'] for r in act),
                    f"Camera pose={any(r['pose_valid'] for r in pose)} | resize transform unverified",
                    f"Geometry: {sum(m['geometry_status']=='quarantined' for m in masks)} quarantined, {sum(m['geometry_status']=='repaired_for_rasterisation' for m in masks)} repaired; no metric depth."]
                for j,line in enumerate(lines):draw.text((x+5,y+366+j*18),line[:100],fill='white')
            dest=ROOT/'reports'/f'qa.{v}.jpg';sheet.save(dest,quality=90);output_images.append(str(dest.relative_to(ROOT)))
            atomic_json(ROOT/'reports'/f'qa_samples.{v}.json',chosen)
        summary.update(video_id=v,pipeline_sha256=code_hash(),config_sha256=config_hash(cfg),annotation_frames=len(ann),unresolved_annotation_frames=unresolved,decoded_frames=len(frames),
            contacts=source_contacts,gates=gates,automated_pass=all(gates.values()),visual_review='pending',qa_images=output_images,
            limitations=['Dense interior timing inferred only when endpoint native ordinals agree.','Camera resize/crop not validated; scale remains reconstruction units.','Identity scopes are hypotheses, not confirmed tracks.'])
    p=write_rows(out,'qa_flags',v,sorted(flags,key=lambda r:(r['stage'],r['code'],r['record_uid'])))
    atomic_json(ROOT/'reports'/f'qa.{v}.json',summary)
    stage_manifest(cfg,v,'run_qa',[out/f'{table}.{v}.parquet' for table in ['frames','annotation_frames','masks','contacts','entities','actions','camera_frames','windows_1s_hop250ms']],[p])
    if not all(gates.values()):raise ValueError('QA gates failed: '+str({k:x for k,x in gates.items() if not x}))
    return summary
