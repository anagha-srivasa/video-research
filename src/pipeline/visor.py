from __future__ import annotations
import collections
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
from .common import ROOT,LOG,TableWriter,uid,dumps,digest,atomic_json,stage_manifest,read,write_rows,seconds_us
from .video import load_timeline
from .geometry import geometry

def number(name):return int(re.search(r'(\d+)\.(jpg|png)$',name)[1])

def prepare_dense(cfg,v):
    raw=ROOT/cfg['raw']/f'visor/{v}_interpolations.zip'
    staging=ROOT/cfg['staging'];uncorrected=staging/'visor_uncorrected';corrected=staging/'visor_corrected'
    uncorrected.mkdir(exist_ok=True);corrected.mkdir(exist_ok=True)
    with zipfile.ZipFile(raw) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failure: '+bad)
        names=[n for n in z.namelist() if Path(n).name==f'{v}_interpolations.json' and not n.startswith('__MACOSX/')]
        if len(names)!=1:raise ValueError('Unexpected interpolation archive contents: '+v)
        path=uncorrected/f'{v}_interpolations.json'
        with path.open('wb') as f,z.open(names[0]) as src:
            import shutil;shutil.copyfileobj(src,f)
    # Use the pinned official function, not a local reimplementation. This pilot
    # excludes the two special sparse-insertion videos; main still calls that
    # official routine and sees neither special input file in this directory.
    vendor=staging/'vendor_support'
    result=corrected/path.name
    result.unlink(missing_ok=True)  # A premature upstream exit must not reuse stale output.
    wrapper=("import importlib.util;from pathlib import Path;"
             f"s=importlib.util.spec_from_file_location('official',{str(vendor/'correct_interpolations.py')!r});"
             "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
             f"m.correct_interpolations({str(uncorrected)!r},{str(corrected)!r},specific_json_list=[{v!r}]);"
             f"m.add_mssing_sparse_annotations({str(corrected)!r},[])")
    with (staging/f'correction.{v}.log').open('w') as log:
        subprocess.run([sys.executable,'-c',wrapper],cwd=vendor,stdout=log,stderr=subprocess.STDOUT,check=True)
    data=json.loads(result.read_text())
    if not data['video_annotations']:raise ValueError('Correction emitted no annotations: '+v)
    if any(a['image']['name'][0]!='P' for a in data['video_annotations']):raise ValueError('Correction failed to normalize names')
    stage_manifest(cfg,v,'prepare_visor_dense',[raw,vendor/'correct_interpolations.py',vendor/'commit.json',vendor/'P02_01.json',vendor/'P03_14.json'],[result],
                   dict(tool_commit=json.loads((vendor/'commit.json').read_text())['sha'],
                        special_insertions='not_applicable_to_pilot',record_count=len(data['video_annotations'])))
    return result

def contact_value(raw,targets):
    if raw=='hand-not-in-contact':return 'hand_not_in_contact',None,'not_applicable'
    if raw=='none-of-the-above':return 'unsegmented_contact_target',None,'target_not_segmented'
    if raw=='inconclusive':return 'inconclusive',None,'inconclusive'
    matches=targets.get(str(raw),[])
    if len(matches)==1:return 'resolved_target',matches[0],'resolved'
    return 'unresolved_target_id',None,'duplicate_target_id' if matches else 'target_id_missing'

def hand_side(obj):
    raw=obj.get('on_which_hand');flags=[]
    if isinstance(raw,list):
        flags.append('source_hand_side_list:'+dumps(raw))
        side=raw[0] if len(raw)==1 and isinstance(raw[0],str) else 'inconclusive'
    elif isinstance(raw,str) and raw:side=raw
    elif raw is None:side=str(obj.get('name',''))
    else:side='inconclusive';flags.append('unsupported_source_hand_side:'+dumps(raw))
    name=str(obj.get('name','')).lower()
    if ('left' in name and 'right' in side.lower()) or ('right' in name and 'left' in side.lower()):flags.append('source_hand_name_side_disagreement')
    return side,flags

def normalize_visor(cfg,choice):
    v=choice['video_id'];out=ROOT/cfg['output'];raw=ROOT/cfg['raw']
    timeline=load_timeline(cfg,v);native=read(out,'videos',v)[0]
    sparse_path=raw/f'visor/{v}.json';dense_path=ROOT/cfg['staging']/f'visor_corrected/{v}_interpolations.json'
    sparse=json.loads(sparse_path.read_text())['video_annotations']
    timestamps=json.loads((ROOT/'work/vendor_support/frames_to_timestamps.json').read_text())['timestamps']
    epic=json.loads((raw/'visor/frame_mapping.json').read_text()).get(v,{})
    sparse_by_name={a['image']['name']:a for a in sparse}
    header=next((p for p in [ROOT/'work/vendor_support'/f'{v}.sparse_header.bin',ROOT/'work/vendor_support'/f'{v}.sparse_sample.jpg'] if p.exists()),None)
    if header is None:raise FileNotFoundError('Official sparse JPEG dimension verification is not complete for '+v)
    with Image.open(header) as image:sparse_width,sparse_height=image.size
    entity_stats={};counts=collections.Counter();quarantine=[]
    writers={name:TableWriter(out,name,v,cfg['batch_rows']) for name in ['annotation_frames','masks','contacts']}
    def align(image,is_dense):
        name=image['name'];jpg=name.replace('.png','.jpg');flags=[]
        official=seconds_us(timestamps.get(jpg))
        mapped=number(epic[jpg]) if jpg in epic else None
        if official is not None:
            row,error,limit=timeline.nearest(official)
            status='aligned_official_sparse' if row and error<=limit else 'unresolved'
        elif is_dense:
            # Native ordinal transport is an inference, never an official time.
            start=image.get('interpolation_start_frame','').replace('.png','.jpg')
            end=image.get('interpolation_end_frame','').replace('.png','.jpg')
            sa=seconds_us(timestamps.get(start));ea=seconds_us(timestamps.get(end))
            sr,se,sl=timeline.nearest(sa);er,ee,el=timeline.nearest(ea)
            row=None;error=None;limit=None;status='unresolved'
            if sr and er and se<=sl and ee<=el:
                step=number(name)-number(start);idx=sr['decode_index']+step
                endpoint_consistent=(er['decode_index']-sr['decode_index']==number(end)-number(start))
                if endpoint_consistent and 0<=idx<len(timeline.rows) and number(start)<=number(name)<=number(end):
                    row=timeline.rows[idx];status='dense_ordinal_inferred';limit=sl
                    flags.append('dense_time_inferred_from_sparse_anchor_and_native_ordinal')
            if row is None:flags.append('dense_anchor_missing_or_inconsistent')
        else:row=None;error=None;limit=None;status='unresolved'
        if row is None or status=='unresolved':
            return None,None,official,error,limit,status,mapped,flags+['time_alignment_unresolved']
        return row['native_frame_uid'],row['decode_index'],official,error,limit,status,mapped,flags
    try:
        for is_dense,path in [(False,sparse_path),(True,dense_path)]:
            records=sparse if not is_dense else json.loads(path.read_text())['video_annotations']
            source_hash=digest(path)
            for offset,record in enumerate(records):
                image=record['image'];name=image['name']
                if image.get('video')!=v:raise ValueError('Foreign annotation frame '+name)
                afuid=uid(v,source_hash,offset);interp=str(image['interpolation']) if 'interpolation' in image else None
                subsequence=image.get('subsequence')
                if is_dense and not subsequence:
                    anchors=[sparse_by_name.get(image.get(k,'').replace('.png','.jpg')) for k in ['interpolation_start_frame','interpolation_end_frame']]
                    seqs={a['image'].get('subsequence') for a in anchors if a}
                    if len(seqs)==1:subsequence=next(iter(seqs))
                nf,ni,vt,err,limit,status,mapped,aflags=align(image,is_dense)
                t=timeline.rows[ni]['timestamp_us'] if ni is not None else None
                width,height=(854,480) if is_dense else (sparse_width,sparse_height)
                transform_valid=is_dense or (width==native['native_width'] and height==native['native_height'])
                if not transform_valid:aflags.append('annotation_coordinate_transform_unverified')
                writers['annotation_frames'].add(dict(video_id=v,annotation_frame_uid=afuid,
                    source_kind='dense' if is_dense else 'manual_sparse',source_file=str(path.relative_to(ROOT)),source_record_offset=offset,
                    visor_frame_name=name,epic_mapped_frame_index=mapped,native_frame_uid=nf,native_decode_index=ni,
                    visor_timestamp_us=vt,timestamp_us=t,alignment_error_us=err,alignment_limit_us=limit,alignment_status=status,
                    annotation_width=width,annotation_height=height,dimension_source='official_dense_854x480' if is_dense else str(header.relative_to(ROOT)),
                    T_annotation_to_native=[native['native_width']/width,0.,0.,0.,native['native_height']/height,0.,0.,0.,1.],
                    transform_valid=transform_valid,transform_reason='official_dense_full_frame_scale' if is_dense else 'official_sparse_jpeg_matches_native_grid' if transform_valid else 'manual_resize_or_crop_unverified',
                    annotation_coordinate_system='annotation_pixel_xy',native_coordinate_system='native_video_pixel_xy',
                    subsequence=subsequence,interpolation_id=interp,quality_flags=aflags))
                targets=collections.defaultdict(list);mask_records=[]
                for j,obj in enumerate(record['annotations']):
                    typ=obj.get('type') if is_dense else None
                    kind=('dense_gt_endpoint' if typ==1 else 'dense_interpolated' if typ==0 else 'unknown_dense_type') if is_dense else 'manual_sparse'
                    source_id=str(obj.get('id',obj.get('key','')))
                    scope=dumps([v,subsequence or f'interpolation:{interp}',source_id])
                    euid=uid('provisional_entity',scope) if source_id else uid('unlinked_observation',afuid,j)
                    muid=uid(afuid,j,source_id);segments=obj.get('segments',[])
                    geom=geometry(segments,width,height)
                    flags=list(aflags)+geom.pop('quality_flags')
                    if kind=='unknown_dense_type':flags.append('unknown_mask_provenance')
                    if not subsequence:flags.append('identity_scoped_to_interpolation_only')
                    if not source_id:flags.append('missing_source_identity')
                    row=dict(video_id=v,mask_uid=muid,annotation_frame_uid=afuid,timestamp_us=t,source_mask_id=source_id,
                        source_kind=kind,interpolation_id=interp,interpolation_type=typ,entity_uid=euid,entity_scope=scope,
                        entity_name_open=str(obj.get('name','')),class_id=obj.get('class_id'),exhaustive_status=obj.get('exhaustive','unavailable'),
                        segments_raw=dumps(segments),annotation_width=width,annotation_height=height,quality_prior=None,
                        quality_flags=sorted(set(flags)),source_file=str(path.relative_to(ROOT)),source_record_offset=offset,**geom)
                    writers['masks'].add(row);mask_records.append((obj,row));targets[source_id].append(row)
                    counts[kind]+=1
                    if geom['geometry_status']=='quarantined':
                        counts['quarantined_geometry']+=1
                        if len(quarantine)<100:quarantine.append(dict(mask_uid=muid,flags=row['quality_flags']))
                    st=entity_stats.setdefault(euid,dict(scope=scope,id=source_id,names=set(),count=0,start=None,end=None,flags=set(),previous=None,frame=None))
                    st['names'].add(row['entity_name_open']);st['count']+=1
                    if t is not None:
                        st['start']=min(st['start'],t) if st['start'] is not None else t
                        st['end']=max(st['end'],t) if st['end'] is not None else t
                    if st['frame']==afuid:st['flags'].add('simultaneous_duplicate_source_id')
                    prev=st['previous']
                    if prev and t is not None and 0<t-prev[0]<=250000 and row['centroid_x'] is not None:
                        motion=((row['centroid_x']/width-prev[1])**2+(row['centroid_y']/height-prev[2])**2)**.5
                        if motion>.4:st['flags'].add('large_short_time_displacement')
                    if t is not None and row['centroid_x'] is not None:st['previous']=(t,row['centroid_x']/width,row['centroid_y']/height)
                    st['frame']=afuid
                for obj,hand in mask_records:
                    if 'in_contact_object' not in obj:continue
                    raw_contact=obj['in_contact_object'];cs,target,resolution=contact_value(raw_contact,targets)
                    side,side_flags=hand_side(obj)
                    writers['contacts'].add(dict(video_id=v,contact_uid=uid(hand['mask_uid'],'contact'),annotation_frame_uid=afuid,timestamp_us=t,
                        hand_mask_uid=hand['mask_uid'],hand_side=side,contact_status=cs,
                        raw_contact_value=dumps(raw_contact),target_source_mask_id=str(raw_contact) if cs in ['resolved_target','unresolved_target_id'] else None,
                        target_mask_uid=target['mask_uid'] if target else None,target_entity_uid=target['entity_uid'] if target else None,
                        source_kind=hand['source_kind'],interpolation_id=interp,resolution_status=resolution,
                        quality_flags=side_flags+(['contact_target_unresolved'] if cs=='unresolved_target_id' else [])))
                if (offset+1)%2000==0:LOG.info('%s: %s frames %d/%d; masks=%d',v,'dense' if is_dense else 'manual',offset+1,len(records),sum(counts[k] for k in ['manual_sparse','dense_gt_endpoint','dense_interpolated']))
        for writer in writers.values():writer.close()
    except BaseException:
        for writer in writers.values():
            try:writer.writer.close();writer.temp.unlink(missing_ok=True)
            except Exception:pass
        raise
    entities=[]
    for euid,st in sorted(entity_stats.items()):
        if len(st['names'])>1:st['flags'].add('same_source_id_multiple_names')
        entities.append(dict(video_id=v,entity_uid=euid,entity_scope=st['scope'],source_mask_id=st['id'],names_json=dumps(sorted(st['names'])),
            observation_count=st['count'],start_us=st['start'],end_us=st['end'],identity_status='provisional_source_scope_not_verified_track',quality_flags=sorted(st['flags'])))
    write_rows(out,'entities',v,entities)
    stage_manifest(cfg,v,'normalize_visor',[sparse_path,dense_path,header,ROOT/'work/vendor_support/frames_to_timestamps.json',ROOT/'work/vendor_support/generate_dense_frames.py',raw/'visor/frame_mapping.json',out/f'frames.{v}.parquet'],
                   [out/f'{name}.{v}.parquet' for name in [*writers,'entities']],dict(counts=counts,quarantine_examples=quarantine))
