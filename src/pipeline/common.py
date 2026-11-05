from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/deps'))
LOG=logging.getLogger('pipeline')

def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_publisher_file(path,entry):
    if Path(path).stat().st_size!=int(entry['extent']['@value']) or digest(path,'sha1')!=entry['sha1']:
        raise ValueError('Publisher size/SHA1 mismatch: '+str(path))

def dumps(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def uid(*parts):return hashlib.sha256(dumps(parts).encode()).hexdigest()[:32]
def config_hash(cfg):return hashlib.sha256(dumps(cfg).encode()).hexdigest()
def code_hash():return hashlib.sha256(dumps([(p.name,digest(p)) for p in sorted((ROOT/'src/pipeline').glob('*.py'))]).encode()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(dumps(value)+'\n');temp.replace(path)
def seconds_us(value):
    from decimal import Decimal, ROUND_HALF_EVEN
    if value is None or str(value).strip().lower() in ['', 'nan']:return None
    if ':' in str(value):
        h,m,s=value.split(':');v=Decimal(h)*3600+Decimal(m)*60+Decimal(s)
    else:v=Decimal(str(value))
    return int((v*1000000).quantize(Decimal('1'),rounding=ROUND_HALF_EVEN))

S=pa.string();I=pa.int64();F=pa.float64();B=pa.bool_();LS=pa.list_(S)
SCHEMAS={}
def schema(name,fields):SCHEMAS[name]=pa.schema(fields)
base=[('video_id',S)]
schema('videos',base+[('participant_id',S),('path',S),('sha256',S),('official_md5',S),('action_split',S),('visor_split',S),('codec',S),('decoder_backend',S),('native_width',I),('native_height',I),('pixel_format',S),('time_base_num',I),('time_base_den',I),('p0',I),('duration_us',I),('frame_count',I),('license',S)])
schema('frames',base+[('native_frame_uid',S),('decode_index',I),('pts_raw',I),('dts_raw',I),('time_base_num',I),('time_base_den',I),('timestamp_us',I),('duration_us',I),('keyframe',B),('native_width',I),('native_height',I),('pixel_format',S),('decode_status',S),('quality_flags',LS)])
schema('annotation_frames',base+[('annotation_frame_uid',S),('source_kind',S),('source_file',S),('source_record_offset',I),('visor_frame_name',S),('epic_mapped_frame_index',I),('native_frame_uid',S),('native_decode_index',I),('visor_timestamp_us',I),('timestamp_us',I),('alignment_error_us',I),('alignment_limit_us',I),('alignment_status',S),('annotation_width',I),('annotation_height',I),('dimension_source',S),('subsequence',S),('interpolation_id',S),('quality_flags',LS)])
schema('masks',base+[('mask_uid',S),('annotation_frame_uid',S),('timestamp_us',I),('source_mask_id',S),('source_kind',S),('interpolation_id',S),('interpolation_type',I),('entity_uid',S),('entity_scope',S),('entity_name_open',S),('class_id',I),('exhaustive_status',S),('segments_raw',S),('annotation_width',I),('annotation_height',I),('bbox_xmin',F),('bbox_ymin',F),('bbox_xmax',F),('bbox_ymax',F),('area_px',F),('centroid_x',F),('centroid_y',F),('perimeter_px',F),('component_count',I),('hole_count',I),('contours_json',S),('geometry_status',S),('repair_code',S),('rasterisation_policy',S),('quality_prior',F),('quality_flags',LS),('source_file',S),('source_record_offset',I)])
schema('contacts',base+[('contact_uid',S),('annotation_frame_uid',S),('timestamp_us',I),('hand_mask_uid',S),('hand_side',S),('contact_status',S),('raw_contact_value',S),('target_source_mask_id',S),('target_mask_uid',S),('target_entity_uid',S),('source_kind',S),('interpolation_id',S),('resolution_status',S),('quality_flags',LS)])
schema('entities',base+[('entity_uid',S),('entity_scope',S),('source_mask_id',S),('names_json',S),('observation_count',I),('start_us',I),('end_us',I),('identity_status',S),('quality_flags',LS)])
schema('actions',base+[('action_uid',S),('narration_id',S),('participant_id',S),('action_split',S),('visor_split',S),('start_us',I),('stop_us',I),('narration_timestamp_us',I),('start_frame_source',I),('stop_frame_source',I),('narration',S),('verb',S),('verb_class',I),('noun',S),('noun_class',I),('all_nouns',LS),('all_noun_classes',pa.list_(I)),('source_file',S),('quality_flags',LS)])
schema('camera_frames',base+[('camera_uid',S),('source_frame_name',S),('native_frame_uid',S),('timestamp_us',I),('alignment_status',S),('pose_valid',B),('camera_model',S),('camera_width',I),('camera_height',I),('camera_params',pa.list_(F)),('T_world_to_camera',pa.list_(F)),('T_camera_to_world',pa.list_(F)),('T_native_to_camera',pa.list_(F)),('transform_valid',B),('transform_reason',S),('geometry_scale',S),('source_file',S),('quality_flags',LS)])
schema('event_evidence',base+[('evidence_uid',S),('timestamp_us',I),('evidence_type',S),('source_uid',S),('previous_observed_us',I),('confidence_class',S),('quality_flags',LS)])
schema('windows_1s_hop250ms',base+[('window_uid',S),('start_us',I),('end_us',I),('native_frame_begin',I),('native_frame_end',I),('annotation_frame_uids',LS),('action_uids',LS),('camera_pose_coverage',F),('manual_mask_count',I),('interpolated_mask_count',I),('contact_count',I),('coverage_fraction',F),('quality_flags',LS)])
schema('qa_flags',base+[('record_uid',S),('stage',S),('severity',S),('code',S),('details',S)])
for name,fields in {
    'annotation_frames':[('T_annotation_to_native',pa.list_(F)),('transform_valid',B),('transform_reason',S),('annotation_coordinate_system',S),('native_coordinate_system',S)],
    'camera_frames':[('source_timestamp_estimate_us',I),('alignment_error_us',I),('alignment_limit_us',I)],
}.items():
    for field in fields:SCHEMAS[name]=SCHEMAS[name].append(pa.field(*field))

class TableWriter:
    def __init__(self,out,table,video,batch=1000):
        self.path=Path(out)/f'{table}.{video}.parquet';self.temp=self.path.with_suffix('.tmp')
        self.writer=pq.ParquetWriter(self.temp,SCHEMAS[table],compression='zstd',version='2.6')
        self.rows=[];self.batch=batch;self.count=0
    def add(self,row):
        self.rows.append(row);self.count+=1
        if len(self.rows)>=self.batch:self.flush()
    def flush(self):
        if self.rows:self.writer.write_table(pa.Table.from_pylist(self.rows,schema=self.writer.schema));self.rows=[]
    def close(self):self.flush();self.writer.close();self.temp.replace(self.path)
    def __enter__(self):return self
    def __exit__(self,t,v,tb):
        if t is None:self.close()
        else:self.writer.close();self.temp.unlink(missing_ok=True)

def read(out,table,video):return pq.read_table(Path(out)/f'{table}.{video}.parquet').to_pylist()
def write_rows(out,table,video,rows):
    with TableWriter(out,table,video) as w:
        for row in rows:w.add(row)
    return w.path

def source_paths(cfg,video):
    raw=ROOT/cfg['raw']
    return [raw/f'videos/{video}.MP4',raw/f'visor/{video}.json',raw/f'visor/{video}_interpolations.zip',raw/f'camera/{video}.json',raw/'visor/frame_mapping.json',raw/'actions/EPIC_100_train.csv',raw/'actions/EPIC_100_validation.csv',ROOT/'work/vendor_support/frames_to_timestamps.json',ROOT/'work/vendor_support/correct_interpolations.py',raw/'metadata/VISOR_publisher_info.json']

def tool_versions():
    import importlib.metadata
    return {**{p:importlib.metadata.version(p) for p in ['numpy','scipy','pyarrow','duckdb','shapely','opencv-python-headless','av','Pillow','tqdm','PyYAML']},'python':sys.version}

def stage_manifest(cfg,video,stage,inputs,outputs,extra=None):
    versions=tool_versions()
    value=dict(stage=stage,video_id=video,schema_version=cfg['schema_version'],config_sha256=config_hash(cfg),
               source_files={str(Path(p).relative_to(ROOT)):digest(p) for p in inputs},
               outputs={str(Path(p).relative_to(ROOT)):digest(p) for p in outputs},
               pipeline_sha256=code_hash(),
               tool_versions=versions,completed=True,extra=extra or {})
    atomic_json(ROOT/cfg['output']/f'{stage}.{video}.manifest.json',value)
    return value
