from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import json
import subprocess
import sys
from pathlib import Path
import duckdb
import numpy as np
from .common import ROOT,SCHEMAS

@dataclass
class CanonicalWindow:
    video_id:str
    participant_id:str
    start_us:int
    end_us:int
    frame_timestamps_us:np.ndarray
    frame_refs:list
    entity_observations:list
    mask_observations:list
    contact_relations:list
    camera_observations:list
    action_intervals:list
    action_split:str
    visor_split:str
    coverage:dict
    quality_flags:list

class ByteCache:
    def __init__(self,limit):self.limit=limit;self.size=0;self.items=OrderedDict()
    def get(self,key):
        if key not in self.items:return None
        value=self.items.pop(key);self.items[key]=value;return value.copy()
    def put(self,key,value):
        if value.nbytes>self.limit:return
        if key in self.items:self.size-=self.items.pop(key).nbytes
        while self.items and self.size+value.nbytes>self.limit:self.size-=self.items.popitem(last=False)[1].nbytes
        self.items[key]=value.copy();self.size+=value.nbytes

class Dataset:
    def __init__(self,path=None,*,root=None,require_ready=True,cache_bytes=64*1024*1024):
        self.root=Path(root or ROOT);self.path=Path(path or self.root/'canonical/v1')
        status=self.path/'status.json'
        if require_ready and (not status.exists() or not json.loads(status.read_text()).get('ingestion_ready')):
            raise RuntimeError('Canonical dataset has not passed its ingestion-readiness gates. Use require_ready=False only for explicit QA.')
        self.db=duckdb.connect();self.db.execute('SET threads=2');self.cache=ByteCache(cache_bytes);self.decoder=None
        for table in SCHEMAS:
            paths=sorted(str(p) for p in self.path.glob(f'{table}.*.parquet'))
            if paths:self.db.from_parquet(paths).create_view(table)
    def close(self):
        self.db.close()
        if self.decoder is not None:
            self.decoder.stdin.close()
            try:self.decoder.wait(timeout=5)
            except subprocess.TimeoutExpired:self.decoder.terminate();self.decoder.wait(timeout=5)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def rows(self,sql,params=None):
        cur=self.db.execute(sql,params or []);columns=[x[0] for x in cur.description]
        return [dict(zip(columns,r)) for r in cur.fetchall()]
    def get_window(self,video_id,*,end_time_us,duration_us=1000000):
        if duration_us<=0 or end_time_us<duration_us:raise ValueError('A window must have positive duration and a nonnegative start')
        start=end_time_us-duration_us;end=end_time_us
        videos=self.rows('SELECT * FROM videos WHERE video_id=?',[video_id])
        if len(videos)!=1:raise KeyError(video_id)
        video=videos[0];params=[video_id,start,end]
        frames=self.rows('SELECT * FROM frames WHERE video_id=? AND timestamp_us>=? AND timestamp_us<? ORDER BY timestamp_us,decode_index',params)
        masks=self.rows('SELECT * FROM masks WHERE video_id=? AND timestamp_us>=? AND timestamp_us<? ORDER BY timestamp_us,mask_uid',params)
        annotation_rows={a['annotation_frame_uid']:a for a in self.rows('SELECT * FROM annotation_frames WHERE video_id=? AND timestamp_us>=? AND timestamp_us<?',params)}
        calibration=self.rows('SELECT T_native_to_camera,transform_valid,transform_reason FROM camera_frames WHERE video_id=? LIMIT 1',[video_id])
        for mask in masks:
            a=annotation_rows[mask['annotation_frame_uid']]
            mapping=dict(T_annotation_to_native=a['T_annotation_to_native'],transform_valid=a['transform_valid'],transform_reason=a['transform_reason'],
                         annotation_coordinate_system=a['annotation_coordinate_system'],native_coordinate_system=a['native_coordinate_system'])
            if calibration:
                c=calibration[0]
                mapping.update(T_annotation_to_camera=(np.array(c['T_native_to_camera']).reshape(3,3)@np.array(a['T_annotation_to_native']).reshape(3,3)).ravel().tolist(),
                               camera_transform_valid=bool(a['transform_valid'] and c['transform_valid']),camera_transform_reason=c['transform_reason'])
            mask['coordinate_transform']=mapping
        contacts=self.rows('SELECT * FROM contacts WHERE video_id=? AND timestamp_us>=? AND timestamp_us<? ORDER BY timestamp_us,contact_uid',params)
        actions=self.rows('SELECT * FROM actions WHERE video_id=? AND start_us<? AND stop_us>? ORDER BY start_us,action_uid',[video_id,end,start])
        entities=self.rows('SELECT * FROM entities WHERE entity_uid IN (SELECT DISTINCT entity_uid FROM masks WHERE video_id=? AND timestamp_us>=? AND timestamp_us<?) ORDER BY entity_uid',params)
        camera=self.rows('SELECT * FROM camera_frames WHERE video_id=? AND timestamp_us>=? AND timestamp_us<? ORDER BY timestamp_us,camera_uid',params)
        poses_by_frame={}
        for c in camera:
            if c['pose_valid']:poses_by_frame.setdefault(c['native_frame_uid'],[]).append(c)
        for group in poses_by_frame.values():
            if len(group)>1:
                for c in group:c['quality_flags']=c['quality_flags']+['multiple_camera_records_for_native_frame']
        camera_observations=camera+[dict(video_id=video_id,native_frame_uid=f['native_frame_uid'],timestamp_us=f['timestamp_us'],pose_valid=False,transform_valid=False,quality_flags=['no_registered_aligned_camera_pose']) for f in frames if f['native_frame_uid'] not in poses_by_frame]
        unresolved=self.rows("SELECT count(*) AS n FROM annotation_frames WHERE video_id=? AND alignment_status='unresolved'",[video_id])[0]['n']
        flags=sorted({x for records in [frames,masks,contacts,camera,actions] for r in records for x in r['quality_flags']})
        if unresolved:flags.append('video_contains_unresolved_annotation_times')
        coverage=dict(native_frames=len(frames),mask_observations=len(masks),manual_masks=sum(m['source_kind']=='manual_sparse' for m in masks),
            interpolated_masks=sum(m['source_kind']=='dense_interpolated' for m in masks),camera_pose_fraction=sum(f['native_frame_uid'] in poses_by_frame for f in frames)/len(frames) if frames else 0.,
            unresolved_annotation_frames_in_video=unresolved,source_end_us=video['duration_us'])
        refs=[dict(native_frame_uid=f['native_frame_uid'],video_id=video_id,timestamp_us=f['timestamp_us'],pts_raw=f['pts_raw'],decode_index=f['decode_index']) for f in frames]
        return CanonicalWindow(video_id,video['participant_id'],start,end,np.asarray([f['timestamp_us'] for f in frames],dtype=np.int64),refs,entities,masks,contacts,camera_observations,actions,video['action_split'],video['visor_split'],coverage,sorted(set(flags)))
    def get_frame(self,video_id,timestamp_us):
        key=('rgb',video_id,timestamp_us);cached=self.cache.get(key)
        if cached is not None:return cached
        video=self.rows('SELECT * FROM videos WHERE video_id=?',[video_id])
        if len(video)!=1:raise KeyError(video_id)
        target=self.rows('SELECT * FROM frames WHERE video_id=? AND timestamp_us IS NOT NULL ORDER BY abs(timestamp_us-?), timestamp_us,decode_index LIMIT 1',[video_id,timestamp_us])[0]
        if self.decoder is None:self.decoder=subprocess.Popen([sys.executable,'-m','pipeline.decoder'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        request=dict(path=str(self.root/video[0]['path']),pts=target['pts_raw'])
        self.decoder.stdin.write((json.dumps(request)+'\n').encode());self.decoder.stdin.flush()
        line=self.decoder.stdout.readline()
        if not line:raise RuntimeError('Isolated decoder exited unexpectedly')
        header=json.loads(line)
        if 'error' in header:raise RuntimeError(header['error'])
        payload=self.decoder.stdout.read(header['bytes'])
        if len(payload)!=header['bytes']:raise RuntimeError('Incomplete isolated decoder response')
        rgb=np.frombuffer(payload,dtype=np.uint8).reshape(header['shape']).copy()
        self.cache.put(key,rgb);return rgb
    def get_mask(self,mask_uid):
        from .geometry import rasterize
        key=('mask',mask_uid);cached=self.cache.get(key)
        if cached is not None:return cached
        rows=self.rows('SELECT * FROM masks WHERE mask_uid=?',[mask_uid])
        if len(rows)!=1:raise KeyError(mask_uid)
        m=rows[0]
        if m['geometry_status']=='quarantined':raise ValueError('Mask quarantined: '+str(m['quality_flags']))
        result=rasterize(json.loads(m['segments_raw']),m['annotation_width'],m['annotation_height'])
        self.cache.put(key,result);return result
