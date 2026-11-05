from __future__ import annotations
from bisect import bisect_left
from fractions import Fraction
from statistics import median
import json
import numpy as np
from .common import ROOT,LOG,TableWriter,write_rows,read,uid,digest,stage_manifest

class Timeline:
    def __init__(self,rows,margin=1000):
        self.rows=rows;self.margin=margin
        self.ordered=sorted((r for r in rows if r['timestamp_us'] is not None),key=lambda r:(r['timestamp_us'],r['decode_index']))
        self.times=[r['timestamp_us'] for r in self.ordered]
    def nearest(self,t):
        if t is None or not self.times:return None,None,None
        i=bisect_left(self.times,t)
        options=[j for j in [i-1,i] if 0<=j<len(self.times)]
        j=min(options,key=lambda j:(abs(self.times[j]-t),self.times[j]))
        local=self.times[max(0,j-3):j+4]
        delta=[b-a for a,b in zip(local,local[1:]) if b>a]
        limit=int(median(delta)/2)+self.margin if delta else self.margin
        return self.ordered[j],abs(self.times[j]-t),limit

def index_video(cfg,choice,override=None):
    import av
    from av.codec.hwaccel import HWAccel
    v=choice['video_id'];out=ROOT/cfg['output'];path=ROOT/cfg['raw']/f'videos/{v}.MP4'
    actual=override or path
    if digest(actual,'md5')!=choice['official_md5']:raise ValueError(f'{v}: official MD5 mismatch')
    LOG.info('%s: full native decode/PTS indexing',v)
    count=0;p0=None;previous=None;last_time=None
    with av.open(str(actual)) as source:
        source_format=source.streams.video[0].codec_context.format.name
    device=cfg.get('hardware_device')
    hw=HWAccel(device,allow_software_fallback=False) if device else None
    with av.open(str(actual),hwaccel=hw) as c,TableWriter(out,'frames',v) as w:
        stream=c.streams.video[0];stream.codec_context.thread_count=cfg['decode_threads']
        tb=stream.time_base
        metadata=dict(video_id=v,participant_id=v.split('_')[0],path=str(path.relative_to(ROOT)),sha256=digest(actual),
                      official_md5=choice['official_md5'],action_split=choice['action_split'],visor_split=choice['visor_split'],
                      codec=stream.codec_context.name,decoder_backend=device or 'software',native_width=stream.width,native_height=stream.height,
                      pixel_format=source_format,time_base_num=tb.numerator,time_base_den=tb.denominator,license=cfg['license'])
        declared_duration=round(stream.duration*tb*1000000) if stream.duration is not None else None
        for i,frame in enumerate(c.decode(stream)):
            flags=[]
            if p0 is None and frame.pts is not None:p0=frame.pts
            t=round((frame.pts-p0)*tb*1000000) if frame.pts is not None and p0 is not None else None
            if t is None:flags.append('missing_pts')
            if t is not None and last_time is not None and t<=last_time:flags.append('non_increasing_presentation_time')
            duration=round(frame.duration*tb*1000000) if frame.duration else None
            row=dict(video_id=v,native_frame_uid=uid(v,'native',i,frame.pts),decode_index=i,pts_raw=frame.pts,
                     dts_raw=frame.dts,time_base_num=tb.numerator,time_base_den=tb.denominator,
                     timestamp_us=t,duration_us=duration,keyframe=frame.key_frame,native_width=frame.width,
                     native_height=frame.height,pixel_format=frame.format.name,decode_status='decoded',quality_flags=flags)
            if previous is not None:
                if previous['duration_us'] is None and previous['timestamp_us'] is not None and t is not None:
                    previous['duration_us']=t-previous['timestamp_us']
                    previous['quality_flags'].append('duration_from_next_pts')
                w.add(previous)
            previous=row;last_time=t;count=i+1
            if count%10000==0:LOG.info('%s: decoded %d frames',v,count)
        if previous is None:raise ValueError('No decoded frames: '+v)
        w.add(previous)
        end=previous['timestamp_us'] or 0
        end+=previous['duration_us'] or 0
        metadata.update(p0=p0,frame_count=count,duration_us=max(end,declared_duration or 0))
    video_path=write_rows(out,'videos',v,[metadata])
    stage_manifest(cfg,v,'index_video_pts',[actual],[video_path,out/f'frames.{v}.parquet'])
    return metadata

def load_timeline(cfg,v):return Timeline(read(ROOT/cfg['output'],'frames',v),cfg['alignment_rounding_margin_us'])

if __name__=='__main__':
    import argparse,logging
    p=argparse.ArgumentParser();p.add_argument('--config-json',required=True);p.add_argument('--choice-json',required=True)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    index_video(json.loads(a.config_json),json.loads(a.choice_json))
