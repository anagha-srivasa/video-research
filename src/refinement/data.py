from collections import defaultdict
import json,logging
import numpy as np
import pyarrow.parquet as pq
import duckdb
from pipeline.common import ROOT,digest,atomic_json
from track1.spatial import resample,dominant_ring
from track1.artifacts import load_shard,save_shard,save_index,memory_guard
from .protocol import config,folder,rank,split,source_identity

LOG=logging.getLogger('refinement')

def build_video(video):
    cfg=config();memory_guard(cfg);out=folder('cache');base=ROOT/'derived/track1_spatial_v1'
    fingerprints={p.name:digest(p) for p in [base/f'{video}.pt',base/f'{video}.parquet',ROOT/f'canonical/v1/masks.{video}.parquet']}
    marker=out/f'{video}.cache.json'
    if marker.exists() and json.loads(marker.read_text()).get('sources')==fingerprints:
        return json.loads(marker.read_text())
    rows=pq.read_table(base/f'{video}.parquet').to_pylist();state=load_shard(base/f'{video}.pt')
    selected=[]
    for kind,n in cfg['spatial_cases_per_kind'].items():
        available=[r for r in rows if r['source_kind']==kind]
        # Include old invalid fits; never improve metrics by silently dropping them.
        available.sort(key=lambda r:rank(str(cfg['seed'])+r['mask_uid']))
        selected+=available[:n]
    wanted=[r['mask_uid'] for r in selected]
    with duckdb.connect() as db:
        db.execute('SET threads=1')
        values=db.execute('SELECT mask_uid,contours_json,annotation_width,annotation_height FROM read_parquet(?) WHERE mask_uid IN (SELECT unnest(?))',[str(ROOT/f'canonical/v1/masks.{video}.parquet'),wanted]).fetchall()
    geometry={uid:(rings,w,h) for uid,rings,w,h in values};ys=[];baseline=[];metadata=[]
    for r in selected:
        rings,aw,ah=geometry[r['mask_uid']];transform=np.diag([r['width']/aw,r['height']/ah,1.])
        points,_,_=dominant_ring(dict(contours_json=rings),transform)
        y=resample(points,128);y=np.roll(y,-2*r['seam_shift'],axis=0)
        ys.append(y);baseline.append(state['controls'][r['row_index']]);metadata.append(dict(**{k:r[k] for k in ['mask_uid','video_id','source_kind','geometry_weight','provenance_weight','component_count','hole_count','spatial_valid']},split=split(video),row_index=len(metadata)))
    save_shard(out/f'{video}.spatial.pt',dict(points=np.stack(ys).astype(np.float32),baseline_controls=np.stack(baseline).astype(np.float32)))
    save_index(out/f'{video}.spatial.parquet',metadata)
    tracks=defaultdict(list)
    for r in rows:
        if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
    cases=[];arrays=defaultdict(list);offset=0
    for track in sorted(tracks,key=lambda t:rank(str(cfg['seed'])+t)):
        rs=tracks[track]
        if len(rs)<20:continue
        times=np.array([r['timestamp_us'] for r in rs],dtype=np.int64);ids=np.array([r['row_index'] for r in rs])
        candidates=np.arange(max(1000000,((times[0]+249999)//250000)*250000),times[-1]+1,250000)
        candidates=sorted(candidates,key=lambda t:rank(f'{track}:{t}'))
        taken=0
        for end in candidates:
            lo=np.searchsorted(times,end-1000000);hi=np.searchsorted(times,end,side='right');ts=times[lo:hi]
            if len(ts)<20 or ts[-1]-ts[0]<650000:continue
            t=(ts-(end-1000000))/1e6
            if max(np.diff(np.r_[0,t,1]))>.35:continue
            local=ids[lo:hi];n=len(local)
            for name,value in dict(time=t,h=state['h'][local],variance=state['variance'][local],prior=np.array([r['provenance_weight'] for r in rs[lo:hi]]),quality=np.array([r['geometry_weight'] for r in rs[lo:hi]])).items():arrays[name].append(value)
            cases.append(dict(video_id=video,entity_track_id=track,window_end_us=int(end),offset=offset,count=n,split=split(video)));offset+=n;taken+=1
            if taken>=cfg['max_windows_per_track']:break
        if len(cases)>=cfg['temporal_cases_per_video']:break
    save_shard(out/f'{video}.temporal.pt',{k:np.concatenate(v).astype(np.float64) for k,v in arrays.items()});save_index(out/f'{video}.temporal.parquet',cases)
    report=dict(video_id=video,sources=fingerprints,spatial_cases=len(selected),temporal_cases=len(cases),split=split(video),sampling_seed=cfg['seed'])
    atomic_json(marker,report);LOG.info('Cache %s %s',video,report);return report

def spatial_cases(videos):
    result=[]
    for v in videos:
        data=load_shard(folder('cache')/f'{v}.spatial.pt');meta=pq.read_table(folder('cache')/f'{v}.spatial.parquet').to_pylist()
        for i,r in enumerate(meta):result.append(dict(**r,y=data['points'][i].astype(float),baseline=data['baseline_controls'][i].astype(float)))
    return result

def temporal_cases(videos):
    result=[]
    active=folder('reports')/'representation.json'
    refined=active.exists() and json.loads(active.read_text()).get('temporal_source')=='refined'
    for v in videos:
        data=load_shard(folder('cache')/f'{v}.temporal.pt');meta=pq.read_table(folder('cache')/f'{v}.temporal.parquet').to_pylist()
        if refined:
            base=ROOT/'derived/track1_refined_v2/spatial';state=load_shard(base/f'{v}.pt');rows=pq.read_table(base/f'{v}.parquet',columns=['entity_track_id','timestamp_us','row_index']).to_pylist()
            lookup={(r['entity_track_id'],r['timestamp_us']):r['row_index'] for r in rows}
        for r in meta:
            a=r['offset'];b=a+r['count'];case=dict(**r,**{k:x[a:b] for k,x in data.items()})
            if refined:
                indices=[lookup[(r['entity_track_id'],r['window_end_us']-1000000+int(round(t*1e6)))] for t in case['time']]
                case['h']=state['h'][indices].astype(float);case['variance']=state['variance'][indices].astype(float)
            result.append(case)
    return result

def holdout(t,protocol,centre=.5):
    if protocol=='single':mask=np.zeros(len(t),bool);mask[np.argmin(abs(t-centre))]=True
    elif protocol=='gap100':mask=abs(t-centre)<=.05
    elif protocol=='endpoint100':mask=t>=.9
    else:raise ValueError(protocol)
    return mask
