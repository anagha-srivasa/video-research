"""Bounded, contiguous native RGB clips. NPY shards have no codec or compression."""
import collections, json, logging, shutil, time
import numpy as np
import pyarrow.parquet as pq
from .common import *
LOG=logging.getLogger('data_v2')

def plan():
    import duckdb
    selected=[];episodes=[];max_frames=CFG['rgb_budget_bytes']//(1920*1080*3)
    per_video=max_frames//len(VIDEOS);duration=int(CFG['clip_seconds']*1e6)
    for v in VIDEOS:
        with duckdb.connect() as db:
            db.execute('SET threads=1');db.execute("SET memory_limit='1GB'")
            masks=db.execute('''SELECT m.mask_uid,m.timestamp_us,m.area_px,m.hole_count,m.component_count,m.source_kind,
                m.geometry_status,a.native_frame_uid FROM read_parquet(?) m JOIN read_parquet(?) a USING(annotation_frame_uid)
                WHERE m.timestamp_us IS NOT NULL AND m.source_kind='manual_sparse'
                ORDER BY m.timestamp_us,m.mask_uid''',[str(ROOT/f'canonical/v1/masks.{v}.parquet'),str(ROOT/f'canonical/v1/annotation_frames.{v}.parquet')]).fetchall()
            contact={r[0] for r in db.execute("SELECT hand_mask_uid FROM read_parquet(?) WHERE contact_status='resolved_target'",[str(ROOT/f'canonical/v1/contacts.{v}.parquet')]).fetchall()}
        frames=pq.read_table(ROOT/f'canonical/v1/frames.{v}.parquet').to_pylist()
        times=np.array([r['timestamp_us'] for r in frames]);groups=collections.defaultdict(list)
        for uid,t,area,holes,components,kind,status,native in masks:
            labels=['manual']
            if holes: labels.append('hole')
            if components and components>1: labels.append('multi_component')
            if area and area<854*480*.005: labels.append('small')
            if uid in contact: labels.append('contact_relation')
            if status=='quarantined': labels.append('quarantined')
            for label in labels: groups[label].append((rank(uid+label),uid,t))
        for label in groups: groups[label].sort()
        labels=sorted(groups);seen=set();used=set();chosen=[];cursor=collections.Counter()
        while True:
            progressed=False
            for label in labels:
                while cursor[label]<len(groups[label]):
                    _,uid,t=groups[label][cursor[label]];cursor[label]+=1
                    begin=max(0,t-duration//2);end=begin+duration
                    indices=np.flatnonzero((times>=begin)&(times<end));fresh=set(map(int,indices))-used
                    if not fresh or len(fresh)<len(indices)*.75: continue
                    if len(used|fresh)>per_video: continue
                    if uid in seen:continue
                    seen.add(uid);used.update(fresh);chosen.append(dict(video_id=v,split=split(v),episode_id=uid,start_us=int(begin),end_us=int(end),selection_stratum=label,anchor_mask_uid=uid));progressed=True;break
            if not progressed:break
        for i in sorted(used):
            selected.append(dict(frames[i],split=split(v),storage_row=0))
        local=[r for r in selected if r['video_id']==v]
        for i,r in enumerate(local):r['storage_row']=i
        episodes.extend(chosen)
        LOG.info('%s partition frames=%d episodes=%d strata=%s',v,len(used),len(chosen),dict(collections.Counter(r['selection_stratum'] for r in chosen)))
    save(FRAMES/'frames.parquet',selected);save(FRAMES/'episodes.parquet',episodes)
    rgb_bytes=len(selected)*1920*1080*3+len(VIDEOS)*128
    # All existing dataset source files count against the 250 GB product budget.
    sources_bytes=sum(p.stat().st_size for p in (ROOT/'dataset').rglob('*') if p.is_file() and FRAMES not in p.parents)
    labels_bytes=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file())
    if rgb_bytes+sources_bytes+labels_bytes>CFG['budget_bytes']:raise ValueError('Partition exceeds total product capacity')
    missing_bytes=sum(sum(r['video_id']==v for r in selected)*1920*1080*3+128 for v in VIDEOS if not (FRAMES/f'{v}.rgb.npy').exists())
    if shutil.disk_usage(ROOT).free<missing_bytes+CFG['disk_reserve_bytes']:raise OSError('Insufficient disk for uncompressed partition and reserve')
    result=dict(version=CFG['version'],frames=len(selected),episodes=len(episodes),rgb_bytes=rgb_bytes,
                source_bytes=sources_bytes,canonical_v2_bytes=labels_bytes,total_product_estimate_bytes=rgb_bytes+sources_bytes+labels_bytes,
                budget_bytes=CFG['budget_bytes'],rgb_format='uint8 RGB, native 1920x1080, NPY, no compression',
                split_policy='existing five training and two exploratory validation participants; no fresh test set',
                selection='round-robin manual/hole/multi_component/small/contact/quarantined, contiguous 2s clips; overlapping frames deduplicated',
                frames_index_sha256=digest(FRAMES/'frames.parquet'),episodes_sha256=digest(FRAMES/'episodes.parquet'),code=identity())
    atomic_json(FRAMES/'plan.json',result);return result

def extract(v):
    # This module never imports OpenCV: avoid conflicting macOS FFmpeg libraries.
    import av
    from av.codec.hwaccel import HWAccel
    split(v);rows=[r for r in pq.read_table(FRAMES/'frames.parquet').to_pylist() if r['video_id']==v]
    target=FRAMES/f'{v}.rgb.npy';marker=FRAMES/f'{v}.json';source=ROOT/f'dataset/videos/{v}.MP4'
    fingerprint=dict(source_sha256=digest(source),frame_index_sha256=digest(FRAMES/'frames.parquet'),code=identity())
    if marker.exists() and target.exists():
        previous=json.loads(marker.read_text())
        if previous['inputs']==fingerprint and previous['sha256']==digest(target):return previous
        raise ValueError('Existing partition differs; use a new version before replacement')
    temp=target.with_suffix('.npy.tmp');a=np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(len(rows),1080,1920,3))
    wanted={r['pts_raw']:r for r in rows}
    if len(wanted)!=len(rows):raise ValueError('Duplicate PTS in extraction request')
    seen=set();started=time.monotonic()
    backend='videotoolbox'
    try:
        ctx=av.open(str(source),hwaccel=HWAccel('videotoolbox',allow_software_fallback=False))
    except Exception:
        ctx=av.open(str(source));backend='software'
    with ctx as c:
        stream=c.streams.video[0];stream.codec_context.thread_count=2
        for frame in c.decode(stream):
            if frame.pts not in wanted:continue
            row=wanted[frame.pts]
            if frame.pts in seen:raise ValueError('Repeated PTS while extracting')
            if (frame.width,frame.height)!=(1920,1080):raise ValueError('Unexpected native resolution')
            # RGB conversion only: no resizing, frame dropping inside selected clips or compression.
            rgb=frame.to_ndarray(format='rgb24');a[row['storage_row']]=rgb;seen.add(frame.pts)
            if len(seen)%128==0:
                a.flush();a._mmap.close()
                a=np.load(temp,mmap_mode="r+")
                guard()
                if shutil.disk_usage(ROOT).free<CFG['disk_reserve_bytes']:raise OSError('Disk reserve reached')
                LOG.info('%s decoded %d/%d selected frames elapsed %.1fs',v,len(seen),len(rows),time.monotonic()-started)
            if len(seen)==len(rows):break
    if seen!=set(wanted):raise ValueError('Incomplete native PTS extraction')
    a.flush();del a;temp.replace(target)
    result=dict(video_id=v,frames=len(rows),shape=[len(rows),1080,1920,3],dtype='uint8',compression=None,
                bytes=target.stat().st_size,sha256=digest(target),inputs=fingerprint,decoder=backend,seconds=time.monotonic()-started)
    atomic_json(marker,result);return result

def run():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    p=plan()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as ex:
        futures=[ex.submit(extract,v) for v in VIDEOS]
        results=[]
        for f in as_completed(futures):results.append(f.result())
    atomic_json(REPORT/'partition.json',dict(plan=p,videos=results,complete=True,compression=None,code=identity()))
