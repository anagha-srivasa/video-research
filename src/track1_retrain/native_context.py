"""Isolated native-context decoder. Deliberately never imports OpenCV or pipeline."""
import json,sys,time,hashlib,shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
CTX=ROOT/'dataset/context_v3'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def atomic_json(path,data):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)
def guard():
 import psutil
 if psutil.virtual_memory().available<5*2**30:raise MemoryError('Available memory reserve')
 if shutil.disk_usage(ROOT).free<20000000000:raise OSError('Disk reserve')
def extract(v):
 # Kept separate from the published v2 extractor to preserve its source receipt.
 import av
 from av.codec.hwaccel import HWAccel
 rows=[r for r in pq.read_table(CTX/'extra_frames.parquet').to_pylist() if r['video_id']==v];path=CTX/f'{v}.rgb.npy';marker=CTX/f'{v}.json';source=ROOT/f'dataset/videos/{v}.MP4';fp=dict(index_sha256=digest(CTX/'extra_frames.parquet'),source_sha256=pq.read_table(ROOT/f'canonical/v1/videos.{v}.parquet').to_pylist()[0]['sha256'])
 if marker.exists() and path.exists():
  m=json.loads(marker.read_text())
  if m['inputs']==fp and m['sha256']==digest(path):return m
  raise ValueError('Existing context shard differs')
 assert digest(source)==fp['source_sha256'],'Source checksum mismatch'
 temp=path.with_suffix('.npy.tmp');array=np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(len(rows),1080,1920,3));wanted={r['pts_raw']:r['storage_row'] for r in rows};assert len(wanted)==len(rows);seen=set();started=time.monotonic()
 try:c=av.open(str(source),hwaccel=HWAccel('videotoolbox',allow_software_fallback=False));backend='videotoolbox'
 except Exception:c=av.open(str(source));backend='software'
 with c:
  stream=c.streams.video[0];stream.codec_context.thread_count=2
  for f in c.decode(stream):
   if f.pts not in wanted:continue
   assert (f.width,f.height)==(1920,1080);array[wanted[f.pts]]=f.to_ndarray(format='rgb24');seen.add(f.pts)
   if len(seen)%128==0:array.flush();array._mmap.close();array=np.load(temp,mmap_mode='r+');guard()
   if len(seen)==len(rows):break
 assert seen==set(wanted);array.flush();array._mmap.close();temp.replace(path);result=dict(video_id=v,frames=len(rows),bytes=path.stat().st_size,sha256=digest(path),inputs=fp,compression=None,backend=backend,seconds=time.monotonic()-started);atomic_json(marker,result);return result

if __name__=='__main__':extract(sys.argv[1])
