"""Loopback raw-only reviewer service; no candidate modules are imported."""
import argparse,io,json,threading,subprocess,sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import numpy as np
from PIL import Image
from . import ROOT,REF,WORK
from .protocol import Repository,build,write,digest,status


def byte_range(header,size):
 if not header:return 0,size-1,200
 import re
 match=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
 if not match or not any(match.groups()):raise ValueError('Unsupported video byte range')
 first,last=match.groups()
 if not first:
  length=int(last)
  if length<=0:raise ValueError('Invalid suffix range')
  return max(0,size-length),size-1,206
 start=int(first);end=min(int(last),size-1) if last else size-1
 if start>=size or end<start:raise ValueError('Video byte range outside file')
 return start,end,206

class Server(ThreadingHTTPServer):
 daemon_threads=True
 def __init__(self,address,role,repo,read_only=False):
  self.role=role;self.repo=repo;self.read_only=read_only;self.verified_arrays={};self.verified_sources={};self.frame_lock=threading.Lock();self.write_lock=threading.Lock();super().__init__(address,Handler)

 def source(self,video):
  import pyarrow.parquet as pq
  path=ROOT/f'dataset/videos/{video}.MP4';stat=path.stat();fingerprint=(stat.st_size,stat.st_mtime_ns)
  if self.verified_sources.get(video)!=fingerprint:
   expected=pq.read_table(ROOT/f'canonical/v1/videos.{video}.parquet',columns=['sha256']).to_pylist()[0]['sha256']
   if digest(path)!=expected:raise ValueError('Original source checksum mismatch')
   self.verified_sources[video]=fingerprint
  return path

class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,code,value,mime='application/json'):
  raw=value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'");self.end_headers()
  if self.command!='HEAD':self.wfile.write(raw)
 def do_HEAD(self):
  self.do_GET()
 def do_GET(self):
  try:self.get()
  except (BrokenPipeError,ConnectionResetError):pass
  except (ValueError,KeyError,IndexError,OSError,subprocess.SubprocessError) as e:
   try:self.send(400,dict(error=str(e)))
   except (BrokenPipeError,ConnectionResetError):pass
 def get(self):
  expected=f'127.0.0.1:{self.server.server_address[1]}'
  if self.headers.get('Host')!=expected:return self.send(403,dict(error='Use the loopback review URL'))
  route=urlparse(self.path).path;parts=route.strip('/').split('/');role=self.server.role;repo=self.server.repo
  if route=='/favicon.ico':return self.send(204,b'','image/x-icon')
  if route=='/':return self.send(200,(ROOT/'src/track1_reference'/('adjudicator.html' if role=='adjudicator' else 'reviewer.html')).read_bytes(),'text/html; charset=utf-8')
  if route=='/media.js':return self.send(200,(ROOT/'src/track1_reference/review_media.js').read_bytes(),'text/javascript')
  if route=='/app.js':return self.send(200,(ROOT/'src/track1_reference'/('adjudicator.js' if role=='adjudicator' else 'reviewer.js')).read_bytes(),'text/javascript')
  if route=='/api/tasks':
   tasks=[]
   for taskid,t in repo.tasks.items():
    own=repo.own(role,taskid);tasks.append(dict(t,status='submitted' if own and own.get('submitted') else 'draft' if own else 'pending'))
   return self.send(200,dict(role=role,read_only=self.server.read_only,tasks=tasks))
  if len(parts)==3 and parts[:2]==['api','clip']:
   taskid=parts[2];task=repo.tasks[taskid];entry=repo.private['tasks'][taskid];frames=repo.private['clips'][task['clip_id']]
   self.server.source(entry['video_id'])
   return self.send(200,dict(url=f'/api/video/{taskid}',start_seconds=frames[0]['timestamp_us']/1e6,end_seconds=entry['end_us']/1e6,marked_seconds=frames[task['query_index']]['timestamp_us']/1e6,frame_seconds=[f['timestamp_us']/1e6 for f in frames],playback='Original source video, no transcoding or exported clip'))
  if len(parts)==3 and parts[:2]==['api','video']:
   taskid=parts[2]
   if taskid not in repo.tasks:raise ValueError('Unknown task')
   video=repo.private['tasks'][taskid]['video_id'];path=self.server.source(video);size=path.stat().st_size
   try:start,end,code=byte_range(self.headers.get('Range'),size)
   except ValueError:
    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
   self.send_response(code);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','private, max-age=3600');self.send_header('X-Content-Type-Options','nosniff')
   if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
   self.end_headers()
   if self.command=='HEAD':return
   with path.open('rb') as source:
    source.seek(start);remaining=end-start+1
    while remaining:
     chunk=source.read(min(256*1024,remaining))
     if not chunk:break
     self.wfile.write(chunk);remaining-=len(chunk)
   return
  if len(parts)==3 and parts[:2]==['api','own']:return self.send(200,repo.own(role,parts[2]))
  if len(parts)==3 and parts[:2]==['api','reviews']:
   if role!='adjudicator':return self.send(403,dict(error='Other reviewer annotations are hidden'))
   taskid=parts[2];values={r:repo.own(r,taskid) for r in ['A','B']}
   if not all(x and x.get('submitted') for x in values.values()):return self.send(409,dict(error='Two submitted independent reviews are required'))
   return self.send(200,values)
  if len(parts)==4 and parts[:2]==['api','frame']:
   taskid,index=parts[2],int(parts[3]);task=repo.tasks[taskid]
   if not 0<=index<task['frame_count']:raise ValueError('Frame outside task')
   entry=repo.private['clips'][task['clip_id']][index];path=(ROOT/entry['path']).resolve()
   if not path.is_relative_to((ROOT/'dataset').resolve()):raise ValueError('Frame path outside dataset')
   with self.server.frame_lock:
    copied=path.is_file()
    if copied:
     stat=path.stat();fingerprint=(stat.st_size,stat.st_mtime_ns);receipt=path.with_name(path.name.replace('.rgb.npy','.json'))
     if self.server.verified_arrays.get(str(path))!=fingerprint:
      copied=receipt.is_file() and digest(path)==json.loads(receipt.read_text()).get('sha256')
      if copied:self.server.verified_arrays[str(path)]=fingerprint
    if copied:
     a=np.load(path,mmap_mode='r');rgb=np.array(a[entry['storage_row']],copy=True);a._mmap.close();buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG',compress_level=0);raw=buffer.getvalue()
    else:
     import pyarrow.parquet as pq
     video=entry['video_id'];source=ROOT/f'dataset/videos/{video}.MP4';stat=source.stat();fingerprint=(stat.st_size,stat.st_mtime_ns)
     if self.server.verified_sources.get(video)!=fingerprint:
      expected=pq.read_table(ROOT/f'canonical/v1/videos.{video}.parquet',columns=['sha256']).to_pylist()[0]['sha256']
      if digest(source)!=expected:raise ValueError('Original source checksum mismatch')
      self.server.verified_sources[video]=fingerprint
     raw=subprocess.run([sys.executable,'-m','track1_reference.native_frame',video,str(entry['pts_raw'])],check=True,capture_output=True,timeout=30).stdout
   return self.send(200,raw,'image/png')
  if len(parts)==3 and parts[:2]==['api','source']:
   if role!='adjudicator' or not repo.accepted(parts[2]):return self.send(403,dict(error='Source annotations stay hidden until independent adjudication is locked'))
   import pyarrow.parquet as pq
   from data_v2.geometry import normalize
   row=repo.private['tasks'][parts[2]];wanted=[row.get('hand_mask_uid'),row.get('object_mask_uid')];wanted=[x for x in wanted if x];source=pq.read_table(ROOT/f'canonical/v2/masks.{row["video_id"]}.parquet',columns=['contours_json','annotation_width','annotation_height'],filters=[('mask_uid','in',wanted)]).to_pylist() if wanted else [];curves=[]
   for r in source:
    for ring in normalize(r['contours_json']):curves.append((np.asarray(ring['points'])*[1920/r['annotation_width'],1080/r['annotation_height']]).tolist())
   return self.send(200,dict(original_source_annotation_only=True,frame_alignment_delta_us=row.get('frame_alignment_delta_us'),curves=curves))
  return self.send(404,dict(error='Not a reviewer route'))
 def do_POST(self):
  try:
   expected=f'http://127.0.0.1:{self.server.server_address[1]}'
   if self.headers.get('Host')!=expected.removeprefix('http://') or self.headers.get('Origin') not in [None,expected] or self.headers.get('X-Review-Intent')!='annotation' or self.headers.get('Content-Type')!='application/json':return self.send(403,dict(error='Same-origin annotation requests only'))
   if self.server.read_only:return self.send(403,dict(error='Read-only verification session; annotations cannot be written'))
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<2000000:raise ValueError('Invalid annotation size')
   value=json.loads(self.rfile.read(length));parts=urlparse(self.path).path.strip('/').split('/')
   if len(parts)!=3 or parts[:2] not in [['api','draft'],['api','submit'],['api','source_audit']]:return self.send(404,dict(error='Not an annotation route'))
   with self.server.write_lock:
    if parts[1]=='source_audit':
     if self.server.role!='adjudicator' or not self.server.repo.accepted(parts[2]):return self.send(403,dict(error='Reference must be independently adjudicated first'))
     allowed=json.loads((REF/'protocol.json').read_text())['source_failure_categories']
     if value.get('category') not in allowed or not value.get('notes'):raise ValueError('Choose a source failure category and explain the evidence')
     write(self.server.repo.base/'source_audit'/f'{parts[2]}.json',dict(value,reference_sha256=digest(self.server.repo.path('adjudicator',parts[2]))));result=dict(saved=True)
    else:
     result=self.server.repo.save(self.server.role,parts[2],value,submit=parts[1]=='submit')
     if parts[1]=='submit' and self.server.repo.base==REF:status(self.server.repo)
   self.send(200,result)
  except (ValueError,KeyError,TypeError) as e:self.send(400,dict(error=str(e)))
  except (BrokenPipeError,ConnectionResetError):pass

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['A','B','adjudicator'],default='A');parser.add_argument('--port',type=int,default=8765);parser.add_argument('--read-only',action='store_true');args=parser.parse_args();build();server=Server(('127.0.0.1',args.port),args.role,Repository(),args.read_only);print(f'Raw-only reviewer {args.role}: http://127.0.0.1:{args.port}',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':main()
