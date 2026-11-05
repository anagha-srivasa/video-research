"""Read-only local interface for Qwen outputs; separate from blinded human review."""
import json,sqlite3,threading,collections,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from . import ROOT,REF,WORK
from .adaptive_review import STORE,group_key
from .machine import decode
LOCK=threading.Lock();CACHE=collections.OrderedDict()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,value,mime='application/json',code=200):
  raw=value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  try:self.get()
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as e:
   try:self.send(dict(error=str(e)),code=400)
   except (BrokenPipeError,ConnectionResetError):pass
 def get(self):
  if self.headers.get('Host')!='127.0.0.1:8767':return self.send(dict(error='Use the loopback URL'),code=403)
  u=urlparse(self.path);q=parse_qs(u.query);tasks={t['task_id']:t for t in json.loads((REF/'tasks.json').read_text())}
  if u.path=='/':return self.send((ROOT/'src/track1_reference/frame_dashboard.html').read_bytes(),'text/html; charset=utf-8')
  if u.path=='/api/status':
   value=json.loads((STORE/'status.json').read_text()) if (STORE/'status.json').exists() else dict(state='starting',counts={})
   try:os.kill(int((ROOT/'work/track1_vlm/frame_review.pid').read_text()),0);value['worker_alive']=True
   except (OSError,ValueError):value['worker_alive']=False
   if not value['worker_alive'] and value['state'] in ['running','verifying_sources']:value['state']='worker_not_running'
   return self.send(value)
  if u.path=='/api/tasks':return self.send(list(tasks.values()))
  if u.path=='/favicon.ico':return self.send(b'','image/x-icon',204)
  task=tasks[q.get('task',[''])[0]];idx=int(q.get('frame',[task['query_index']])[0])
  if not 0<=idx<task['frame_count']:raise ValueError('Frame out of range')
  if u.path=='/api/image':
   key=(task['clip_id'],idx)
   with LOCK:
    if key not in CACHE:
     private=json.loads((WORK/'task_map.json').read_text());CACHE[key]=decode(private['clips'][task['clip_id']][idx])
    CACHE.move_to_end(key)
    while len(CACHE)>2:CACHE.popitem(last=False)
    raw=CACHE[key]
   return self.send(raw,'image/png')
  db=sqlite3.connect('file:'+str(STORE/'reviews.sqlite')+'?mode=ro',uri=True)
  try:
   if u.path=='/api/review':
    row=db.execute('SELECT value,flags,timestamp_us FROM reviews WHERE task_id=? AND idx=?',(task['task_id'],idx)).fetchone();shared=False
    if not row:
     for candidate in tasks.values():
      if candidate['task_id']!=task['task_id'] and group_key(candidate)==group_key(task):
       row=db.execute('SELECT value,flags,timestamp_us FROM reviews WHERE task_id=? AND idx=?',(candidate['task_id'],idx)).fetchone()
       if row:shared=True;break
    if row and json.loads(row[0]).get('semantic_group'):
     representative=next(t for t in tasks.values() if group_key(t)==group_key(task))
     shared=shared or representative['task_id']!=task['task_id']
    diagnostic=db.execute('SELECT value FROM frame_checks WHERE clip=? AND idx=?',(task['clip_id'],idx)).fetchone()
    planned=db.execute('SELECT status,error FROM review_plan WHERE group_id=? AND idx=?',(group_key(task),idx)).fetchone()
    attempted=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='attempts'").fetchone()
    attempt=db.execute('SELECT status,error FROM attempts WHERE task_id=? AND idx=?',(task['task_id'],idx)).fetchone() if attempted else None
    return self.send(dict(ready=bool(row),shared_cue=shared,frame_checked=bool(diagnostic),diagnostic=json.loads(diagnostic[0]) if diagnostic else None,group_id=group_key(task),attempt_status=('complete' if row else planned[0] if planned else 'not_selected'),error=planned[1] if planned else None,review=json.loads(row[0]) if row else None,flags=json.loads(row[1]) if row else [],timestamp_us=row[2] if row else None))
   if u.path=='/api/timeline':
    rows=db.execute('SELECT idx,value FROM reviews WHERE task_id=? ORDER BY idx',(task['task_id'],)).fetchall();events=db.execute('SELECT idx,kind,value FROM transitions WHERE task_id=? ORDER BY idx',(task['task_id'],)).fetchall();checks=db.execute('SELECT idx FROM frame_checks WHERE clip=? ORDER BY idx',(task['clip_id'],)).fetchall();return self.send(dict(checked_frames=[r[0] for r in checks],frames=[dict(index=i,visibility=json.loads(r)['visibility'],physical_contact=json.loads(r)['physical_contact']) for i,r in rows],transitions=[dict(index=i,kind=k,**json.loads(v)) for i,k,v in events]))
   return self.send(dict(error='Unknown route'),code=404)
  finally:db.close()
def main():
 server=ThreadingHTTPServer(('127.0.0.1',8767),Handler);server.daemon_threads=True;print('Qwen machine reviews: http://127.0.0.1:8767',flush=True)
 try:server.serve_forever()
 finally:server.server_close()
if __name__=='__main__':main()
