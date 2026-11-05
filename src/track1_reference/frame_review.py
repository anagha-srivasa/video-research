"""Resumable, native-frame Qwen machine reviews, isolated from human references."""
import argparse,base64,collections,hashlib,io,json,os,sqlite3,time,urllib.request,subprocess
from pathlib import Path
from PIL import Image
from . import ROOT,REF,WORK
from .machine import decode
from .machine_refine import crop
STORE=ROOT/'derived/track1_qwen_review';REPORT=ROOT/'reports/track1_vlm'
MODEL='track1-qwen'
SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SYSTEM='''You review raw egocentric RGB as a cautious machine reviewer. If depth/contact is unobservable, choose unknown promptly rather than repeatedly speculating. Review cases independently. You are NOT human ground truth. The class cues are unverified and may be wrong. Review ONLY the CENTER frame; neighbor images provide retrospective context. Distinguish a visible outer silhouette from an interaction border hidden by fingers. Never infer physical contact from image overlap, an occluded grip, or the class cue. If touching lacks directly visible evidence, physical_contact MUST be unknown. Do not hallucinate hidden boundaries, contents of containers, identities across occlusion, or event times. For each supplied task return one review. Output ONLY JSON: {"reviews":[{"task_id":"...","found":true,"description":"concise visible facts","visibility":"visible|partial|hidden|unknown","projected":"gap|meeting|overlap|unknown","physical_contact":"yes|no|unknown","evidence":"visible evidence for the judgment, focused on the cued pair","uncertainty":["explain every partial, hidden or ambiguous observation"],"attention_box":[left,top,right,bottom],"landmark":null,"landmark_definition":"","motion_observation":"quiet|moving|unknown","identity_evidence":"unknown|visible_distinctive_cue"}]}. Coordinates are normalized 0..1000; attention_box is an approximate VISIBLE interaction ROI, not a segmentation boundary, and may be null. Do not provide invented contour points. For motion tasks only, landmark is [x,y] of a clearly visible fixed physical point or null when hidden/ambiguous; use the supplied landmark_definition consistently if present. Never use a changing silhouette centroid. All point coordinates are approximate machine proposals, not certified pixel-accurate references. Motion is image-plane observation only. A contact task concerns hand-to-cued-object, NOT object-to-board/tray. Keep descriptions concise and evidence-specific.'''
CONFIG=dict(version=2,model=MODEL,installed_model='qwen/qwen3.8-27b@4bit',context_length=16384,temperature=0,max_tokens=8192,batch_tasks=2,all_native_frames=True,query_first=True,context_policy='retrospective native neighbors; center-only judgment; not causal forecast evidence',crop_policy='second pass on valid union ROI under 65% native area; original pixels only',human_reference_eligible=False,abstention_policy='preserve raw claims; uncertainty/conflicting passes abstain; hidden landmarks suppressed',prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest())

def connect():
 STORE.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(STORE/'reviews.sqlite',timeout=60);db.execute('PRAGMA journal_mode=WAL');db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS frames(clip TEXT,idx INTEGER,timestamp_us INTEGER,uid TEXT,image_sha256 TEXT,PRIMARY KEY(clip,idx)); CREATE TABLE IF NOT EXISTS reviews(task_id TEXT,clip TEXT,idx INTEGER,timestamp_us INTEGER,value TEXT,flags TEXT,batch_id TEXT,PRIMARY KEY(task_id,idx)); CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,clip TEXT,idx INTEGER,status TEXT,seconds REAL,response TEXT,error TEXT); CREATE TABLE IF NOT EXISTS attempts(task_id TEXT,idx INTEGER,status TEXT,batch_id TEXT,error TEXT,started REAL,PRIMARY KEY(task_id,idx)); CREATE TABLE IF NOT EXISTS transitions(task_id TEXT,idx INTEGER,kind TEXT,value TEXT,PRIMARY KEY(task_id,idx,kind));''');db.execute("INSERT OR IGNORE INTO attempts SELECT task_id,idx,'complete',batch_id,NULL,NULL FROM reviews");db.commit();return db

def validate(value,tasks):
 if not isinstance(value,dict) or not isinstance(value.get('reviews'),list):raise ValueError('Missing reviews array')
 rows=value['reviews'];expected={t['task_id'] for t in tasks}
 if len(rows)!=len(expected) or {r.get('task_id') for r in rows}!=expected:raise ValueError('Task identities missing, duplicated or invented')
 enums={'visibility':['visible','partial','hidden','unknown'],'projected':['gap','meeting','overlap','unknown'],'physical_contact':['yes','no','unknown'],'motion_observation':['quiet','moving','unknown'],'identity_evidence':['unknown','visible_distinctive_cue']}
 for r in rows:
  if type(r.get('found')) is not bool:raise ValueError('found must be boolean')
  for k,vs in enums.items():
   if r.get(k) not in vs:raise ValueError('Invalid '+k)
  for k in ['description','evidence','landmark_definition']:
   if not isinstance(r.get(k),str):raise ValueError('Missing text '+k)
  if not isinstance(r.get('uncertainty'),list) or not all(isinstance(s,str) for s in r['uncertainty']):raise ValueError('Invalid uncertainty list')
  box=r.get('attention_box')
  if box is not None and (not isinstance(box,list) or len(box)!=4 or any(type(x) not in [int,float] for x in box) or not 0<=box[0]<box[2]<=1000 or not 0<=box[1]<box[3]<=1000):raise ValueError('Invalid attention box')
  point=r.get('landmark')
  if point is not None and (not isinstance(point,list) or len(point)!=2 or any(type(x) not in [int,float] or not 0<=x<=1000 for x in point)):raise ValueError('Invalid landmark')
 return rows

class ReviewOutputError(ValueError):
 def __init__(self,message,response):super().__init__(message);self.response=response

def infer(images,tasks,extra=''):
 import psutil
 if psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Memory guard before inference')
 content=[dict(type='text',text='Review these tasks in the CENTER frame: '+json.dumps([{k:t.get(k,'') for k in ['task_id','kind','cue','landmark_definition']} for t in tasks])+extra)]
 for label,raw in images:content.extend([dict(type='text',text=label),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(raw).decode()))])
 body=dict(model=MODEL,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],temperature=0,max_tokens=CONFIG['max_tokens'],stream=False)
 req=urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1200) as r:response=json.load(r)
 choice=response['choices'][0]
 if choice.get('finish_reason')!='stop':raise ReviewOutputError('Truncated model output; not accepted',response)
 text=choice['message'].get('content','')
 try:value=json.loads(text[text.find('{'):text.rfind('}')+1]);rows=validate(value,tasks)
 except (ValueError,TypeError,KeyError) as exc:raise ReviewOutputError(str(exc),response) from exc
 return rows,response

def flags(row,first=None):
 out=['machine_only','approximate_localization_not_boundary_reference']
 if row['visibility']!='visible':out.append('interaction_edge_not_fully_visible')
 if row['visibility']!='visible' and not row['uncertainty']:out.append('missing_uncertainty_explanation')
 if row['physical_contact']!='unknown':out.append('physical_contact_claim_needs_verification')
 if not row['found']:out.append('target_not_established')
 if first and any(row[k]!=first[k] for k in ['found','visibility','projected','physical_contact']):out.append('full_view_closeup_disagreement')
 return out

def conservative(row,first=None):
 import copy,re
 out=copy.deepcopy(row);out['model_reported']={k:row[k] for k in ['visibility','projected','physical_contact','landmark']};checks=flags(row,first)
 text=' '.join(row['uncertainty']).lower()
 hidden=bool(re.search(r'partly occluded|partially occluded|covered by|occluded by|part.{0,35}covered|hidden by|not.{0,15}visible',text))
 if not row['found']:
  for key in ['visibility','projected','physical_contact']:out[key]='unknown'
  out['landmark']=None;checks.append('unestablished_target_abstained')
 if row['visibility']=='visible' and hidden and row['found']:
  out['visibility']='partial';checks.append('model_visibility_contradicts_its_uncertainty')
 if row['physical_contact']!='unknown' and (not row['found'] or out['visibility']!='visible' or row['uncertainty']):
  out['physical_contact']='unknown';checks.append('contact_claim_with_uncertainty_abstained')
 if first:
  for key in ['visibility','projected','physical_contact']:
   if row[key]!=first[key]:out[key]='unknown'
 if out['visibility'] in ['hidden','unknown'] and row['landmark'] is not None:
  out['landmark']=None;checks.append('uncertain_landmark_suppressed')
 out['policy']='Conservative machine abstention; original model fields and raw responses preserved'
 return out,checks

def schedule(tasks,private):
 groups=collections.OrderedDict()
 for t in tasks:
  if t['kind']=='contact':groups.setdefault((t['clip_id'],t['query_index']),[]).append(t)
  else:groups.setdefault((t['clip_id'],len(private['clips'][t['clip_id']])//2),[]).append(t)
 for (clip,idx),ts in groups.items():yield 'query',clip,idx,ts
 byclip=collections.OrderedDict()
 for t in tasks:byclip.setdefault(t['clip_id'],[]).append(t)
 for clip,ts in byclip.items():
  for idx in range(len(private['clips'][clip])):yield 'dense',clip,idx,ts

def transitions(db,task_id):
 rows=db.execute('SELECT idx,timestamp_us,value FROM reviews WHERE task_id=? ORDER BY idx',(task_id,)).fetchall();db.execute('DELETE FROM transitions WHERE task_id=?',(task_id,));previous=None
 for idx,ts,raw in rows:
  r=json.loads(raw)
  if previous and idx==previous[0]+1:
   p=previous[2]
   for field in ['projected','physical_contact','motion_observation']:
    if p[field]!=r[field] and 'unknown' not in [p[field],r[field]]:
     value=dict(from_value=p[field],to_value=r[field],earliest_us=previous[1],latest_us=ts,machine_candidate_only=True,physical_event_certified=False)
     db.execute('INSERT INTO transitions VALUES(?,?,?,?)',(task_id,idx,field,json.dumps(value)))
  previous=(idx,ts,r)

def status(db,state,phase='',error=None):
 counts={k:db.execute('SELECT count(*) FROM '+k).fetchone()[0] for k in ['frames','reviews','batches','transitions','attempts']};failed=db.execute("SELECT count(*) FROM batches WHERE status='failed'").fetchone()[0]
 value=dict(state=state,phase=phase,updated_unix=time.time(),counts=counts,failed_batches=failed,expected_unique_native_frames=6480,expected_case_frames=58800,model=CONFIG['installed_model'],human_reference_eligible=False,scientific_acceptance=False,track2_started=False,error=error)
 temp=STORE/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(STORE/'status.json')

def verify_runtime():
 models=json.loads(subprocess.check_output(['lms','ps','--json']))
 found=next((m for m in models if m.get('identifier')==MODEL),None)
 if not found or found.get('modelKey')!='qwen/qwen3.8-27b' or not found.get('vision') or found.get('contextLength',0)<CONFIG['context_length']:
  raise RuntimeError('Load the selected Qwen vision model with identifier track1-qwen and 16384 context before resuming')
 return found

def run(max_batches=0,retry_failed=False):
 import psutil,pyarrow.parquet as pq
 from .protocol import digest
 import fcntl
 STORE.mkdir(parents=True,exist_ok=True);lock=(STORE/'worker.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('A frame reviewer is already running')
 runtime=verify_runtime();(STORE/'runtime.json').write_text(json.dumps(runtime,indent=2));db=connect();encoded=json.dumps(CONFIG,sort_keys=True);old=db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
 if old and old[0]!=encoded:raise RuntimeError('Configuration changed; use a distinct review release, not mixed results')
 db.execute("INSERT OR IGNORE INTO meta VALUES('config',?)",(encoded,));db.commit()
 if retry_failed:
  db.execute('CREATE TABLE IF NOT EXISTS failed_history AS SELECT *, 0 AS archived_unix FROM batches WHERE 0')
  db.execute("INSERT INTO failed_history SELECT *,? FROM batches WHERE status='failed'",(time.time(),));db.execute("DELETE FROM batches WHERE status='failed'");db.commit()
 (STORE/'config.json').write_text(json.dumps(CONFIG,indent=2));tasks=json.loads((REF/'tasks.json').read_text());private=json.loads((WORK/'task_map.json').read_text());status(db,'verifying_sources');cache=collections.OrderedDict();done=0
 try:
  provenance=dict(tasks_sha256=digest(REF/'tasks.json'),task_map_sha256=digest(WORK/'task_map.json'),source_sha256={})
  for video in sorted({private['tasks'][t['task_id']]['video_id'] for t in tasks}):
   expected=pq.read_table(ROOT/f'canonical/v1/videos.{video}.parquet',columns=['sha256']).to_pylist()[0]['sha256'];print('Checking original '+video,flush=True)
   if digest(ROOT/f'dataset/videos/{video}.MP4')!=expected:raise RuntimeError('Source checksum mismatch '+video)
   provenance['source_sha256'][video]=expected
  encoded_provenance=json.dumps(provenance,sort_keys=True);old_provenance=db.execute("SELECT value FROM meta WHERE key='provenance'").fetchone()
  if old_provenance and old_provenance[0]!=encoded_provenance:raise RuntimeError('Source/task provenance changed; refusing mixed review data')
  db.execute("INSERT OR IGNORE INTO meta VALUES('provenance',?)",(encoded_provenance,));db.commit();(STORE/'provenance.json').write_text(json.dumps(provenance,indent=2))
  def image_at(clip,idx):
   key=(clip,idx)
   if key not in cache:cache[key]=decode(private['clips'][clip][idx])
   cache.move_to_end(key)
   while len(cache)>4:cache.popitem(last=False)
   return cache[key]
  for phase,clip,idx,ts in schedule(tasks,private):
   remaining=[dict(t) for t in ts if not db.execute('SELECT 1 FROM reviews WHERE task_id=? AND idx=?',(t['task_id'],idx)).fetchone()]
   if not remaining:continue
   for offset in range(0,len(remaining),CONFIG['batch_tasks']):
    chunk=remaining[offset:offset+CONFIG['batch_tasks']];batch_id=hashlib.sha256((clip+':'+str(idx)+':'+','.join(t['task_id'] for t in chunk)+encoded).encode()).hexdigest()[:24]
    if db.execute("SELECT 1 FROM batches WHERE id=? AND status='failed'",(batch_id,)).fetchone():continue
    if psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Memory guard: less than 6 GiB available; safe to resume later')
    if psutil.disk_usage(STORE).free<12*1024**3:raise RuntimeError('Disk guard: less than 12 GiB free')
    started=time.time();fs=private['clips'][clip];indices=[max(0,idx-3),idx,min(len(fs)-1,idx+3)]
    for task in chunk:db.execute('INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?)',(task['task_id'],idx,'processing',batch_id,None,started))
    db.commit()
    try:images=[(f'{label} native frame {i} timestamp {fs[i]["timestamp_us"]}us',image_at(clip,i)) for label,i in zip(['BEFORE','CENTER','AFTER'],indices)]
    except Exception as exc:
     for task in chunk:db.execute('UPDATE attempts SET status=?,error=? WHERE task_id=? AND idx=?',('failed',str(exc),task['task_id'],idx))
     db.execute('INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,?)',(batch_id,clip,idx,'failed',time.time()-started,None,'Exact native decode failed: '+str(exc)));db.commit();status(db,'running',phase);print('  DECODE FAILED CLOSED: '+str(exc),flush=True);continue
    print(f'{phase} clip={clip} frame={idx}/{len(fs)-1} tasks={len(chunk)} batch={batch_id}',flush=True);status(db,'running',phase)
    with (ROOT/'work/track1_vlm/frame_resources.jsonl').open('a') as log:log.write(json.dumps(dict(time=started,available_gib=psutil.virtual_memory().available/2**30,worker_rss_gib=psutil.Process().memory_info().rss/2**30,batch=batch_id))+'\n')
    for task in chunk:
     if task['kind']=='motion_identity':
      anchor=db.execute('SELECT value FROM reviews WHERE task_id=? ORDER BY idx LIMIT 1',(task['task_id'],)).fetchone()
      if anchor:task['landmark_definition']=json.loads(anchor[0]).get('landmark_definition','')
    try:
     first,response=infer(images,chunk);final=first;close_response=None;bounds=None;boxes=[r['attention_box'] for r in first if r.get('attention_box')]
     if boxes:
      union=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
      cut=crop(images[1][1],union)
      if cut and (cut[1][2]-cut[1][0])*(cut[1][3]-cut[1][1])<1920*1080*.65:
       zoom,bounds=cut;final,close_response=infer([images[1],('CENTER attention crop of the same original pixels; not a boundary; could be mislocalized',zoom)],chunk,' Recheck the interaction using full frame and close-up. If the crop misses the targets, abstain. Never infer a covered border.')
     f=fs[idx];uid=f.get('native_frame_uid','');db.execute('INSERT OR REPLACE INTO frames VALUES(?,?,?,?,?)',(clip,idx,f['timestamp_us'],uid,hashlib.sha256(images[1][1]).hexdigest()))
     byid={r['task_id']:r for r in first}
     for reported in final:
      row,checks=conservative(reported,byid[reported['task_id']])
      row['frame_indices_used']=indices;row['attention_crop_native']=bounds;row['coordinates_quality']='unvalidated_machine_estimate';row['boundaries_available']=False
      db.execute('INSERT OR REPLACE INTO reviews VALUES(?,?,?,?,?,?,?)',(row['task_id'],clip,idx,f['timestamp_us'],json.dumps(row),json.dumps(checks),batch_id))
     for task in chunk:db.execute("UPDATE attempts SET status='complete',error=NULL WHERE task_id=? AND idx=?",(task['task_id'],idx))
     record=dict(first=response,closeup=close_response,attention_crop_native=bounds,worker_source_sha256=SOURCE_SHA256)
     db.execute('INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,?)',(batch_id,clip,idx,'complete',time.time()-started,json.dumps(record),None));db.commit()
     for task in chunk:transitions(db,task['task_id'])
     db.commit();done+=1;print(f'  committed {time.time()-started:.1f}s; closeup={bounds is not None}',flush=True)
    except Exception as exc:
     db.rollback()
     for task in chunk:db.execute("UPDATE attempts SET status='failed',error=? WHERE task_id=? AND idx=?",(str(exc),task['task_id'],idx))
     db.execute('INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,?)',(batch_id,clip,idx,'failed',time.time()-started,json.dumps(getattr(exc,'response',None)),str(exc)));db.commit();print('  FAILED CLOSED: '+str(exc),flush=True)
    status(db,'running',phase)
    if max_batches and done>=max_batches:status(db,'pilot_paused',phase);return
  failed=db.execute("SELECT count(*) FROM batches WHERE status='failed'").fetchone()[0];status(db,'completed_with_failures' if failed else 'complete')
 except KeyboardInterrupt:status(db,'interrupted');raise
 except Exception as exc:status(db,'stopped',error=str(exc));raise
 finally:db.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--max-batches',type=int,default=0);p.add_argument('--retry-failed',action='store_true');a=p.parse_args();run(a.max_batches,a.retry_failed)
