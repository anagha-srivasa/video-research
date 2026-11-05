"""Native-frame QC plus selected Qwen evidence. Sparse semantics stay explicitly sparse."""
import argparse,base64,collections,concurrent.futures,hashlib,io,json,multiprocessing,sqlite3,time,urllib.request
from pathlib import Path
from . import ROOT,REF,WORK
from . import frame_review as legacy
from .machine import decode
from .machine_refine import crop
STORE=ROOT/'derived/track1_qwen_review_v3'
SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SYSTEM='''Review one raw egocentric RGB case carefully. The class cue is unverified; multiple matching objects mean ambiguity, not permission to guess the source identity. Judge only the CENTER frame; labeled neighbor images supply retrospective context. A visible outer silhouette does not establish a visible interaction border. Do not infer touching from image overlap or an occluded grip. Physical contact is unknown without directly visible evidence. Keep uncertain physical interpretations unknown rather than repeatedly speculating about depth. Never invent hidden outlines, container contents, identity across occlusion, velocities or event times. Return one JSON object only, with these fields: found (boolean), description (concise visible facts), visibility (visible/partial/hidden/unknown for the interaction-side border), projected (gap/meeting/overlap/unknown), physical_contact (yes/no/unknown), evidence (specific visible evidence), uncertainty (array of strings explaining every ambiguity), attention_box ([left,top,right,bottom] normalized to 0..1000 or null), landmark ([x,y] normalized to 0..1000 or null), landmark_definition (string), motion_observation (quiet/moving/unknown for image-plane change only), identity_evidence (unknown/visible_distinctive_cue). Coordinates are approximate machine proposals, never pixel-accurate contours. Only motion tasks need a clearly visible fixed physical landmark; use a supplied landmark definition consistently and leave hidden/ambiguous landmarks null. Never substitute a moving silhouette centroid. Contact tasks concern the hand and cued object, not an object's contact with the board or tray. Do not output task IDs; the application binds this single request to its case.'''
CONFIG=dict(version=3,model='qwen/qwen3.8-27b@4bit',temperature=0,context_length=16384,max_tokens=8192,batch_cases=1,prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),response_contract='single-case locally validated JSON; MLX constrained-output routing failed pilot',risk_gap_merge_us=100_000,semantic_anchor_interval_us=1_000_000,all_native_frames_checked=True,labels_propagated=False,independent_accuracy_validated=False,selection_scope='query targets, one-second sentinels, motion/retention/light-change episodes; unresolved intervals retained',crop_policy='small/low-resolution visible targets or a decisive contact claim; never try to reveal hidden geometry')

def group_key(t):return hashlib.sha256(json.dumps([t['clip_id'],t['kind'],t['cue']]).encode()).hexdigest()[:20]
def load():
 tasks=json.loads((REF/'tasks.json').read_text());private=json.loads((WORK/'task_map.json').read_text());groups=collections.defaultdict(list)
 for t in tasks:groups[group_key(t)].append(t)
 return tasks,private,groups

def connect():
 STORE.mkdir(parents=True,exist_ok=True);path=STORE/'reviews.sqlite'
 if not path.exists():
  manifest=json.loads((ROOT/'work/track1_vlm/preserved_manifest.json').read_text());source=ROOT/manifest['backup']
  if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['sha256']:raise RuntimeError('Preserved ledger hash mismatch')
  src=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True);db=sqlite3.connect(path);src.backup(db);src.close();db.close()
 db=sqlite3.connect(path,timeout=60);db.execute('PRAGMA journal_mode=WAL');db.executescript('''CREATE TABLE IF NOT EXISTS frame_checks(clip TEXT,idx INTEGER,timestamp_us INTEGER,uid TEXT,value TEXT,PRIMARY KEY(clip,idx));CREATE TABLE IF NOT EXISTS clip_checks(clip TEXT PRIMARY KEY,status TEXT,seconds REAL,error TEXT);CREATE TABLE IF NOT EXISTS review_plan(group_id TEXT,clip TEXT,idx INTEGER,reason TEXT,status TEXT,error TEXT,PRIMARY KEY(group_id,idx));CREATE TABLE IF NOT EXISTS adaptive_responses(id INTEGER PRIMARY KEY,group_id TEXT,idx INTEGER,pass TEXT,seconds REAL,value TEXT,error TEXT);CREATE TABLE IF NOT EXISTS landmark_seeds(group_id TEXT PRIMARY KEY,definition TEXT,source_task TEXT,source_idx INTEGER);CREATE TABLE IF NOT EXISTS adaptive_meta(key TEXT PRIMARY KEY,value TEXT);''')
 encoded=json.dumps(CONFIG,sort_keys=True);old=db.execute("SELECT value FROM adaptive_meta WHERE key='config'").fetchone()
 if old and old[0]!=encoded:raise RuntimeError('Adaptive configuration changed; refusing mixed releases')
 db.execute("INSERT OR IGNORE INTO adaptive_meta VALUES('config',?)",(encoded,));db.commit();return db

def status(db,state,phase,error=None):
 checked=db.execute('SELECT count(*) FROM frame_checks').fetchone()[0];direct=db.execute('SELECT count(*) FROM reviews').fetchone()[0];plan=dict(db.execute('SELECT status,count(*) FROM review_plan GROUP BY status').fetchall());failures=db.execute("SELECT count(*) FROM adaptive_responses WHERE error IS NOT NULL").fetchone()[0]
 s=dict(state=state,phase=phase,updated_unix=time.time(),workflow='adaptive_v3',model=CONFIG['model'],counts=dict(frames=db.execute('SELECT count(*) FROM frames').fetchone()[0],reviews=direct,batches=db.execute('SELECT count(*) FROM batches').fetchone()[0],frame_checks=checked),expected_unique_native_frames=6480,expected_case_frames=58800,planned_semantic_jobs=plan,failed_batches=failures,legacy_failed_batches=db.execute("SELECT count(*) FROM batches WHERE status='failed'").fetchone()[0],unreviewed_case_frames=58800-direct,independent_accuracy_validated=False,human_reference_eligible=False,scientific_acceptance=False,track2_started=False,error=error,progress_definition='Frame QC is not direct semantic review. Unreviewed labels remain absent.')
 temp=STORE/'status.tmp';temp.write_text(json.dumps(s,indent=2));temp.replace(STORE/'status.json')

def source_check(db,private,tasks):
 import pyarrow.parquet as pq
 from .protocol import digest
 hashes={}
 for v in sorted({private['tasks'][t['task_id']]['video_id'] for t in tasks}):
  expected=pq.read_table(ROOT/f'canonical/v1/videos.{v}.parquet',columns=['sha256']).to_pylist()[0]['sha256'];print('Checking original '+v,flush=True)
  if digest(ROOT/f'dataset/videos/{v}.MP4')!=expected:raise RuntimeError('Source checksum mismatch '+v)
  hashes[v]=expected
 p=dict(sources=hashes,tasks_sha256=digest(REF/'tasks.json'),map_sha256=digest(WORK/'task_map.json'));encoded=json.dumps(p,sort_keys=True);old=db.execute("SELECT value FROM adaptive_meta WHERE key='provenance'").fetchone()
 if old and old[0]!=encoded:raise RuntimeError('Adaptive source provenance changed')
 db.execute("INSERT OR IGNORE INTO adaptive_meta VALUES('provenance',?)",(encoded,));db.commit();(STORE/'config.json').write_text(json.dumps(CONFIG,indent=2));(STORE/'provenance.json').write_text(json.dumps(p,indent=2))

def anchors(db,groups,clip):
 bytask={t['task_id']:g for g,ts in groups.items() for t in ts};result=collections.defaultdict(dict)
 for tid,idx,raw in db.execute('SELECT task_id,idx,value FROM reviews WHERE clip=? ORDER BY idx',(clip,)):
  r=json.loads(raw);b=r.get('attention_box')
  if r.get('found') and b and len(b)==4 and 0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000:result[idx][bytask[tid]]=[b[0]*1.92,b[1]*1.08,b[2]*1.92,b[3]*1.08]
 return {i:list(v.items()) for i,v in result.items()}

def scan_all(db,private,groups):
 import psutil
 from .frame_checks import scan
 clips=list(dict.fromkeys(t['clip_id'] for ts in groups.values() for t in ts));pending=[c for c in clips if not db.execute("SELECT 1 FROM clip_checks WHERE clip=? AND status='complete'",(c,)).fetchone()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  # Only two clips resident/in flight; no RGB file copies.
  jobs=iter(pending);active={}
  while True:
   while len(active)<2:
    try:c=next(jobs)
    except StopIteration:break
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Memory guard before native frame scan')
    active[pool.submit(scan,c,anchors(db,groups,c))]=c
   if not active:break
   done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
   for job in done:
    clip=active.pop(job)
    try:
     _,rows,seconds=job.result()
     if len(rows)!=len(private['clips'][clip]):raise RuntimeError('Incomplete native scan')
     for idx,value in rows:
      f=private['clips'][clip][idx];db.execute('INSERT OR REPLACE INTO frame_checks VALUES(?,?,?,?,?)',(clip,idx,f['timestamp_us'],f['native_frame_uid'],json.dumps(value)))
     db.execute('INSERT OR REPLACE INTO clip_checks VALUES(?,?,?,NULL)',(clip,'complete',seconds));print(f'Checked {clip}: {len(rows)} native frames in {seconds:.1f}s',flush=True)
    except Exception as e:db.execute('INSERT OR REPLACE INTO clip_checks VALUES(?,?,0,?)',(clip,'failed',str(e)));print('Native scan failed: '+str(e),flush=True)
    db.commit();status(db,'running','native_frame_checks')

def pick_frames(fs,metrics,query_indices):
 selected={i:{'requested_query'} for i in query_indices};origin=fs[0]['timestamp_us'];next_us=origin
 for i,f in enumerate(fs):
  if f['timestamp_us']>=next_us:selected.setdefault(i,set()).add('one_second_sentinel');next_us=f['timestamp_us']+CONFIG['semantic_anchor_interval_us']
 selected.setdefault(len(fs)-1,set()).add('context_end')
 # Each contiguous diagnostic episode gets its strongest representative and both boundaries.
 episode=[];last_risk=None
 def flush():
  if not episode:return
  worst=max(episode,key=lambda i:metrics[i].get('p95_flow_px') or 0)
  for i in {episode[0],worst,episode[-1]}:selected.setdefault(i,set()).add('motion_or_support_episode')
  episode.clear()
 for i,m in enumerate(metrics):
  risk=i>0 and (not m.get('flow_valid') or (m.get('p95_flow_px') or 0)>18 or m.get('mean_abs_luma_change',0)>18)
  if risk:
   if last_risk is not None and fs[i]['timestamp_us']-fs[last_risk]['timestamp_us']>CONFIG['risk_gap_merge_us']:flush()
   episode.append(i);last_risk=i
  elif episode and fs[i]['timestamp_us']-fs[episode[-1]]['timestamp_us']>CONFIG['risk_gap_merge_us']:flush()
 flush();return selected

def plan(db,private,groups):
 for group,ts in groups.items():
  clip=ts[0]['clip_id'];fs=private['clips'][clip];rows=db.execute('SELECT idx,value FROM frame_checks WHERE clip=? ORDER BY idx',(clip,)).fetchall()
  if len(rows)!=len(fs):continue
  picks=pick_frames(fs,[json.loads(r[1]) for r in rows],[t['query_index'] for t in ts]);members={t['task_id'] for t in ts}
  for idx,reasons in picks.items():
   existing={r[0] for r in db.execute('SELECT task_id FROM reviews WHERE clip=? AND idx=?',(clip,idx))};state='preserved' if members&existing else 'pending'
   db.execute('INSERT OR IGNORE INTO review_plan VALUES(?,?,?,?,?,NULL)',(group,clip,idx,json.dumps(sorted(reasons)),state))
 db.commit()

def parse(text,task):
 start=text.find('{')
 if start<0:raise ValueError('No final JSON object; reasoning-only output is not accepted')
 value,end=json.JSONDecoder().raw_decode(text[start:]);tail=text[start+end:].strip();notes=[]
 if tail and any(ch not in '} ]`\r\n\t' for ch in tail):raise ValueError('Unexpected extra response content')
 if tail:notes.append('discarded trailing fence/closing punctuation; parsed object unchanged')
 if isinstance(value,dict) and 'reviews' in value:
  if not isinstance(value['reviews'],list) or len(value['reviews'])!=1:raise ValueError('Expected exactly one case')
  value=value['reviews'][0]
 if not isinstance(value,dict):raise ValueError('Expected review object')
 if 'task_id' in value and value['task_id']!=task['task_id']:raise ValueError('Unexpected model case identity')
 r=dict(value,task_id=task['task_id'])
 if isinstance(r.get('uncertainty'),str):r['uncertainty']=[r['uncertainty']] if r['uncertainty'] else [];notes.append('single uncertainty string retained as one list item')
 if not isinstance(r.get('uncertainty'),list) or not all(isinstance(x,str) for x in r['uncertainty']):raise ValueError('Invalid uncertainty field')
 b=r.get('attention_box')
 if b is not None and (not isinstance(b,list) or len(b)!=4 or any(type(x) not in [int,float] for x in b) or not 0<=b[0]<b[2]<=1000 or not 0<=b[1]<b[3]<=1000):
  r['attention_box']=None;r['uncertainty'].append('Model attention coordinates invalid; localization unavailable.');notes.append('invalid coordinates removed; no scale or geometry guessed')
 r['format_normalizations']=notes
 return legacy.validate(dict(reviews=[r]),[task])[0]

def infer(images,task,extra=''):
 import psutil
 if psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Memory guard before Qwen inference')
 content=[]
 # Stable image evidence precedes the varying cue, enabling exact-prefix reuse.
 for label,raw in images:content.extend([dict(type='text',text=label),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(raw).decode()))])
 content.append(dict(type='text',text='Review only this task in the CENTER frame: '+json.dumps({k:task.get(k,'') for k in ['kind','cue','landmark_definition']})+extra))
 body=dict(model=legacy.MODEL,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],temperature=0,max_tokens=8192,stream=False)
 started=time.monotonic();req=urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1200) as r:response=json.load(r)
 choice=response['choices'][0]
 if choice.get('finish_reason')!='stop':raise legacy.ReviewOutputError('Incomplete structured response',response)
 text=choice['message'].get('content','')
 try:
  result=parse(text,task)
 except (ValueError,TypeError,KeyError) as e:raise legacy.ReviewOutputError(str(e),response) from e
 return result,response,time.monotonic()-started

def wants_closeup(row):
 box=row.get('attention_box')
 if not box or not row['found'] or row['visibility'] in ['hidden','unknown']:return False
 conservative,_=legacy.conservative(row)
 area=(box[2]-box[0])*(box[3]-box[1])/1e6
 return conservative['physical_contact']!='unknown' or (area<.025 and row['visibility'] in ['visible','partial'])

def review_job(db,private,group,idx,ts):
 clip=ts[0]['clip_id'];fs=private['clips'][clip];task=dict(ts[0])
 if task['kind']=='motion_identity':
  seeded=db.execute('SELECT definition FROM landmark_seeds WHERE group_id=?',(group,)).fetchone()
  if not seeded:
   for source_idx,raw in db.execute('SELECT idx,value FROM reviews WHERE task_id=? ORDER BY idx',(task['task_id'],)):
    definition=json.loads(raw).get('landmark_definition','')
    if definition:
     db.execute('INSERT OR IGNORE INTO landmark_seeds VALUES(?,?,?,?)',(group,definition,task['task_id'],source_idx));db.commit();seeded=(definition,);break
  if seeded:task['landmark_definition']=seeded[0]
 indices=[max(0,idx-3),idx,min(len(fs)-1,idx+3)];images=[(f'{name} native frame {i} timestamp {fs[i]["timestamp_us"]}us',decode(fs[i])) for name,i in zip(['BEFORE','CENTER','AFTER'],indices)];started=time.time();batch='v3-'+group+'-'+str(idx)
 first=None;last=None;response=None;close=None;crop_bounds=None
 try:
  first,response,secs=infer(images,task);db.execute('INSERT INTO adaptive_responses(group_id,idx,pass,seconds,value,error) VALUES(?,?,?,?,?,NULL)',(group,idx,'full',secs,json.dumps(response)));db.commit();last=first
  if wants_closeup(first):
   cut=crop(images[1][1],first['attention_box'])
   if cut and (cut[1][2]-cut[1][0])*(cut[1][3]-cut[1][1])<1920*1080*.65:
    zoom,crop_bounds=cut;last,close,secs=infer([images[1],('CENTER close-up; original pixels, no enhancement',zoom)],task,' Resolve only visible small/unclear interface evidence. Covered physical surfaces remain unknown.');db.execute('INSERT INTO adaptive_responses(group_id,idx,pass,seconds,value,error) VALUES(?,?,?,?,?,NULL)',(group,idx,'closeup',secs,json.dumps(close)));db.commit()
  r,checks=legacy.conservative(last,first if close else None)
  if r['physical_contact']!='unknown' and close is None:
   r['physical_contact']='unknown';r['uncertainty'].append('Definitive contact claim lacked a valid second view; not accepted.');checks.append('definitive_contact_not_rechecked')
  if task['kind']=='motion_identity' and task.get('landmark_definition') and r['landmark_definition']!=task['landmark_definition']:
   r['landmark']=None;r['uncertainty'].append('Landmark definition changed; correspondence not accepted.');checks.append('landmark_definition_mismatch')
  elif task['kind']=='motion_identity' and r['landmark'] is not None and r['landmark_definition']:
   db.execute('INSERT OR IGNORE INTO landmark_seeds VALUES(?,?,?,?)',(group,r['landmark_definition'],task['task_id'],idx))
  r.update(coordinates_quality='unvalidated_machine_estimate',boundaries_available=False,frame_indices_used=indices,attention_crop_native=crop_bounds,semantic_group=group,labels_propagated=False,shared_cue_is_not_identity=True)
  f=fs[idx];db.execute('INSERT OR IGNORE INTO frames VALUES(?,?,?,?,?)',(clip,idx,f['timestamp_us'],f['native_frame_uid'],hashlib.sha256(images[1][1]).hexdigest()))
  for member in ts:
   row=dict(r,task_id=member['task_id']);db.execute('INSERT OR IGNORE INTO reviews VALUES(?,?,?,?,?,?,?)',(member['task_id'],clip,idx,f['timestamp_us'],json.dumps(row),json.dumps(checks+['shared_cue_not_verified_source_instance']),batch))
  db.execute('INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,NULL)',(batch,clip,idx,'complete',time.time()-started,json.dumps(dict(first=response,closeup=close,attention_crop_native=crop_bounds,input_sha256=[hashlib.sha256(b).hexdigest() for _,b in images],worker_source_sha256=SOURCE_SHA256))));db.execute("UPDATE review_plan SET status='complete',error=NULL WHERE group_id=? AND idx=?",(group,idx));db.commit();print(f'Committed selected Qwen group {group} frame {idx}: {time.time()-started:.1f}s; closeup={close is not None}',flush=True);return True
 except Exception as exc:
  db.rollback();db.execute('INSERT INTO adaptive_responses(group_id,idx,pass,seconds,value,error) VALUES(?,?,?,?,?,?)',(group,idx,'failed',time.time()-started,json.dumps(getattr(exc,'response',None)),str(exc)));db.execute("UPDATE review_plan SET status='failed',error=? WHERE group_id=? AND idx=?",(str(exc),group,idx));db.commit();print('Selected review failed closed: '+str(exc),flush=True);return False

def refresh_tracking(db,private,groups,clip):
 from .frame_checks import scan
 _,rows,seconds=scan(clip,anchors(db,groups,clip))
 if len(rows)!=len(private['clips'][clip]):raise RuntimeError('Incomplete tracking refresh')
 for idx,updated in rows:
  old=db.execute('SELECT value FROM frame_checks WHERE clip=? AND idx=?',(clip,idx)).fetchone()
  if not old:raise RuntimeError('Missing checked frame during refresh')
  metrics=json.loads(old[0])
  if metrics['gray_sha256']!=updated['gray_sha256']:raise RuntimeError('Source pixels changed during tracking refresh')
  metrics['attention_tracks']=updated['attention_tracks'];db.execute('UPDATE frame_checks SET value=? WHERE clip=? AND idx=?',(json.dumps(metrics),clip,idx))
 db.commit();print(f'Refreshed attention-region diagnostics {clip}: {seconds:.1f}s; semantic labels not propagated',flush=True)

def run(stage='all',limit=0):
 import fcntl,psutil
 STORE.mkdir(parents=True,exist_ok=True);lock=(STORE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=connect();tasks,private,groups=load()
 try:
  status(db,'running','source_verification');source_check(db,private,tasks)
  if stage in ['all','frames']:status(db,'running','native_frame_checks');scan_all(db,private,groups);plan(db,private,groups)
  if stage in ['all','frames'] and db.execute('SELECT count(*) FROM frame_checks').fetchone()[0]!=sum(len(private['clips'][c]) for c in {t['clip_id'] for t in tasks}):raise RuntimeError('Native frame scan incomplete; refusing coverage completion')
  if stage=='frames':status(db,'frames_checked','awaiting_selected_reviews');return
  legacy.verify_runtime();jobs=db.execute("SELECT group_id,idx FROM review_plan WHERE status='pending' ORDER BY clip,idx,group_id").fetchall();status(db,'running','selected_qwen_reviews');last_clip=None
  for number,(group,idx) in enumerate(jobs):
   if limit and number>=limit:break
   clip=groups[group][0]['clip_id']
   if last_clip is not None and clip!=last_clip:refresh_tracking(db,private,groups,last_clip)
   last_clip=clip
   if psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Memory guard before selected review')
   print(f'Selected job {number+1}/{len(jobs)} group={group} frame={idx}',flush=True)
   for retry in range(2):
    if review_job(db,private,group,idx,groups[group]):break
    if psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Memory guard after failed review')
   status(db,'running','selected_qwen_reviews')
  if last_clip is not None:refresh_tracking(db,private,groups,last_clip)
  failed=db.execute("SELECT count(*) FROM review_plan WHERE status='failed'").fetchone()[0];status(db,('selected_pass_with_unresolved' if failed else 'selected_pass_complete') if not limit else 'pilot_complete','selected_qwen_reviews')
 except KeyboardInterrupt:status(db,'interrupted','');raise
 except Exception as e:status(db,'stopped','',str(e));raise
 finally:db.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','frames','review'],default='all');p.add_argument('--limit',type=int,default=0);a=p.parse_args();run(a.stage,a.limit)
