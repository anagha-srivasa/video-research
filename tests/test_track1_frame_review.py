import pytest
from track1_reference.frame_review import validate,flags,schedule

def row():return dict(task_id='a',found=True,description='Visible hand',visibility='partial',projected='overlap',physical_contact='unknown',evidence='Fingers cover the interface',uncertainty=['occluded'],attention_box=[100,100,600,600],landmark=None,landmark_definition='',motion_observation='unknown',identity_evidence='unknown')
def test_frame_review_rejects_wrong_identity_and_invented_coordinates():
 tasks=[dict(task_id='a')];r=row();assert validate(dict(reviews=[r]),tasks)==[r]
 with pytest.raises(ValueError):validate(dict(reviews=[r,r]),tasks)
 with pytest.raises(ValueError):validate(dict(reviews=[r]),[dict(task_id='b')])
 r['landmark']=[float('nan'),5]
 with pytest.raises(ValueError):validate(dict(reviews=[r]),tasks)
def test_closeup_conflicts_remain_explicit():
 first=row();final=row();final['projected']='gap';assert 'full_view_closeup_disagreement' in flags(final,first)
 assert 'physical_contact_claim_needs_verification' not in flags(final)
 final['physical_contact']='yes';assert 'physical_contact_claim_needs_verification' in flags(final)
def test_schedule_covers_every_case_frame_and_query_first():
 tasks=[dict(task_id='a',kind='contact',clip_id='c',query_index=1),dict(task_id='b',kind='motion_identity',clip_id='c',query_index=0)]
 jobs=list(schedule(tasks,dict(clips=dict(c=[{}, {}, {}]))));assert jobs[0][0]=='query'
 dense={(t['task_id'],i) for phase,c,i,ts in jobs if phase=='dense' for t in ts};assert dense=={(t,i) for t in ['a','b'] for i in range(3)}

def test_frame_worker_commits_and_resumes_without_duplicate_reviews(tmp_path,monkeypatch):
 import hashlib,io,json
 import pyarrow.parquet as pq
 from PIL import Image
 import track1_reference.frame_review as f
 root=tmp_path;ref=root/'human';work=root/'work';store=root/'derived'
 ref.mkdir();work.mkdir();(root/'dataset/videos').mkdir(parents=True)
 source=root/'dataset/videos/P01_01.MP4';source.write_bytes(b'fixture source');sha=hashlib.sha256(source.read_bytes()).hexdigest()
 tasks=[dict(task_id='a',kind='contact',clip_id='c',query_index=1,frame_count=2,cue='hand / cup')]
 (ref/'tasks.json').write_text(json.dumps(tasks));private=dict(tasks=dict(a=dict(video_id='P01_01')),clips=dict(c=[dict(timestamp_us=i*20000,video_id='P01_01',pts_raw=i,native_frame_uid=str(i)) for i in range(2)]));(work/'task_map.json').write_text(json.dumps(private))
 for k,v in [('ROOT',root),('REF',ref),('WORK',work),('STORE',store)]:monkeypatch.setattr(f,k,v)
 class Table:
  def to_pylist(self):return [dict(sha256=sha)]
 monkeypatch.setattr(pq,'read_table',lambda *a,**kw:Table())
 buf=io.BytesIO();Image.new('RGB',(1920,1080)).save(buf,format='PNG');monkeypatch.setattr(f,'decode',lambda frame:buf.getvalue());calls=[]
 def infer(images,ts,extra=''):
  calls.append(ts);r=row();r['attention_box']=None
  return [r],dict(choices=[dict(finish_reason='stop')])
 monkeypatch.setattr(f,'infer',infer);monkeypatch.setattr(f,'verify_runtime',lambda:dict(fixture=True))
 (work/'track1_vlm').mkdir()
 f.run(max_batches=1);assert len(calls)==1
 f.run();assert len(calls)==2
 db=f.connect();assert db.execute('SELECT count(*) FROM reviews').fetchone()[0]==2;assert db.execute('SELECT count(*) FROM batches').fetchone()[0]==2;db.close()
 assert json.loads((ref/'tasks.json').read_text())==tasks
 db=f.connect();assert db.execute("SELECT count(*) FROM attempts WHERE status='complete'").fetchone()[0]==2
 db.execute('DELETE FROM reviews');db.execute('DELETE FROM batches');db.commit();db.close()
 def reject(*args,**kwargs):raise f.ReviewOutputError('truncated',dict(choices=[dict(finish_reason='length')]))
 monkeypatch.setattr(f,'infer',reject);f.run();db=f.connect()
 assert db.execute('SELECT count(*) FROM reviews').fetchone()[0]==0
 assert db.execute("SELECT count(*) FROM attempts WHERE status='failed'").fetchone()[0]==2;db.close()
 monkeypatch.setattr(f,'infer',infer);f.run(retry_failed=True);db=f.connect()
 assert db.execute('SELECT count(*) FROM failed_history').fetchone()[0]==2
 assert db.execute('SELECT count(*) FROM reviews').fetchone()[0]==2;db.close()

def test_conservative_policy_keeps_raw_claim_but_abstains_on_contradiction():
 from track1_reference.frame_review import conservative
 r=row();r.update(visibility='visible',physical_contact='yes',uncertainty=['The interface is partly occluded by the fingers.'])
 out,checks=conservative(r)
 assert out['visibility']=='partial' and out['physical_contact']=='unknown'
 assert out['model_reported']['physical_contact']=='yes' and r['visibility']=='visible'
 assert 'model_visibility_contradicts_its_uncertainty' in checks
 first=row();out,checks=conservative(r,first);assert out['visibility']=='unknown'

def test_transition_candidates_never_bridge_missing_or_unknown_frames(tmp_path,monkeypatch):
 import json
 import track1_reference.frame_review as f
 monkeypatch.setattr(f,'STORE',tmp_path);db=f.connect()
 def put(i,relation):
  r=row();r['projected']=relation
  db.execute('INSERT OR REPLACE INTO reviews VALUES(?,?,?,?,?,?,?)',('a','c',i,i*20000,json.dumps(r),'[]','b'))
 put(0,'gap');put(2,'meeting');f.transitions(db,'a');assert db.execute('SELECT count(*) FROM transitions').fetchone()[0]==0
 put(1,'unknown');f.transitions(db,'a');assert db.execute('SELECT count(*) FROM transitions').fetchone()[0]==0
 put(1,'gap');f.transitions(db,'a');events=db.execute('SELECT value FROM transitions').fetchall();assert len(events)==1
 e=json.loads(events[0][0]);assert e['earliest_us']==20000 and e['latest_us']==40000 and not e['physical_event_certified'];db.close()
