import hashlib,io,json,sqlite3
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from track1_reference import adaptive_review as a
from track1_reference.frame_checks import motion

def sample(task='case'):
 return dict(task_id=task,found=True,description='Partly visible hand and object',visibility='partial',projected='overlap',physical_contact='unknown',evidence='Covered interface',uncertainty=['occluded'],attention_box=[100,100,400,400],landmark=None,landmark_definition='',motion_observation='unknown',identity_evidence='unknown')

def test_native_motion_proxy_recovers_known_translation_and_rejects_blank():
 rng=np.random.default_rng(2);im=rng.integers(0,256,(512,512),dtype=np.uint8);shifted=cv2.warpAffine(im,np.float32([[1,0,5],[0,1,3]]),(512,512));r=motion(im,shifted)
 assert r['flow_valid'];np.testing.assert_allclose(np.asarray(r['transform'])[:,2],[5,3],atol=.15)
 assert not motion(np.zeros_like(im),np.zeros_like(im))['flow_valid']

def test_selection_keeps_short_detected_episode_queries_and_endpoints():
 fs=[dict(timestamp_us=i*20000) for i in range(101)];metrics=[dict(flow_valid=True,p95_flow_px=0) for f in fs];metrics[17]['p95_flow_px']=30
 picks=a.pick_frames(fs,metrics,[23]);assert {0,17,23,50,100}<=set(picks)
 assert 'motion_or_support_episode' in picks[17]

def test_closeup_is_evidence_specific_and_definite_contact_is_rechecked():
 r=sample();assert not a.wants_closeup(r)
 r.update(visibility='hidden',physical_contact='yes');assert not a.wants_closeup(r)
 r.update(visibility='visible',uncertainty=[]);assert a.wants_closeup(r)

def test_single_case_contract_preserves_text_and_drops_unusable_coordinates():
 import pytest
 r=sample();r.pop('task_id');r['uncertainty']='covered';r['attention_box']=[1200,20,1400,50]
 parsed=a.parse(json.dumps(r)+'}]}',dict(task_id='known'))
 assert parsed['task_id']=='known' and parsed['uncertainty'][0]=='covered'
 assert parsed['attention_box'] is None and parsed['description']==r['description']
 r['task_id']='wrong'
 with pytest.raises(ValueError):a.parse(json.dumps(r),dict(task_id='known'))
 with pytest.raises(ValueError):a.parse('',dict(task_id='known'))

def test_saved_reviews_survive_new_ledger_and_shared_queries_are_explicit(tmp_path,monkeypatch):
 from track1_reference import frame_review as old
 store=tmp_path/'old';monkeypatch.setattr(old,'STORE',store);c=old.connect();prior=json.dumps(sample());c.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?)',('case','clip',0,0,prior,'[]','old'));c.commit();c.close()
 backup=store/'reviews.sqlite';work=tmp_path/'work/track1_vlm';work.mkdir(parents=True);(work/'preserved_manifest.json').write_text(json.dumps(dict(backup=str(backup.relative_to(tmp_path)),sha256=hashlib.sha256(backup.read_bytes()).hexdigest())))
 monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'STORE',tmp_path/'new');db=a.connect();assert db.execute('SELECT value FROM reviews').fetchone()[0]==prior
 ts=[dict(task_id=x,kind='contact',clip_id='clip',cue='hand / cup') for x in ['case','alias']];private=dict(clips=dict(clip=[dict(timestamp_us=i*20000,native_frame_uid=str(i)) for i in range(3)]));buf=io.BytesIO();Image.new('RGB',(1920,1080)).save(buf,format='PNG');monkeypatch.setattr(a,'decode',lambda f:buf.getvalue());monkeypatch.setattr(a,'infer',lambda images,t,extra='':(sample(t['task_id']),{'choices':[{'finish_reason':'stop'}]},.1))
 db.execute("INSERT INTO review_plan VALUES('group','clip',1,'[]','pending',NULL)");db.commit();a.review_job(db,private,'group',1,ts)
 assert db.execute("SELECT value FROM reviews WHERE task_id='case' AND idx=0").fetchone()[0]==prior
 assert db.execute('SELECT count(*) FROM reviews').fetchone()[0]==3
 for raw, in db.execute('SELECT value FROM reviews WHERE idx=1'):
  r=json.loads(raw);assert not r['labels_propagated'] and r['shared_cue_is_not_identity'] and not r['boundaries_available']
 assert db.execute("SELECT status FROM review_plan").fetchone()[0]=='complete'
 def unrechecked(images,t,extra=''):
  r=sample(t['task_id']);r.update(visibility='visible',physical_contact='yes',uncertainty=[],attention_box=None)
  return r,{'choices':[{'finish_reason':'stop'}]},.1
 monkeypatch.setattr(a,'infer',unrechecked);db.execute("INSERT INTO review_plan VALUES('other','clip',2,'[]','pending',NULL)");db.commit();a.review_job(db,private,'other',2,ts)
 r=json.loads(db.execute("SELECT value FROM reviews WHERE task_id='case' AND idx=2").fetchone()[0]);assert r['physical_contact']=='unknown' and r['model_reported']['physical_contact']=='yes';db.close()

def test_tracking_refresh_checks_pixels_and_keeps_global_diagnostics(tmp_path,monkeypatch):
 import pytest
 from track1_reference import frame_review as old
 import track1_reference.frame_checks as checks
 monkeypatch.setattr(old,'STORE',tmp_path/'db');c=old.connect();c.execute('CREATE TABLE frame_checks(clip TEXT,idx INTEGER,timestamp_us INTEGER,uid TEXT,value TEXT,PRIMARY KEY(clip,idx))');metrics=dict(gray_sha256='same',median_flow_px=7,attention_tracks={});c.execute('INSERT INTO frame_checks VALUES(?,?,?,?,?)',('clip',0,0,'uid',json.dumps(metrics)));c.commit();monkeypatch.setattr(checks,'scan',lambda *args:('clip',[(0,dict(gray_sha256='same',median_flow_px=99,attention_tracks={'g':{'valid':True}}))],.01));a.refresh_tracking(c,dict(clips=dict(clip=[{}])),{},'clip');r=json.loads(c.execute('SELECT value FROM frame_checks').fetchone()[0]);assert r['median_flow_px']==7 and r['attention_tracks']['g']['valid']
 monkeypatch.setattr(checks,'scan',lambda *args:('clip',[(0,dict(gray_sha256='changed',attention_tracks={}))],.01))
 with pytest.raises(RuntimeError,match='pixels changed'):a.refresh_tracking(c,dict(clips=dict(clip=[{}])),{},'clip')
 c.close()
