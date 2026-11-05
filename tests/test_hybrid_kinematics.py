import numpy as np
from numpy.testing import assert_allclose
from track1_lab.causal_kinematics import estimate
from track1_lab.hybrid import source_eligibility

def test_local_motion_is_causal_and_derivative_is_independent_of_history():
 t=np.linspace(0,1.1,56);p=np.c_[10+3*t,20-2*t];v=np.full_like(p,.01);a=estimate(t,p,v,1.)
 p[t>1.]+=1e8;b=estimate(t,p,v,1.)
 assert a.support_valid and a.latest_input_time<=1.
 assert_allclose(a.position,[13,18],atol=1e-12);assert_allclose(a.velocity,[3,-2],atol=1e-11);assert_allclose(a.velocity,b.velocity,atol=0);assert np.linalg.eigvalsh(a.covariance).min()>0
 assert not a.covariance_calibrated and a.derivative_reliable is None and a.acceleration is None

def test_robust_motion_limits_outlier_effect_and_invalid_support_is_explicit():
 t=np.linspace(.8,1,11);p=np.c_[2*t,3*t];p[-2]+=[2,-3];variance=np.full_like(p,.0001);ordinary=estimate(t,p,variance,1.);robust=estimate(t,p,variance,1.,robust=True)
 assert np.linalg.norm(robust.velocity-[2,3])<np.linalg.norm(ordinary.velocity-[2,3])
 absent=estimate(t,p,variance,2.);assert not absent.support_valid and absent.position is None and absent.covariance is None
 duplicate=estimate([.9,.9,1.],np.zeros((3,2)),np.ones((3,2)),1.);assert duplicate.failure_reason=='duplicate_timestamps'

def test_pre_fit_eligibility_keeps_reasons_for_unscorable_probes():
 t=np.linspace(0,1,51);hidden=t>.9;assert source_eligibility(t,hidden)[0]
 eligible,reason=source_eligibility(t,np.zeros(len(t),bool));assert not eligible and reason=='no_holdout_target'
 sparse=np.array([.5,.7,.9,1.]);assert source_eligibility(sparse,np.array([False,False,False,True]))[1]=='too_few_source_observations'

def test_reference_queue_regeneration_preserves_reviewer_work(tmp_path,monkeypatch):
 import csv
 from types import SimpleNamespace
 from track1_lab import scientific
 target=tmp_path/'independent_reference_queue.csv'
 with target.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['episode_uid','video_id','entity_track_id','independent_reviewer','trajectory_path','review_note']);writer.writeheader();writer.writerow(dict(episode_uid='episode',video_id='P02_122',entity_track_id='entity',independent_reviewer='reviewer',trajectory_path='reviewed.csv',review_note='keep this annotation'))
 monkeypatch.setattr(scientific,'path',lambda kind:tmp_path);monkeypatch.setattr(scientific,'config',lambda:dict(train=[],validation=[]));monkeypatch.setattr(scientific,'report',lambda *args:None);monkeypatch.setattr(scientific.pq,'read_table',lambda *args,**kwargs:SimpleNamespace(to_pylist=lambda:[dict(episode_uid='episode',video_id='P02_122',entity_track_id='entity',category='unreviewed')]))
 scientific.reference_queue()
 with target.open() as f:r=list(csv.DictReader(f))[0]
 assert r['independent_reviewer']=='reviewer' and r['trajectory_path']=='reviewed.csv' and r['review_note']=='keep this annotation'
