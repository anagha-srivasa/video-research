import copy,json
import numpy as np
import pytest
from track1.numerics import wavelet_matrix
from track1_encoder_revision.covariance import features_and_jacobian,propagate,independent_input_covariance
from track1_encoder_revision.events import InnovationEvidence
from track1_encoder_revision.phase import circular_matrix,align
from track1_encoder_revision.geometry import localize,repair
from track1_encoder_revision.runtime import CompiledGeometry
from track1_lab.topology_certificate import certify,evaluate_curves

def circle():
 a=np.arange(16)*2*np.pi/16;p=np.c_[np.cos(a),np.sin(a)];c=np.zeros((4,35),np.float32);c[0,2:-1]=(wavelet_matrix(16)@p).ravel();return p,c

def test_covariance_jacobian_and_psd():
 p,c=circle();centre=np.array([2.,3.]);scale=4.;h,j=features_and_jacobian([p],centre,scale);x=np.r_[p.ravel(),centre,scale];num=np.zeros_like(j)
 for k in range(len(x)):
  a=x.copy();b=x.copy();a[k]+=1e-5;b[k]-=1e-5
  num[:,k]=(features_and_jacobian([a[:32].reshape(16,2)],a[32:34],a[-1])[0]-features_and_jacobian([b[:32].reshape(16,2)],b[32:34],b[-1])[0])/2e-5
 np.testing.assert_allclose(j,num,atol=1e-9)
 cov=independent_input_covariance([np.eye(16)*.01],np.eye(2)*.03,.02);r=propagate([p],centre,scale,cov);assert np.linalg.eigvalsh(r['covariance']).min()>-1e-12
 assert abs(r['covariance'][0,2])>0 # shared centre creates cross-channel covariance
 with pytest.raises(ValueError):propagate([p],centre,scale,-cov)

def test_event_evidence_is_causal_latched_and_nonmutating():
 detector=InnovationEvidence();events=[];position=np.zeros(2);before=position.copy()
 for i in range(40):
  t=i*.02;point=position+([200*(t-.2),0] if t>.2 else [0,0]);r=detector.update(t,point,[.01,.01],decision_time=.8);assert r['latest_input_time']<=.8
 events=detector.drain(.8);assert any(x['kind']=='motion_onset' for x in events);assert not detector.drain(.8);np.testing.assert_array_equal(position,before)
 with pytest.raises(ValueError):detector.update(1.,[0,0],[1,1],decision_time=.9)

def test_phase_guard_and_fractional_transform():
 p,c=circle();h=np.repeat(c[0][None],10,axis=0);prepared=dict(h=h,variance=np.ones_like(h)*.01,allocation=[16]);q,ledger=align(prepared,np.ones(10),geometry_guard=lambda i,a,b:False)
 np.testing.assert_array_equal(q['h'],h);np.testing.assert_array_equal(prepared['h'],h)
 np.testing.assert_allclose(circular_matrix(16,1.)@p,np.roll(p,-1,axis=0),atol=1e-12)
 assert ledger['static_mask_guard_applied']

def test_compiled_decoder_binding_and_repair_fail_closed():
 p,c=circle();cert=certify(c,[16],['outer'],[-1],window_seconds=1.);meta=dict(allocation=[16],roles=['outer'],parents=[-1],certificate_json=json.dumps(cert),window_seconds=1.,window_start_us=0)
 fast=CompiledGeometry(c,meta,samples=128)
 for t in [0.,.21,.5,1.]:np.testing.assert_allclose(fast.evaluate(t*1e6)[0],evaluate_curves(c,[16],['outer'],t,128)[0],atol=1e-14)
 with pytest.raises(ValueError):fast.evaluate(1000001)
 changed=c.copy();changed[0,2]+=.01
 with pytest.raises(ValueError):CompiledGeometry(changed,meta)
 good,where=localize(c,[16],['outer'],[-1],max_depth=8,max_nodes=255);assert good['certified'] and where is None
 bad=c.copy();bad[2,2:10]+=4.;result=repair(bad,c[0],[16],['outer'],[-1],np.linspace(0,1,10),np.repeat(c[0][None],10,axis=0))
 if result['coefficients'] is not None:assert result['certificate']['certified']
 assert not result['geometry_covariance_available']


def test_innovation_does_not_change_switching_bank():
 from track1_remediation.motion import switching,predict
 t=np.arange(0,.6,.02);p=np.c_[100+20*t,50-2*t];var=np.ones_like(p)
 before=switching(t,p,var,t[-1]);detector=InnovationEvidence();saved=p.copy()
 for ti,point,v in zip(t,p,var):detector.update(ti,point,v)
 after=switching(t,p,var,t[-1]);np.testing.assert_array_equal(p,saved)
 for h in [.25,.5]:np.testing.assert_array_equal(predict(before,h)[0],predict(after,h)[0])


def test_monte_carlo_conditional_covariance():
 p,_=circle();scale=10.;centre=np.array([.1,.2]);joint=independent_input_covariance([np.eye(16)*1e-4],np.eye(2)*1e-4,1e-4);r=propagate([p],centre,scale,joint);rng=np.random.default_rng(19);noise=rng.multivariate_normal(np.zeros(35),joint,size=12000);controls=p.ravel()[None]+noise[:,:32];centres=centre[None]+noise[:,32:34];scales=scale+noise[:,-1];w=np.kron(wavelet_matrix(16),np.eye(2));shapes=(controls-np.tile(centres,(1,16)))@w.T/scales[:,None];values=np.c_[centres,shapes,np.log(scales)];empirical=np.var(values,axis=0);np.testing.assert_allclose(empirical,np.diag(r['covariance']),rtol=.05,atol=1e-8)


def test_event_short_recent_support_stays_finite_after_gap():
 detector=InnovationEvidence(window=.08)
 for t in [0.,.02,.04,.06,.16]:
  r=detector.update(t,[0.,0.],[1.,1.])
  assert all(value is None or np.isfinite(value) for key,value in r.items() if key in ['innovation_score','cusum','speed_native_px_s'])
 assert not r['support_valid']


def test_reference_audit_rejects_machine_adjudication(tmp_path,monkeypatch):
 from track1_encoder_revision import references
 paths={role:tmp_path/(role+'.json') for role in ['A','B']}
 for role in paths:paths[role].write_text(json.dumps(dict(reviewer_id=role,model='qwen')))
 class FakeRepository:
  tasks={'t':dict(kind='contact')}
  def accepted(self,task_id):return dict(outcome='usable',model='qwen')
  def path(self,role,task_id):return paths[role]
 monkeypatch.setattr(references,'Repository',FakeRepository)
 monkeypatch.setattr(references,'validate',lambda *a:None)
 result=references.audit();assert result['accepted_usable']==0 and result['tasks'][0]['errors']


def test_covariance_respects_normalized_position_units():
 p,_=circle();cov=independent_input_covariance([np.eye(16)],np.diag([4.,9.]),1.);a=np.diag([1/1000.,1/500.]);r=propagate([p],[0.,0.],20.,cov,a)
 np.testing.assert_allclose(np.diag(r['covariance'])[:2],[4e-6,36e-6],atol=1e-15)
