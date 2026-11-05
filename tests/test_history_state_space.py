import numpy as np
import pytest
from numpy.testing import assert_allclose
from track1.numerics import basis
from track1_lab.history_detail import fit_detail,evaluate
from track1_lab.state_space import filter_states,checked_covariance

def test_neutral_detail_preserves_serialized_endpoints_and_derivatives():
 t=np.linspace(0,1.2,61);h=np.zeros((len(t),35));h[:,0]=.03*np.sin(t*9);h[:,1]=.01*t;var=np.full_like(h,1e-6);options=dict(regularization=1/3,variance_power=.25,provenance_power=.5,loss='l2');args=[t,h,var,np.ones(len(t)),np.ones(len(t)),options,np.ones(35)];m=fit_detail(*args)
 assert m['orthogonality_error']<1e-10
 args[1]=h.copy();args[1][t>1]=1e8;n=fit_detail(*args);assert_allclose(m['detail'],n['detail'],atol=0)
 for key in ['coarse','detail','transform']:m[key]=m[key].astype(np.float32).astype(float)
 for d in [0,1]:assert_allclose(evaluate(m,[1.],d),(basis(np.array([1.]),d)*2**d)@m['coarse'],atol=0,rtol=0)
 x=np.array([.61]);eps=1e-6;assert_allclose((evaluate(m,x+eps)-evaluate(m,x-eps))/(2*eps),evaluate(m,x,1),atol=1e-9)
 with pytest.raises(ValueError):evaluate(m,[1.1])

def test_kalman_is_causal_handles_missing_samples_and_rejects_late_duplicates():
 t=np.arange(0,1.201,.02);p=np.c_[.1*t,-.2*t];v=np.full_like(p,1e-10);first=filter_states(t,p,v,[.8,1.],robust=False);changed=p.copy();changed[t>1]=np.nan;second=filter_states(t,changed,v,[.8,1.],robust=False)
 for a,b in zip(first,second):assert_allclose(a['state'],b['state'],atol=0);assert a['latest_input_time']<=1.;assert np.linalg.eigvalsh(a['covariance']).min()>=0
 assert_allclose(first[-1]['state'],[.1,-.2,.1,-.2],atol=1e-4)
 keep=~((t>.5)&(t<.6));assert np.isfinite(filter_states(t[keep],p[keep],v[keep],[1.])[0]['state']).all()
 with pytest.raises(ValueError):filter_states([0.,0.],np.zeros((2,2)),np.ones((2,2)),[1.])
 with pytest.raises(ValueError):filter_states([.1,0.],np.zeros((2,2)),np.ones((2,2)),[1.])
 assert checked_covariance(np.diag([1.,-1e-14])).min()>=0
 with pytest.raises(ValueError):checked_covariance(np.diag([1.,-.1]))

def test_reference_ingestion_rejects_unverified_files_and_leaves_locked_labels_unread(tmp_path):
 import csv
 from pipeline.common import ROOT,digest
 from track1_lab.remediation_audit import reference_file
 assert reference_file(dict(trajectory_path=''))['status']=='pending'
 assert reference_file(dict(trajectory_path='/outside-project.csv'))['failure_code']=='reference_outside_project'
 assert reference_file(dict(reference_partition='locked',trajectory_path='/not-to-be-opened.csv'))['status']=='declared_locked_not_opened'
 file=tmp_path/'anchors.csv'
 with file.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['timestamp_us','x_native_px','y_native_px','visible','reviewer_id']);writer.writeheader()
  for i in range(21):writer.writerow(dict(timestamp_us=1000000+i*50000,x_native_px=i,y_native_px=2*i,visible='true',reviewer_id='independent_person'))
 row=dict(trajectory_path=str(file.relative_to(ROOT)),independent_reviewer='independent_person',trajectory_sha256=digest(file));result=reference_file(row);assert result['status']=='development_reference_submitted' and not result['scientific_reference_ready']
 row['trajectory_sha256']='bad';assert reference_file(row)['status']=='rejected'
