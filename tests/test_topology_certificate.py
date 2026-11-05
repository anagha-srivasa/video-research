import numpy as np
from track1.numerics import wavelet_matrix,basis
from refinement.kernels import periodic_design
from track1_lab.topology_certificate import nets,certify

def state(rings,allocation):
 c=np.zeros((4,2+2*sum(allocation)+1));c[0,2:-1]=np.concatenate([(wavelet_matrix(m)@p).ravel() for p,m in zip(rings,allocation)]);return c.astype(np.float32)

def circle(m,radius=1.,centre=(0,0)):
 a=np.arange(m)*2*np.pi/m;return np.asarray(centre)+radius*np.c_[np.cos(a),np.sin(a)]

def test_tensor_bezier_conversion_matches_periodic_space_and_legendre_time():
 rng=np.random.default_rng(3);c=rng.normal(size=(4,67)).astype(np.float32);lo,hi,ids=nets(c,[32])
 def bern(x):return np.array([(1-x)**3,3*x*(1-x)**2,3*x*x*(1-x),x**3])
 for t in [.13,.5,.91]:
  controls=wavelet_matrix(32).T@(basis(2*t-1)@c.astype(float))[2:66].reshape(32,2)
  for i,u in [(0,.25),(5,.75),(31,.5)]:
   expected=periodic_design((i+u)/32,m=32)@controls;lower=np.einsum('t,s,tsc->c',bern(t),bern(u),lo[i]);upper=np.einsum('t,s,tsc->c',bern(t),bern(u),hi[i]);assert np.all(expected>=lower-1e-12) and np.all(expected<=upper+1e-12)

def test_canonical_decoder_has_shared_seams_and_matches_uniform_basis():
 from track1_lab.topology_certificate import canonical_nets,evaluate_curves
 from track1_lab.ring_topology import curves
 c=state([circle(24,3),circle(8,.5)],[24,8]);point,ids=canonical_nets(c,[24,8])
 for ring in [0,1]:
  part=point[ids==ring];np.testing.assert_array_equal(part[:,:,3],np.roll(part[:,:,0],-1,axis=0))
 for a,b in zip(evaluate_curves(c,[24,8],['outer','hole'],.37),curves(basis(2*.37-1)@c,[24,8],['outer','hole'])):np.testing.assert_allclose(a,b,atol=1e-12)

def test_static_circle_hole_and_disconnected_components_certify():
 c=state([circle(24,3),circle(8,.5)],[24,8]);result=certify(c,[24,8],['outer','hole'],[-1,0]);assert result['certified'] and result['min_clearance']>0
 c=state([circle(16,1,(-2,0)),circle(16,1,(2,0))],[16,16]);assert certify(c,[16,16],['outer','outer'],[-1,-1])['certified']
 c=state([circle(32)],[32]);result=certify(c,[32],['outer'],[-1]);assert result['certified'] and result['min_clearance'] is None

def test_reader_binds_certificate_to_coefficients_and_declared_decoder():
 import pytest
 from track1_lab.scientific import FunctionalEntityState
 c=state([circle(32)],[32]);certificate=certify(c,[32],['outer'],[-1]);geometry=dict(allocation=[32],roles=['outer'],parents=[-1],window_seconds=.5,topology_epoch_valid=True,ring_correspondence_epoch_valid=True,topology_certificate=certificate);record=FunctionalEntityState(1000000,'example',c,np.zeros(67),np.zeros(67),geometry,None,None,{},None,'test')
 assert record.normalized_geometry_curves(750000)[0].shape==(512,2)
 geometry['topology_epoch_valid']=False
 with pytest.raises(ValueError,match='epoch'):record.normalized_geometry_curves(750000)
 geometry['topology_epoch_valid']=True
 c[0,2]+=1
 with pytest.raises(ValueError,match='changed after certification'):record.normalized_geometry_curves(750000)

def test_compound_fit_never_bridges_a_known_topology_epoch_change():
 from track1_lab.compound import encode_sequence
 p=circle(32);ring=dict(role='outer',area=3.,points=p,index=0);frames={i:dict(uid=str(i),allocation=(32,),roles=('outer',),rings=[ring],centre=np.zeros(2),scale=1.,position=np.zeros(2),log_scale=0.,source_topology=(1,0)) for i in [0,1,2]};frames[1]['source_topology']=(2,0)
 kept,features,excluded,_=encode_sequence([0,1,2],((32,),('outer',)),frames,lambda *args:dict(controls=p,valid=True))
 assert kept.tolist()==[2] and len(features)==1 and excluded['discarded_prior_epoch_observations']==1

def test_continuous_crossing_and_touching_fail_closed():
 c=state([circle(16,1,(-2,0)),circle(16,1,(2,0))],[16,16]);motion=wavelet_matrix(16)@np.tile([2.,0.],(16,1));c[0,2:34]+=motion.ravel();c[1,2:34]=motion.ravel();c[0,34:66]-=motion.ravel();c[1,34:66]=-motion.ravel();assert not certify(c,[16,16],['outer','outer'],[-1,-1],max_depth=5)['certified']
 # Circle control points do not interpolate a unit circle: the actual spline
 # radius is smaller. Test an actual overlap and a separated sub-tolerance gap.
 touching=state([circle(16,1,(0,0)),circle(16,1,(1.9,0))],[16,16]);assert not certify(touching,[16,16],['outer','outer'],[-1,-1],max_depth=3)['certified']
 near=state([circle(16,1,(0,0)),circle(16,1,(1.96,0))],[16,16]);assert not certify(near,[16,16],['outer','outer'],[-1,-1],max_depth=3,tolerance=.02)['certified']
 result=certify(state([circle(32)],[32]),[32],['outer'],[-1],max_nodes=0);assert not result['certified'] and result['failure_code']=='interval_budget_exhausted'
