import copy
import numpy as np
import pytest
from track1_retrain.temporal import regauge,prepare,phase
from track1.numerics import wavelet_matrix

def fixture():
 t=np.linspace(0,1,51);angle=np.arange(16)*2*np.pi/16;control=np.c_[np.cos(angle),np.sin(angle)];controls=np.array([control+[tt,0] for tt in t]);h=np.array([np.r_[tt,0.,(wavelet_matrix(16)@control).ravel(),0.] for tt in t]);var=np.ones_like(h)*.01
 rows=[dict(timestamp_us=int(tt*1e6),centre_native=[tt,0.],scale_native=1.,fit16_valid=True,prior=1.,quality=1.,row_index=i) for i,tt in enumerate(t)]
 return rows,dict(controls=controls,h=h,variance=var),dict(indices=list(range(51)),window_end_us=1000000)

def test_hidden_targets_do_not_change_retained_gauge():
 rows,data,case=fixture();hidden=np.abs(np.linspace(0,1,51)-.5)<.06;p=prepare(case,rows,data,16,hidden)
 changed=copy.deepcopy(data);changed['h'][hidden]+=100;changed['controls'][hidden]*=-12
 q=prepare(case,rows,changed,16,hidden)
 np.testing.assert_array_equal(p['h'],q['h']);np.testing.assert_array_equal(p['variance'],q['variance'])

def test_cyclic_storage_phase_does_not_change_reencoded_inputs():
 rows,data,case=fixture();a=prepare(case,rows,data,16);other=copy.deepcopy(data)
 for i in range(51):other['controls'][i]=np.roll(other['controls'][i],i%16,axis=0)
 b=prepare(case,rows,other,16);np.testing.assert_allclose(a['h'],b['h'],atol=1e-12)

def test_gap_and_invalid_source_fail_closed():
 rows,data,case=fixture();case['indices']=list(range(10))
 with pytest.raises(ValueError):prepare(case,rows,data,16)

def test_sequence_indices_do_not_bridge_gaps_or_epoch_changes():
 from track1_retrain.store import sequences
 rows=[dict(video_id='P02_122',entity_track_id='track',model=16,valid=True,window_end_us=i*250000,epoch='a',row_index=i,geometry_training_eligible=True) for i in range(8)]
 assert len(sequences(rows,4))==5
 rows[4]['epoch']='b'
 assert len(sequences(rows,4))==1
 rows[4]['window_end_us']+=100000
 assert len(sequences(rows,16))==0

def test_compound_hidden_geometry_never_selects_retained_mode():
 rows,_,case=fixture();t=np.linspace(0,1,51);a=np.arange(24)*2*np.pi/24;b=np.arange(8)*2*np.pi/8;outer=np.c_[3*np.cos(a),3*np.sin(a)];hole=np.c_[.5*np.cos(b),.5*np.sin(b)];controls=np.array([np.r_[outer,hole]+[tt,0] for tt in t]);h=np.array([np.r_[tt,0.,(wavelet_matrix(24)@outer).ravel(),(wavelet_matrix(8)@hole).ravel(),0.] for tt in t]);data=dict(controls=controls,h=h,variance=np.ones_like(h)*.01)
 for row in rows:row.update(fit32_valid=True,allocation=[24,8],roles=['outer','hole'],parents=[-1,0],source_signature='outer_with_hole')
 hidden=abs(t-.5)<.06;p=prepare(case,rows,data,32,hidden)
 for i in np.flatnonzero(hidden):rows[i].update(source_signature='changed_hidden_topology',fit32_valid=False)
 q=prepare(case,rows,data,32,hidden);np.testing.assert_array_equal(p['h'],q['h'])

def test_isolated_spatial_failure_is_filtered_without_losing_supported_window():
 rows,data,case=fixture();rows[25]['fit16_valid']=False
 p=prepare(case,rows,data,16)
 assert len(p['ids'])==50 and p['dropped_invalid_observations']==1 and 25 not in p['ids']

def test_mask_metric_crop_cannot_hide_false_positives():
 from track1_retrain.evaluation import matched_masks
 source=dict(segments_raw='[[[8,8],[12,8],[12,12],[8,12]]]',annotation_width=24,annotation_height=24)
 pred=([np.array([[1.,1.],[20.,1.],[20.,20.],[1.,20.]])],['outer'],[-1])
 result=matched_masks(source,[pred],np.array([24,24]))[0]
 assert result['iou']==pytest.approx(25/400)

def test_channel_family_fit_matches_independent_solutions():
 from track1_retrain.temporal import fit
 from track1_lab.temporal import fit as original
 rows,data,case=fixture();t=np.linspace(0,1,51);h=data['h'];var=data['variance'];prior=np.ones(51);options=dict(regularization=.3,loss='l2',families=dict(position=dict(regularization=.01,loss='huber'),shape=dict(regularization=3.,loss='l2'),scale=dict(regularization=.1,loss='l2')));scale=np.ones(35)
 result=fit(t,h,var,prior,prior,options,scale)
 for name,sl in [('position',slice(0,2)),('shape',slice(2,-1)),('scale',slice(-1,None))]:
  local={k:v for k,v in options.items() if k!='families'};local.update(options['families'][name]);expected=original(t,h[:,sl],var[:,sl],prior,prior,local,scale[sl]);np.testing.assert_allclose(result['coefficients'][:,sl],expected['coefficients'])

def test_target_ring_permutation_does_not_create_latent_error():
 from track1_retrain.temporal import target_h
 def circle(m,r,centre):
  a=np.arange(m)*2*np.pi/m;return np.c_[r*np.cos(a),r*np.sin(a)]+centre
 pieces=[circle(16,1.,[0,0]),circle(8,.2,[-2,0]),circle(8,.2,[2,0])];allocation=[16,8,8];reference=np.r_[0.,0.,np.concatenate([(wavelet_matrix(m)@p).ravel() for m,p in zip(allocation,pieces)]),0.]
 controls=np.concatenate([pieces[0],pieces[2],pieces[1]])[None];rows=[dict(fit32_valid=True,allocation=allocation,roles=['outer']*3,parents=[-1]*3,centre_native=[0.,0.],scale_native=1.)];data=dict(controls=controls,h=reference[None]);prepared=dict(h=reference[None],allocation=allocation,roles=['outer']*3,parents=[-1]*3)
 target,compatible=target_h(rows,data,[0],32,prepared)
 assert compatible.all();np.testing.assert_allclose(target[0],reference,atol=1e-12)
