import numpy as np
import pytest
from dataclasses import replace
from track1_fmr.belief_filter import BeliefTracker, ellipse_iou
from track1_fmr.dorsal_flux import measure


def cloud(x=.3, color=(220,20,10)):
    rgb=np.zeros((100,100,3),np.uint8);rgb[:]=color
    mask=np.zeros((100,100),bool)
    column=int(x*100)
    mask[40:50,column:column+10]=True
    return measure(rgb,mask)


def test_shape_ellipse_iou_identity_disjoint_and_covariance_independence():
    s=np.diag([.002,.001])
    assert ellipse_iou([.3,.4],s,[.3,.4],s)==pytest.approx(1.,abs=1e-5)
    assert ellipse_iou([.2,.4],s,[.8,.4],s)==0
    assert 0 < ellipse_iou([.3,.4],s,[.35,.4],s) < 1


def test_flow_advection_changes_prediction_but_never_observed_status():
    tracker=BeliefTracker()
    a=cloud()
    initial=tracker.update(0.,[a])['tracks'][0]
    result=tracker.update(.1,[],flow_advection={0:(np.array([.02,0]),np.eye(2)*1e-5)})
    state=result['tracks'][0]
    assert state.status=='PREDICTED' and result['flow_advected']==1
    np.testing.assert_allclose(state.mean[:2],initial.mean[:2]+[.02,0])
    np.testing.assert_allclose(state.mean[2:],[.2,0])
    np.testing.assert_array_equal(state.shape_scatter,initial.shape_scatter)
    assert np.linalg.eigvalsh(state.covariance).min() >= 0
    assert not result['covariance_calibrated']
    b=replace(a,centroid=a.centroid+[.04,0])
    observed=tracker.update(.2,[b],flow_advection={0:(np.array([.02,0]),np.eye(2)*1e-5)})
    assert observed['tracks'][0].status=='OBSERVED' and observed['matched']==1


def test_joint_assignment_uses_color_when_positions_and_shape_are_ambiguous():
    red,green=cloud(color=(220,20,10)),cloud(color=(20,220,10))
    tracker=BeliefTracker()
    tracker.update(0.,[red,green])
    result=tracker.update(.1,[green,red])
    np.testing.assert_array_equal(result['clouds'][0].algebraic_sig,red.algebraic_sig)
    np.testing.assert_array_equal(result['clouds'][1].algebraic_sig,green.algebraic_sig)
    assert result['matched']==2 and not result['identity_verified']


def test_stationary_lock_and_dormant_resurrection_preserve_id():
    from track1_fmr.belief_filter import masked_patch, verify_stationary_lock, bind_anchor
    a=cloud();rng=np.random.default_rng(7)
    gray=rng.integers(0,256,(160,160),dtype=np.uint8)
    context=(gray,np.zeros((160,160,2)),.0035,.005)
    patch,_,_=masked_patch(gray,a.centroid,a.shape_scatter)
    assert verify_stationary_lock(gray,a.centroid,a.shape_scatter,context[1],patch,.0035)[0]
    assert not verify_stationary_lock(255-gray,a.centroid,a.shape_scatter,context[1],patch,.0035)[0]
    tracker=BeliefTracker()
    for t in [0.,.1,.2]:tracker.update(t,[a],visual_context=context)
    result=tracker.update(.3,[],visual_context=context)
    assert result['measurement_sources'][0]=='template_verified' and tracker.misses[0]==0
    assert tracker.lifecycle[0]=='CONFIRMED'
    tracker.update(.35,[a],visual_context=context)
    tracker.update(.5,[])
    assert tracker.lifecycle[0]=='COASTING'
    result=tracker.update(1.4,[])
    assert result['archived']==[0] and 0 in tracker.dormant_entities and 0 not in tracker.tracks
    result=tracker.update(1.5,[a])
    assert result['resurrected']==[0] and set(tracker.tracks)=={0} and tracker.next_id==1
    point=a.centroid+np.array([.05,0])
    broad=replace(tracker.tracks[0],shape_scatter=np.eye(2)*.01)
    assert bind_anchor({0:broad},point)==0
    tracker.update(2.6,[]);tracker.update(12.,[])
    assert not tracker.dormant_entities


def test_conditional_calibration_replay_is_finite_and_noise_sensitive():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py'
    spec=importlib.util.spec_from_file_location('fmr_calibration',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for n in range(8):
        row=np.zeros(28);row[0]=n*.1;row[2:4]=[.3+.005*n,.4]
        row[4:8]=(np.eye(2)*1e-5).ravel();rows.append(row)
    score,count,states=module.replay([('episode',np.array(rows))],.001,.005,True)
    assert np.isfinite(score) and count==7 and len(states)==8
    for mean,cov in states.values():
        assert np.isfinite(mean).all() and np.linalg.eigvalsh(cov).min()>=0
    assert module.replay([('episode',np.array(rows))],.001,.05)[0]!=score


def test_tentative_two_hits_expire_and_ambiguous_reid_abstains():
    tracker=BeliefTracker();a=cloud()
    tracker.update(0.,[a]);tracker.update(.1,[a])
    for t in [.2,.3,.4]:tracker.update(t,[])
    assert not tracker.tracks and not tracker.lifecycle
    # Two identical archived candidates cannot be resolved by appearance alone.
    tracker=BeliefTracker()
    for t in [0.,.05,.1,.15,.2]:tracker.update(t,[a,a])
    tracker.update(1.3,[])
    assert len(tracker.dormant_entities)==2
    result=tracker.update(1.4,[a])
    assert not result['resurrected'] and 2 in tracker.tracks


def test_ambiguous_dormant_candidates_use_visual_similarity_when_available():
    tracker=BeliefTracker();a=cloud()
    for t in [0.,.05,.1,.15,.2]:tracker.update(t,[a,a])
    tracker.update(1.3,[])
    result=tracker.update(1.4,[a],reid_similarity=lambda tid,j:.99 if tid==0 else .8)
    assert result['resurrected']==[0] and 1 in tracker.dormant_entities


def test_masked_lock_rejects_foreground_change_despite_identical_background():
    from track1_fmr.belief_filter import masked_patch,verify_masked_stationary_lock
    rng=np.random.default_rng(42);gray=rng.integers(0,256,(160,160),dtype=np.uint8)
    centre=np.array([.5,.5]);shape=np.eye(2)*.0005
    patch,mask,slices=masked_patch(gray,centre,shape)
    assert mask.sum()>=9 and patch.shape[0]<=13
    current=gray.copy();roi=current[slices];roi[mask]=255-roi[mask]
    flow=np.zeros((160,160,2))
    assert not verify_masked_stationary_lock(current,centre,shape,flow,patch,mask,.0035)[0]
    tiny=np.eye(2)*1e-9
    patch,mask,_=masked_patch(gray,centre,tiny)
    assert not verify_masked_stationary_lock(gray,centre,tiny,flow,patch,mask,.0035)[0]


def test_containment_adapts_confirmed_parent_without_minting_child_id():
    from track1_fmr.belief_filter import containment_association_cost
    parent=replace(cloud(),centroid=np.array([.4,.5]),shape_scatter=np.eye(2)*.02,mass=.2)
    child=replace(parent,centroid=np.array([.52,.5]),shape_scatter=np.eye(2)*.0005,mass=.01)
    tracker=BeliefTracker()
    for t in [0.,.1,.2]:tracker.update(t,[parent])
    result=tracker.update(.3,[child])
    assert set(result['tracks'])=={0} and tracker.next_id==1
    np.testing.assert_allclose(tracker.tracks[0].mean[:2],[.46,.5])
    np.testing.assert_allclose(tracker.tracks[0].shape_scatter,(parent.shape_scatter+child.shape_scatter)/2)
    assert containment_association_cost(parent.centroid,parent.shape_scatter,parent.algebraic_sig,
        child.centroid,child.shape_scatter,child.algebraic_sig,parent_area=.2,child_area=.13) is None


def test_short_confirmed_tracks_never_enter_dormant_archive():
    tracker=BeliefTracker();a=cloud()
    for t in [0.,.1,.2,.3]:tracker.update(t,[a])
    tracker.update(1.4,[])
    assert not tracker.tracks and not tracker.dormant_entities


def test_reference_binding_waits_for_confirmation_but_stops_after_five_visible_times():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py'
    spec=importlib.util.spec_from_file_location('fmr_binding',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mean=np.array([.5,.5,0,0]);key=('episode',123,7)
    states={key:(mean,np.eye(4))};supports={key:(np.eye(2)*.01,np.array([.3,.3,.7,.7]),2)}
    assert module.bind_reference_window(states,supports,'episode',123,mean[:2],[.4,.4,.6,.6],1) is None
    supports[key]=(supports[key][0],supports[key][1],3)
    assert module.bind_reference_window(states,supports,'episode',123,mean[:2],[.4,.4,.6,.6],3)==7
    assert module.bind_reference_window(states,supports,'episode',123,mean[:2],[.4,.4,.6,.6],6) is None


def test_vlm_parser_accepts_fenced_arrays_and_rejects_invalid_geometry():
    import importlib.util,json
    from pathlib import Path
    path=Path(__file__).parents[1]/'scripts/exp_track1_fmr/02b_review_and_calibrate.py'
    spec=importlib.util.spec_from_file_location('fmr_review_parser',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    label=dict(index=0,visible=True,x=.5,y=.5,bbox=[.2,.2,.8,.8],confidence=.9)
    assert module.parse_labels('```json\n'+json.dumps([label])+'\n```',1)[0]==label
    assert module.parse_labels(json.dumps({'labels':[label]}),1)[0]==label
    with pytest.raises(ValueError):module.parse_labels(json.dumps([dict(label,x=.9)]),1)


def test_stationary_noise_multiplier_applies_to_the_actual_verified_measurement():
    a=cloud();gray=np.random.default_rng(9).integers(0,256,(160,160),dtype=np.uint8)
    context=(gray,np.zeros((160,160,2)),.0035,.005)
    tracker=BeliefTracker(stationary_noise_multiplier=3.)
    for t in [0.,.1,.2]:tracker.update(t,[a],visual_context=context)
    result=tracker.update(.3,[],visual_context=context)
    assert result['measurement_sources'][0]=='template_verified'
    np.testing.assert_allclose(result['clouds'][0].obs_cov_c,np.eye(2)*3*.005**2)
