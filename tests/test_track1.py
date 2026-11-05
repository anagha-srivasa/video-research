import numpy as np
from numpy.testing import assert_allclose
from track1.numerics import spline_basis,wavelet_matrix,basis
from track1.spatial import fit_spline,align_seam,soft_masks,resample
from track1.temporal import fit_function,evaluate,collocation
from track1.tracks import assign_tracks
from track1.artifacts import load_config

def test_periodic_partition_derivatives_and_wavelet_covariance():
    phi=spline_basis();assert_allclose(phi.sum(1),1,atol=1e-14)
    for d in [1,2]:assert_allclose(spline_basis(16,64,3,d).sum(1),0,atol=1e-10)
    w=wavelet_matrix();assert_allclose(w.T@w,np.eye(16),atol=1e-14)
    rng=np.random.default_rng(2);x=rng.normal(size=(200000,16));emp=np.cov((x@w.T).T)
    assert_allclose(emp,w@np.eye(16)@w.T,atol=.02)

def test_spatial_solver_qr_parity_normal_covariance_and_seam():
    cfg=load_config()['spatial'];angle=np.arange(64)*2*np.pi/64
    y=np.c_[100+40*np.cos(angle),200+30*np.sin(angle)]
    a=fit_spline(y,cfg);b=fit_spline(y,cfg,force_qr=True)
    assert a['rmse']<.1;assert not a['self_intersection']
    assert_allclose(a['controls'],b['controls'],atol=1e-9)
    assert np.linalg.eigvalsh(a['covariance']).min()>0
    assert_allclose(a['normal_var'],np.einsum('qi,ij,qj->q',spline_basis(),a['covariance'],spline_basis()))
    aligned,_=align_seam(np.roll(y,13,axis=0),y);assert_allclose(aligned,y)
    # Outlier downweighting remains convex Huber, not hard deletion.
    noisy=y.copy();noisy[10]+=[100,-80]
    robust=fit_spline(noisy,cfg);plain=fit_spline(noisy,dict(cfg,irls_iterations=1))
    assert np.linalg.norm(robust['curve']-y)<np.linalg.norm(plain['curve']-y)

def test_concave_core_halo_are_signed_distance_not_radial():
    curve=np.array([[0,0],[6,0],[6,2],[2,2],[2,6],[0,6]],float)
    core,halo=soft_masks(curve,np.ones(6)*.2,np.array([[1,1],[4,4],[5,1]]),tau=.1)
    assert core[0]>.99 and core[1]<.001 and core[2]>.99
    assert np.all(halo>=core)

def test_physical_legendre_derivatives_covariance_and_gl4():
    cfg=load_config()['temporal'];cfg=dict(cfg,lambda_modes=0.,lambda_ridge=1e-12)
    rng=np.random.default_rng(9);t=np.r_[0,np.sort(rng.uniform(0,1,70)),1]
    coefficients=rng.normal(size=(4,35))*.05
    y=basis(2*t-1)@coefficients
    result=fit_function(t,y,np.ones_like(y)*.01,np.ones(len(t)),cfg,0)
    assert result.valid;assert_allclose(result.coefficients,coefficients,atol=1e-9)
    h,v,a=evaluate(coefficients,np.array([1.]))
    assert_allclose(result.endpoint,h[0],atol=1e-9);assert_allclose(result.endpoint_velocity,v[0],atol=1e-8);assert_allclose(result.endpoint_accel,a[0],atol=1e-7)
    assert np.all(result.endpoint_cov>=0)
    nodes=collocation(coefficients);assert nodes['xi'].shape==(4,68)
    for p in range(8):assert_allclose(np.sum(nodes['weights']*nodes['nodes']**p),0 if p%2 else 1/(p+1),atol=1e-14)
    sparse=fit_function(t[:3],y[:3],np.ones_like(y[:3]),np.ones(3),cfg,0);assert not sparse.valid
    gap=fit_function(t[:30],y[:30],np.ones_like(y[:30]),np.ones(30),cfg,0);assert not gap.valid

def test_tracks_no_noun_guessing_and_explicit_duplicates():
    def row(i,t,chain='a',name='cup'):
        return dict(mask_uid=str(i),video_id='v',subsequence='s',interpolation_id=chain,entity_name_open=name,class_id=1,timestamp_us=t,bbox=np.array([0,0,10,10]),source_kind='dense_interpolated')
    rows=assign_tracks([row(0,0),row(1,100000),row(2,100000,'b'),row(3,200000,'b'),row(4,1000000,'c')])
    assert len({r['entity_track_id'] for r in rows[:4]})==1
    assert rows[4]['entity_track_id']!=rows[0]['entity_track_id']
    assert sum(r['selected_observation'] for r in rows)==4
    dup=assign_tracks([row(0,0),row(1,0),row(2,10000)])
    assert len({r['entity_track_id'] for r in dup})==3
    assert all(r['identity_evidence']=='ambiguous_duplicate_split' for r in dup)

def test_functional_rebuild_reads_its_own_output_calibration(tmp_path,monkeypatch):
    """A separate experiment can rebuild without any default derived/ store."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import track1.artifacts as artifacts
    import track1.offline as offline
    cfg=load_config();cfg=dict(cfg,output='alternate')
    monkeypatch.setattr(artifacts,'ROOT',tmp_path);monkeypatch.setattr(offline,'ROOT',tmp_path)
    monkeypatch.setattr(offline,'memory_guard',lambda cfg:None)
    for directory in ['reports','configs','canonical/v1','alternate/track1_spatial_v1','alternate/track1_appearance_v1','alternate/track1_calibration_v1']:(tmp_path/directory).mkdir(parents=True,exist_ok=True)
    (tmp_path/'reports/freeze_manifest.json').write_text('{}')
    (tmp_path/'configs/track1_implementation_plan.md').write_text('fixture')
    (tmp_path/'alternate/track1_calibration_v1/model.json').write_text(json.dumps(dict(kappa_cov=1.,temporal_covariance_multiplier=1.)))
    video='P04_109';n=11;t=np.arange(n)*100000
    rows=[dict(spatial_valid=True,entity_track_id='one',row_index=i,timestamp_us=int(t[i]),provenance_weight=1.,geometry_weight=1.,source_kind='manual_sparse',pose_valid=False) for i in range(n)]
    spatial=tmp_path/f'alternate/track1_spatial_v1/{video}'
    artifacts.save_index(spatial.with_suffix('.parquet'),rows)
    h=np.tile(t[:,None]/1e6,(1,35));artifacts.save_shard(spatial.with_suffix('.pt'),dict(h=h.astype(np.float32),variance=np.ones((n,35),np.float32)*.01))
    artifacts.save_shard(tmp_path/f'alternate/track1_appearance_v1/{video}.pt',dict(embedding=np.zeros((n,128),np.float16),valid=np.ones(n,bool)))
    pq.write_table(pa.Table.from_pylist([dict(duration_us=1250000)]),tmp_path/f'canonical/v1/videos.{video}.parquet')
    result=offline.functional_video(cfg,dict(video_id=video,action_split='train',visor_split='train'))
    assert result['valid']>0
    assert not (tmp_path/'derived').exists()
    rebuilt=artifacts.load_shard(tmp_path/f'alternate/track1_functional_v1/{video}.pt')
    assert rebuilt['coefficients'].shape[1:]==(4,35)
