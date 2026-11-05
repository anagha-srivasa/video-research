import numpy as np
from track1_remediation.correspondence import match_rings

def test_equal_centres_distinct_ring_shapes_and_phase_are_resolved():
 t=np.arange(8)*2*np.pi/8;a=np.c_[np.cos(t),np.sin(t)];b=np.c_[2*np.cos(t),.25*np.sin(t)]
 order,margin=match_rings([np.roll(b,3,axis=0),np.roll(a,2,axis=0)],[a,b],['outer','outer'],[-1,-1],['outer','outer'],[-1,-1])
 assert order==(1,0) and margin>=.05
 ambiguous,_=match_rings([a,a],[a,a],['outer','outer'],[-1,-1],['outer','outer'],[-1,-1]);assert ambiguous is None
 changed,_=match_rings([a],[a,b],['outer'],[-1],['outer','outer'],[-1,-1]);assert changed is None


def test_motion_bank_future_independence_stationary_and_mixture_psd():
 from track1_remediation.motion import switching,predict,lookup_camera
 t=np.arange(0,1.02,.02);p=np.c_[100+30*t,50+0*t];var=np.ones_like(p)
 a=switching(t,p,var,.6);changed=p.copy();changed[t>.6]+=10000;b=switching(t,changed,var,.6)
 np.testing.assert_allclose(a['weights'],b['weights']);x,c=predict(a,.25);y,d=predict(b,.25);np.testing.assert_allclose(x,y);assert np.linalg.eigvalsh(c).min()>0;assert a['latest_input_time']<=.6
 quiet=switching(t,np.ones_like(p)*100,var,1.);point,_=predict(quiet,.5);np.testing.assert_allclose(point,[100,100],atol=1e-5)
 rows=[dict(timestamp_us=400000,marker='past'),dict(timestamp_us=650000,marker='future')];assert lookup_camera(rows,500000)['marker']=='past'


def test_variable_geometry_certificate_bound_to_duration_and_interval():
 import json
 from track1_remediation.store import TieredDataset
 from track1_retrain.transport import anchor_geometry
 from track1.numerics import wavelet_matrix
 from track1_lab.topology_certificate import certify
 t=np.arange(16)*2*np.pi/16;controls=np.c_[np.cos(t),np.sin(t)];h=np.r_[0.,0.,(wavelet_matrix(16)@controls).ravel(),0.];c=np.zeros((4,35),np.float32);c[0]=h;cert=certify(c,[16],['outer'],[-1],window_seconds=.4)
 r=dict(window_start_us=600000,window_end_us=1000000,window_seconds=.4,certificate_json=json.dumps(cert),allocation=[16],roles=['outer'],parents=[-1]);g=dict(coefficients=c,metadata=r);state=dict(geometry32=g)
 assert len(TieredDataset.geometry(state,800000))==1
 import pytest
 with pytest.raises(ValueError,match='Outside'):TieredDataset.geometry(state,500000)
 r['window_seconds']=1.
 with pytest.raises(ValueError,match='binding'):TieredDataset.geometry(state,800000)


def test_latest_invalid_topology_observation_cannot_resurrect_old_epoch():
 from track1_remediation.correspondence import segment_indices
 rows=[dict(candidate_epoch='old',fit32_valid=True),dict(candidate_epoch='new',fit32_valid=False)]
 assert segment_indices(dict(indices=[0,1]),rows)==[]
 assert segment_indices(dict(indices=[0]),rows)==[0]


def test_missing_optional_geometry_is_none_and_keeps_core_kinematics(tmp_path,monkeypatch):
 import pyarrow as pa,pyarrow.parquet as pq
 import track1_remediation.store as store
 class Core:
  def __init__(self,root):self.root=root
  def get_state(self,*args,model=16,**kwargs):return dict(metadata=dict(geometry_training_eligible=False),kinematics=dict(method='original_cv',support_valid=True))
  def close(self):pass
 monkeypatch.setattr(store,'Dataset',Core);monkeypatch.setattr(store,'D',tmp_path);dest=tmp_path/'context';dest.mkdir();pq.write_table(pa.Table.from_pylist([dict(entity_track_id='entity',window_end_us=1000000,row_index=0,valid=False)]),dest/'P02_122.geometry.parquet');np.save(dest/'P02_122.coefficients.npy',np.zeros((1,4,67),np.float32))
 with store.TieredDataset() as ds:
  r=ds.get_state('P02_122','entity',1000000);assert r['geometry32'] is None and r['m32_available'] is False and r['kinematics']['method']=='original_cv'
