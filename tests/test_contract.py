import json
from pathlib import Path
import numpy as np
import pytest
import hashlib
import subprocess
import sys
from pipeline.common import SCHEMAS,write_rows,verify_publisher_file
from pipeline.video import Timeline
from pipeline.geometry import geometry,rasterize
from pipeline.visor import contact_value,hand_side
from pipeline.store import Dataset,ByteCache

def test_irregular_native_clock_and_local_tolerance():
    tl=Timeline([dict(timestamp_us=t,decode_index=i) for i,t in enumerate([0,17000,51000,85000,119000])],margin=1000)
    row,error,limit=tl.nearest(66000)
    assert row['timestamp_us']==51000 and error==15000 and limit==18000
    row,error,limit=tl.nearest(200000)
    assert error>limit  # Far source times cannot be silently attached.

def test_publisher_checksum_rejects_same_size_mutation(tmp_path):
    p=tmp_path/'source.bin';p.write_bytes(b'original')
    entry=dict(extent={'@value':'8'},sha1=hashlib.sha1(b'original').hexdigest())
    verify_publisher_file(p,entry)
    p.write_bytes(b'modified')
    with pytest.raises(ValueError):verify_publisher_file(p,entry)

def test_geometry_preserves_components_and_records_bow_tie_repair():
    rings=[[[1,1],[5,1],[5,5],[1,5]],[[10,10],[13,10],[13,13],[10,13]]]
    original=json.dumps(rings);result=geometry(rings,20,20)
    assert result['component_count']==2 and result['hole_count']==0
    assert json.dumps(rings)==original
    bad=geometry([[[1,1],[9,9],[1,9],[9,1]]],20,20)
    assert bad['geometry_status']=='repaired_for_rasterisation'
    assert bad['repair_code']=='self_intersection_make_valid_linework'
    assert int(rasterize([[[1,1],[9,9],[1,9],[9,1]]],20,20).sum())==bad['area_px']
    assert geometry([[[float('nan'),0],[1,0],[1,1]]],20,20)['geometry_status']=='quarantined'

def test_union_geometry_keeps_hole_topology():
    rings=[[[1,1],[9,1],[9,3],[1,3]],[[1,7],[9,7],[9,9],[1,9]],[[1,1],[3,1],[3,9],[1,9]],[[7,1],[9,1],[9,9],[7,9]]]
    result=geometry(rings,12,12)
    assert result['component_count']==1 and result['hole_count']==1
    assert {r['role'] for r in json.loads(result['contours_json'])}=={'outer','inner'}

def test_contact_states_and_ambiguous_ids_survive():
    assert contact_value('none-of-the-above',{})[0]=='unsegmented_contact_target'
    assert contact_value('hand-not-in-contact',{})[0]=='hand_not_in_contact'
    assert contact_value('inconclusive',{})[0]=='inconclusive'
    assert contact_value('object',{'object':[{},{}]})[2]=='duplicate_target_id'
    assert contact_value('unknown',{})[0]=='unresolved_target_id'

def test_provider_glove_side_lists_and_conflicts_are_explicit():
    side,flags=hand_side(dict(name='right glove',on_which_hand=['left hand']))
    assert side=='left hand' and 'source_hand_name_side_disagreement' in flags
    assert 'source_hand_side_list:["left hand"]' in flags
    side,flags=hand_side(dict(name='glove',on_which_hand=['left hand','right hand']))
    assert side=='inconclusive' and any('left hand' in f and 'right hand' in f for f in flags)

def test_window_overlap_half_open_splits_and_missing_poses(tmp_path):
    v='P99_999'
    for table in SCHEMAS:write_rows(tmp_path,table,v,[])
    write_rows(tmp_path,'videos',v,[dict(video_id=v,participant_id='P99',action_split='validation',visor_split='train',duration_us=2000000)])
    write_rows(tmp_path,'frames',v,[dict(video_id=v,native_frame_uid=str(t),decode_index=i,timestamp_us=t,quality_flags=[]) for i,t in enumerate([0,500000,999999,1000000])])
    write_rows(tmp_path,'actions',v,[dict(video_id=v,action_uid='spanning',start_us=250000,stop_us=1250000,quality_flags=[]),dict(video_id=v,action_uid='touching_end',start_us=1000000,stop_us=1500000,quality_flags=[]),dict(video_id=v,action_uid='touching_start',start_us=-1,stop_us=0,quality_flags=[])])
    with Dataset(tmp_path,require_ready=False) as ds:
        window=ds.get_window(v,end_time_us=1000000)
        assert window.frame_timestamps_us.tolist()==[0,500000,999999]
        assert [a['action_uid'] for a in window.action_intervals]==['spanning']
        assert window.action_split=='validation' and window.visor_split=='train'
        assert all(not p['pose_valid'] for p in window.camera_observations)
    with pytest.raises(RuntimeError):Dataset(tmp_path)

def test_cache_is_bounded_and_returns_independent_arrays():
    cache=ByteCache(8)
    cache.put('a',np.array([1],dtype=np.int64));x=cache.get('a');x[0]=9
    assert cache.get('a')[0]==1
    cache.put('b',np.array([2],dtype=np.int64))
    assert cache.get('a') is None and cache.size<=8

def test_real_vfr_video_index_and_isolated_lazy_decode(tmp_path):
    raw=tmp_path/'raw';(raw/'videos').mkdir(parents=True);out=tmp_path/'canonical';out.mkdir()
    path=raw/'videos/P99_999.MP4'
    encoder='''
import sys,av,numpy as np
from fractions import Fraction
with av.open(sys.argv[1],'w') as c:
 s=c.add_stream('mpeg4',rate=25);s.width=64;s.height=48;s.pix_fmt='yuv420p'
 for i,pts in enumerate([0,1,3,4,7]):
  frame=av.VideoFrame.from_ndarray(np.full((48,64,3),i*40,dtype=np.uint8),format='rgb24')
  frame.pts=pts;frame.time_base=Fraction(1,25)
  for p in s.encode(frame):c.mux(p)
 for p in s.encode():c.mux(p)
'''
    subprocess.run([sys.executable,'-c',encoder,str(path)],check=True)
    cfg=dict(raw=str(raw),output=str(out),decode_threads=2,license='test-fixture',schema_version='test')
    choice=dict(video_id='P99_999',official_md5=hashlib.md5(path.read_bytes()).hexdigest(),action_split='validation',visor_split='train')
    subprocess.run([sys.executable,'-m','pipeline.video','--config-json',json.dumps(cfg),'--choice-json',json.dumps(choice)],check=True)
    with Dataset(out,require_ready=False) as ds:
        frames=ds.rows('SELECT timestamp_us,pts_raw FROM frames ORDER BY decode_index')
        assert [f['timestamp_us'] for f in frames]==[0,40000,120000,160000,280000]
        first=ds.get_frame('P99_999',120000);last=ds.get_frame('P99_999',280000)
        assert first.shape==(48,64,3)
        assert abs(float(first.mean())-80)<5 and abs(float(last.mean())-160)<5
