"""Unknown and shared machine evidence must never become strong supervision."""
from track1_retrain.prepare import masked_annotation


def test_machine_supervision_masks():
    visible=dict(found=True,visibility='visible',uncertainty=[],projected='overlap')
    result=masked_annotation(visible,[])
    assert result['projected_weak_mask'] and result['projected_label']==2
    for field in ['physical_contact_mask','boundary_mask','identity_mask','velocity_mask','event_mask','human_reference_eligible']:
        assert not result[field]
    assert not masked_annotation(visible,[],shared=True)['projected_weak_mask']
    assert not masked_annotation(visible,['full_view_closeup_disagreement'])['projected_weak_mask']
    assert not masked_annotation(dict(visible,uncertainty=['occluded']),[])['projected_weak_mask']
    assert not masked_annotation({},[])['projected_weak_mask']


def test_sequence_contract_rejects_missing_hops(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from track1_retrain import prepare
    monkeypatch.setattr(prepare,'OUT',tmp_path)
    (tmp_path/'indices').mkdir()
    states=[dict(source='context',video_id='P02_122',entity_track_id='x',window_end_us=t,
                 valid=True,epoch='e',episode_id='episode',row_index=i,
                 appearance_valid=True,kinematics_valid=True,m32_available=False)
            for i,t in enumerate([1000000,1250000])]
    seq=dict(source='context',video_id='P02_122',split='train',entity_track_id='x',
             end_times_us=[1000000,1250000],sequence_length=2,episode_id='episode',
             row_indices=[0,1],appearance_mask=[True,True],kinematics_mask=[True,True],m32_mask=[False,False])
    pq.write_table(pa.Table.from_pylist(states),tmp_path/'states.parquet',compression=None)
    path=tmp_path/'indices/train.parquet'
    pq.write_table(pa.Table.from_pylist([seq]),path,compression=None)
    assert prepare.verify_indices()==1
    states[1]['window_end_us']=1500000;seq['end_times_us'][1]=1500000
    pq.write_table(pa.Table.from_pylist(states),tmp_path/'states.parquet',compression=None)
    pq.write_table(pa.Table.from_pylist([seq]),path,compression=None)
    with pytest.raises(AssertionError):prepare.verify_indices()
