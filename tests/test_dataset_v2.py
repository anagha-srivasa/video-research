import json
import numpy as np
import pytest
from pipeline.geometry import geometry
from data_v2.geometry import normalize,render,select

def test_real_canonical_hole_contract():
    source=geometry([[[1,1],[19,1],[19,4],[4,4],[4,16],[19,16],[19,19],[1,19]],[[16,1],[19,1],[19,19],[16,19]]],24,24)
    assert source['hole_count']==1
    rings=normalize(source['contours_json'])
    assert sum(r['role']=='hole' for r in rings)==source['hole_count']
    mask=render(rings,(24,24))
    assert mask[10,10]==0 and mask[2,2]==1
    chosen=select(rings)
    assert normalize(chosen)==chosen

def test_hierarchy_retains_island_and_rejects_unknown():
    rings=[]
    for i,(a,b,role) in enumerate([(1,19,'outer'),(4,16,'inner'),(7,13,'outer')]):
        rings.append(dict(role=role,parent_ring=i-1,depth=i,signed_area=(b-a)**2,points=[[a,a],[b,a],[b,b],[a,b]]))
    mask=render(rings,(21,21))
    assert mask[2,2] and not mask[5,5] and mask[9,9]
    assert len(select(rings,2))==2
    bad=[dict(rings[0],role='background')]
    with pytest.raises(ValueError):normalize(bad)
    with pytest.raises(ValueError):normalize([dict(rings[0],parent_ring=0)])

def test_development_consumers_share_canonical_hole_contract():
    from track1_lab.compound import select_rings
    from track1_lab.architecture import source_geometry
    from track1_lab.geometry_contract import dominant_ring,topology_signature
    source=geometry([[[1,1],[19,1],[19,4],[4,4],[4,16],[19,16],[19,19],[1,19]],[[16,1],[19,1],[19,19],[16,19]]],24,24)
    row=dict(source,annotation_width=24,annotation_height=24)
    rings,allocation,roles,total=select_rings(row,[24,24])
    assert 'hole' in roles and total==2
    assert dominant_ring(row,np.eye(3))[2]==1
    shape,normalized=source_geometry(row)
    from shapely.geometry import Point
    assert not shape.contains(Point(10,10))
    assert topology_signature(normalized)==topology_signature(json.loads(row['contours_json']))

def test_appearance_pool_excludes_canonical_hole():
    from track1_lab.appearance import roi_masks
    source=geometry([[[1,1],[19,1],[19,4],[4,4],[4,16],[19,16],[19,19],[1,19]],[[16,1],[19,1],[19,19],[16,19]]],24,24)
    source.update(annotation_width=24,annotation_height=24)
    angle=np.arange(16)*2*np.pi/16
    state=dict(controls=(np.c_[10+10*np.cos(angle),10+10*np.sin(angle)])[None],normal_variance=np.zeros((1,64)))
    hard,soft,target=roi_masks(dict(row_index=0,width=24,height=24),state,source,(0,0,24,24),size=24)
    assert target[10,10]==0 and target[2,2]==1
    assert hard[10,10]==0
