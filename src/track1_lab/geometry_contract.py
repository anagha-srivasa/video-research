"""Shared canonical geometry boundary for development consumers.

Frozen v1/v2 implementations remain historical artifacts. New lab runs translate
canonical `inner` to `hole` before association, geometry or appearance use.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import make_valid
from data_v2.geometry import normalize, select, render

def shape(rings,scale=(1.,1.)):
    rings=normalize(rings);g=Polygon()
    for r in sorted(rings,key=lambda r:r['depth']):
        if len(r['points'])<3:continue
        p=make_valid(Polygon(np.asarray(r['points'])*scale))
        g=g.union(p) if r['role']=='outer' else g.difference(p)
    return g,rings

def topology_signature(rings):
    rings=normalize(rings)
    def tree(i):
        return (rings[i]['role'],tuple(sorted(tree(j) for j,r in enumerate(rings) if r['parent_ring']==i)))
    return tuple(sorted(tree(i) for i,r in enumerate(rings) if r['parent_ring']==-1))

def render_curves(curves,roles,shape_):
    """Infer containment for decoded curves; reject inconsistent hole hierarchy."""
    ps=[Polygon(p) if len(p)>=3 else Polygon() for p in curves];rings=[]
    for i,(p,role) in enumerate(zip(curves,roles)):
        parents=[j for j,q in enumerate(ps) if i!=j and q.is_valid and ps[i].is_valid and q.contains(ps[i])]
        parent=min(parents,key=lambda j:ps[j].area) if parents else -1
        rings.append(dict(role=role,points=np.asarray(p).tolist(),parent_ring=parent))
    return render(rings,shape_)

def dominant_ring(row,transform):
    import json
    from track1.spatial import dominant_ring as historical_dominant_ring
    return historical_dominant_ring(dict(row,contours_json=json.dumps(normalize(row['contours_json']))),transform)
