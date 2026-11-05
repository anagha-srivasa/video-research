"""Canonical raster rings -> model rings, retaining the full containment tree."""
import json
import numpy as np

VERSION = 'canonical_geometry2'

def normalize(rings):
    if isinstance(rings, str):
        rings = json.loads(rings)
    result = []
    for i, ring in enumerate(rings):
        role = {'outer': 'outer', 'inner': 'hole', 'hole': 'hole'}.get(ring['role'])
        if role is None:
            raise ValueError(f'Unknown ring role: {ring["role"]}')
        p = np.asarray(ring['points'], float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError('Invalid ring coordinates')
        parent = int(ring.get('parent_ring', -1))
        if parent < -1 or parent >= len(rings) or parent == i:
            raise ValueError('Invalid ring parent')
        result.append(dict(ring, role=role, parent_ring=parent, points=p.tolist()))
    for i, r in enumerate(result):
        depth = 0; parent = r['parent_ring']; seen = {i}
        while parent >= 0:
            if parent in seen:
                raise ValueError('Cyclic ring hierarchy')
            seen.add(parent); depth += 1; parent = result[parent]['parent_ring']
        if r['role'] != ('outer' if depth % 2 == 0 else 'hole'):
            raise ValueError('Role disagrees with containment depth')
        if 'depth' in r and r['depth'] != depth:
            raise ValueError('Stored depth disagrees with parents')
        r['depth'] = depth
    return result

def render(rings, shape):
    """Depth order preserves islands within holes; coordinates remain pixel centres."""
    import cv2
    rings = normalize(rings)
    mask = np.zeros(shape, np.uint8)
    for r in sorted(rings, key=lambda r: r['depth']):
        if len(r['points']) >= 3:
            cv2.fillPoly(mask, [np.rint(r['points']).astype(np.int32)], 1 if r['role'] == 'outer' else 0)
    return mask

def select(rings, limit=3):
    """Bounded ring selection retains ancestors and remaps every parent index."""
    rings = normalize(rings); selected = []
    for i in sorted(range(len(rings)), key=lambda i: -abs(rings[i]['signed_area'])):
        if len(rings[i]['points']) < 3: continue
        chain = []; j = i
        while j >= 0 and j not in selected:
            chain.append(j); j = rings[j]['parent_ring']
        extra = list(reversed(chain))
        if any(len(rings[j]["points"]) < 3 for j in extra): continue
        if len(selected) + len(extra) <= limit:
            selected.extend(extra)
    mapping = {old: new for new, old in enumerate(selected)}
    return [dict(rings[i], source_ring_index=i, parent_ring=mapping.get(rings[i]['parent_ring'], -1)) for i in selected]
