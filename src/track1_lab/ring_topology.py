"""Joint polygonal topology checks and explicit, rejectable shape rollback.

Historical compound coefficients use positive parameter orientation for every
ring. Decoding reverses hole traversal; the geometry and coefficient means are
unchanged. This convention is recorded with the candidate, not applied to v2.
"""
import numpy as np
from shapely.geometry import Polygon,LinearRing
from track1.numerics import basis
from .variants import synthesis

def curves(state,allocation,roles,samples=512):
 result=[];offset=2
 for m,role in zip(allocation,roles):
  curve=synthesis(m,samples)@state[offset:offset+2*m].reshape(m,2);offset+=2*m
  result.append(curve[::-1] if role=='hole' else curve)
 return result

def predicate(rings,roles,parents=None):
 """Strict containment tree; touching/intersecting boundaries are rejected."""
 simple=True;orientation=True;polygons=[]
 for curve,role in zip(rings,roles):
  finite=bool(np.isfinite(curve).all());area=float(np.sum(curve[:,0]*np.roll(curve[:,1],-1)-curve[:,1]*np.roll(curve[:,0],-1))/2) if finite else 0.
  simple&=finite and bool(LinearRing(curve).is_simple) and abs(area)>1e-10
  orientation&=area>0 if role=='outer' else area<0
  polygons.append(Polygon(curve) if finite else Polygon())
 inferred=[];containment=True;disjoint=True
 for i,p in enumerate(polygons):
  candidates=[j for j,q in enumerate(polygons) if j!=i and q.is_valid and p.is_valid and q.contains(p)]
  parent=min(candidates,key=lambda j:polygons[j].area) if candidates else -1;inferred.append(parent)
  containment&=(parent>=0 if roles[i]=='hole' else True) and (parent<0 or roles[parent]!=roles[i])
 for i,p in enumerate(polygons):
  for j,q in enumerate(polygons[:i]):
   extent=max(np.ptp(rings[i],axis=0).max(),np.ptp(rings[j],axis=0).max(),1.);clearance=1e-9*extent
   if not p.is_valid or not q.is_valid or p.boundary.distance(q.boundary)<=clearance:disjoint=False
   elif p.intersects(q) and not (p.contains(q) or q.contains(p)):disjoint=False
 if parents is not None:containment&=list(parents)==inferred
 result=dict(simple_loop_valid=bool(simple),ring_orientation_valid=bool(orientation),ring_containment_valid=bool(containment),ring_disjoint_valid=bool(disjoint),ring_parent=inferred)
 result['ring_topology_valid']=bool(simple and orientation and containment and disjoint)
 return result

def check(coefficients,allocation,roles,parents=None,samples=512):
 c=np.asarray(coefficients,np.float32).astype(float);q=len(c);times=np.unique(np.r_[np.linspace(-1,1,65),np.polynomial.legendre.leggauss(q)[0]]);states=basis(times,degree=q-1)@c
 if parents is None:parents=predicate(curves(states[-1],allocation,roles,samples),roles)['ring_parent']
 results=[predicate(curves(h,allocation,roles,samples),roles,parents) for h in states];keys=['simple_loop_valid','ring_orientation_valid','ring_containment_valid','ring_disjoint_valid','ring_topology_valid'];out={k:all(r[k] for r in results) for k in keys};out.update(ring_parent=list(parents),temporal_samples=len(times),spatial_samples=samples)
 return out

def guard(coefficients,allocation,roles):
 c=np.asarray(coefficients,np.float32).astype(float);before=check(c,allocation,roles);baseline=before
 if before['ring_topology_valid']:
  baseline=check(c,allocation,roles,before['ring_parent'],2048)
  if baseline['ring_topology_valid']:return dict(coefficients=c,quality=baseline,alpha=1.,shape_mse=0.,guard_applied=False)
 times=np.linspace(-1,1,17);states=basis(times,degree=len(c)-1)@c;templates=[]
 for i in reversed(range(len(times))):
  result=predicate(curves(states[i],allocation,roles,2048),roles)
  if result['ring_topology_valid']:templates.append((states[i],result['ring_parent']))
  if len(templates)==3:break
 best=None
 for h,parents in templates:
  for alpha in [.9,.75,.5,.25,0.]:
   trial=c.copy();trial[:,2:-1]*=alpha;trial[0,2:-1]+=(1-alpha)*h[2:-1]
   quality=check(trial,allocation,roles,parents,128)
   if quality['ring_topology_valid']:quality=check(trial,allocation,roles,parents,2048)
   if quality['ring_topology_valid']:
    loss=float(np.mean((basis(times,degree=len(c)-1)@(trial-c))[:,2:-1]**2));candidate=dict(coefficients=trial.astype(np.float32).astype(float),quality=quality,alpha=alpha,shape_mse=loss,guard_applied=True)
    if best is None or loss<best['shape_mse']:best=candidate
    break
 if best is not None:return best
 return dict(coefficients=c,quality=baseline,alpha=None,shape_mse=0.,guard_applied=False)

def render(state,allocation,roles,parents,camera,target_shape):
 """Rasterize parent-before-child so islands inside holes survive."""
 import cv2
 from .motion import project
 centre=project(state[:2],camera);k=np.asarray(camera['intrinsic_native']);scale=np.sqrt(k[0,0]*k[1,1])*np.exp(np.clip(state[-1],-15,2));factor=np.asarray(target_shape[::-1])/camera['native_size'];rings=curves(state,allocation,roles,512);mask=np.zeros(target_shape,np.uint8)
 def depth(i):
  seen=set();d=0
  while parents[i]>=0:
   if i in seen:raise ValueError('Cyclic ring hierarchy')
   seen.add(i);i=parents[i];d+=1
  return d
 for i in sorted(range(len(rings)),key=depth):
  points=(rings[i]*scale+centre)*factor;cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1 if roles[i]=='outer' else 0)
 return mask
