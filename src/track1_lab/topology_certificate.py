"""Conservative Bernstein-hull certificate for cubic space/time geometry.

The certificate concerns continuous normalized spline curves decoded from the
serialized float32 state and the recorded float64 wavelet matrix. It does not
certify raster masks, physical contact, identity, or unrepresented source rings.
Interval arithmetic rounds each elementary operation outwards. Unresolved
boxes/local tangent cones fail closed at the declared budget and tolerance.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from track1.numerics import wavelet_matrix
from .ring_topology import predicate

def down(x):return np.nextafter(x,-np.inf)
def up(x):return np.nextafter(x,np.inf)

def input_hashes(coefficients,allocation,roles,parents,window_seconds):
 contract=dict(allocation=list(allocation),roles=list(roles),parents=list(parents),window_seconds=float(window_seconds),decoder_version='canonical_tensor_bezier1')
 return dict(coefficient_sha256=hashlib.sha256(np.asarray(coefficients,dtype='<f4').tobytes(order='C')).hexdigest(),geometry_contract_sha256=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest(),decoder_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def transform(lo,hi,matrix,axis=0,rational=False):
 a=np.moveaxis(lo,axis,0);b=np.moveaxis(hi,axis,0);matrix=np.asarray(matrix,float);ml=down(matrix) if rational else matrix;mh=up(matrix) if rational else matrix;ml=np.where(matrix==0,0,ml);mh=np.where(matrix==0,0,mh);shape=(len(matrix),)+a.shape[1:];outlo=np.zeros(shape);outhi=np.zeros(shape);expand=(len(matrix),)+(1,)*(a.ndim-1)
 for k in range(a.shape[0]):
  lower=ml[:,k].reshape(expand);upper=mh[:,k].reshape(expand);products=np.stack([lower*a[k],lower*b[k],upper*a[k],upper*b[k]]);outlo=down(outlo+down(products.min(0)));outhi=up(outhi+up(products.max(0)))
 return np.moveaxis(outlo,0,axis),np.moveaxis(outhi,0,axis)

def nets(coefficients,allocation):
 c=np.asarray(coefficients,np.float32).astype(float)
 if c.shape!=(4,2+2*sum(allocation)+1) or not np.isfinite(c).all():raise ValueError('Certificate requires finite cubic compound coefficients')
 time=np.array([[1,-1,1,-1],[1,-1/3,-1,3],[1,1/3,-1,-3],[1,1,1,1.]])
 lows=[];highs=[];ring_ids=[];offset=2
 for ring,m in enumerate(allocation):
  block=c[:,offset:offset+2*m].reshape(4,m,2);offset+=2*m;lo,hi=transform(block,block,wavelet_matrix(m).T,axis=1)
  lo,hi=transform(lo,hi,time,axis=0,rational=True)
  for i in range(m):
   conversion=np.zeros((4,m));conversion[0,[(i+j)%m for j in range(3)]]=[1/6,4/6,1/6];conversion[1,[(i+j)%m for j in [1,2]]]=[2/3,1/3];conversion[2,[(i+j)%m for j in [1,2]]]=[1/3,2/3];conversion[3,[(i+j)%m for j in [1,2,3]]]=[1/6,4/6,1/6];a,b=transform(lo,hi,conversion,axis=1,rational=True);lows.append(a);highs.append(b);ring_ids.append(ring)
 return np.stack(lows),np.stack(highs),np.asarray(ring_ids)

def point_net(lo,hi,ids):
 point=lo+(hi-lo)/2;next_index=np.arange(len(ids))+1
 for ring in np.unique(ids):
  at=np.flatnonzero(ids==ring);next_index[at[-1]]=at[0]
 a=np.maximum(lo[:,:,3],lo[next_index,:,0]);b=np.minimum(hi[:,:,3],hi[next_index,:,0])
 if np.any(a>b):raise ArithmeticError('Inconsistent shared endpoint enclosure')
 shared=a+(b-a)/2;point[:,:,3]=shared;point[next_index,:,0]=shared;return point

def canonical_nets(coefficients,allocation):
 """Point control net inside the enclosure, with identical shared endpoints.

Certification targets this decoder, rather than silently certifying a separate
floating-knot SciPy implementation. The intended uniform cubic basis agrees to
roundoff, but only this declared decoder receives the interval certificate.
"""
 lo,hi,ids=nets(coefficients,allocation);return point_net(lo,hi,ids),ids

def evaluate_curves(coefficients,allocation,roles,time,samples=512):
 if not 0<=time<=1:raise ValueError('Certified geometry decoder does not extrapolate')
 point,ids=canonical_nets(coefficients,allocation);bt=np.array([(1-time)**3,3*time*(1-time)**2,3*time*time*(1-time),time**3]);result=[]
 for ring,(m,role) in enumerate(zip(allocation,roles)):
  u=np.arange(samples)*m/samples;piece=np.floor(u).astype(int);z=u-piece;bs=np.c_[(1-z)**3,3*z*(1-z)**2,3*z*z*(1-z),z**3];curve=np.einsum('t,ns,ntsc->nc',bt,bs,point[ids==ring][piece]);result.append(curve[::-1] if role=='hole' else curve)
 return result

def bisect(lo,hi,axis):
 """Outward-rounded de Casteljau at exactly representable one half."""
 a=np.moveaxis(lo,axis,0);b=np.moveaxis(hi,axis,0);leftlo=[a[0]];lefthi=[b[0]];rightlo=[a[-1]];righthi=[b[-1]]
 while len(a)>1:
  a=down(down(a[:-1]+a[1:])*.5);b=up(up(b[:-1]+b[1:])*.5);leftlo.append(a[0]);lefthi.append(b[0]);rightlo.append(a[-1]);righthi.append(b[-1])
 return tuple(np.moveaxis(np.stack(x),0,axis) for x in [leftlo,lefthi,rightlo[::-1],righthi[::-1]])

def spatial_refine(lo,hi,ids):
 a,b,c,d=bisect(lo,hi,2);return np.stack([a,c],axis=1).reshape(-1,4,4,2),np.stack([b,d],axis=1).reshape(-1,4,4,2),np.repeat(ids,2)

def dot_bounds(lo,hi,axis):
 products=np.stack([lo*axis,hi*axis]);return down(down(products.min(0)[...,0])+down(products.min(0)[...,1])),up(up(products.max(0)[...,0])+up(products.max(0)[...,1]))

def interval_check(lo,hi,ids,roles,parents,tolerance):
 # Each piece is injective if a fixed projected derivative stays positive.
 middle=point_net(lo,hi,ids);mid=np.einsum('t,ntsc->nsc',[.125,.375,.375,.125],middle);axis=mid[:,-1]-mid[:,0];norm=np.linalg.norm(axis,axis=1);axis=axis/np.maximum(norm[:,None],1e-30);derivlo=down(lo[:,:,1:]-hi[:,:,:-1]);derivhi=up(hi[:,:,1:]-lo[:,:,:-1]);lower,_=dot_bounds(derivlo,derivhi,axis[:,None,None,:])
 if not np.all(lower>tolerance):return False,None,'local_injectivity_unresolved'
 # Adjacent pieces share only their endpoint when their control hulls occupy
 # opposite open half spaces through that moving endpoint.
 next_index=np.arange(len(ids))+1
 for ring in np.unique(ids):
  at=np.flatnonzero(ids==ring);next_index[at[-1]]=at[0]
 axis=mid[next_index,1]-mid[:,2];axis/=np.maximum(np.linalg.norm(axis,axis=1)[:,None],1e-30)
 leftlo=down(lo[:,:,:3]-hi[:,:,3:4]);lefthi=up(hi[:,:,:3]-lo[:,:,3:4]);rightlo=down(lo[next_index,:,1:]-hi[next_index,:,:1]);righthi=up(hi[next_index,:,1:]-lo[next_index,:,:1]);_,left=dot_bounds(leftlo,lefthi,axis[:,None,None,:]);right,_=dot_bounds(rightlo,righthi,axis[:,None,None,:])
 if not np.all((left< -tolerance)&(right>tolerance)):return False,None,'adjacent_piece_cones_unresolved'
 low=lo.min(axis=(1,2));high=hi.max(axis=(1,2));gap=np.maximum(down(low[None]-high[:,None]),down(low[:,None]-high[None])).max(axis=2);pairs=np.triu(np.ones((len(ids),len(ids)),bool),1);pairs[np.arange(len(ids)),next_index]=False;pairs[next_index,np.arange(len(ids))]=False
 if np.any(gap[pairs]<=tolerance):return False,None,'nonlocal_hull_separation_unresolved'
 # The proven hull separation and local cones also permit deformation to this
 # endpoint polygon without intersections; its hierarchy/orientation is shared.
 polygons=[]
 for ring,role in enumerate(roles):
  p=mid[ids==ring,0];polygons.append(p[::-1] if role=='hole' else p)
 status=predicate(polygons,roles,parents)
 if not status['ring_topology_valid']:return False,None,'anchor_hierarchy_or_orientation_invalid'
 between=pairs&(ids[:,None]!=ids[None]);clearance=float(gap[between].min()) if np.any(between) else None
 return True,clearance,None

def certify(coefficients,allocation,roles,parents,window_seconds=.5,tolerance=1e-8,max_depth=10,max_nodes=511,max_spatial_refinement=2):
 lo,hi,ids=nets(coefficients,allocation);stack=[(lo,hi,0.,1.,0)];leaves=[];visited=0;max_space=0;failure=None
 while stack:
  a,b,start,end,depth=stack.pop();visited+=1
  if visited>max_nodes:failure='interval_budget_exhausted';break
  aa,bb,ii=a,b,ids;passed=False
  for refinement in range(max_spatial_refinement+1):
   passed,clearance,reason=interval_check(aa,bb,ii,roles,parents,tolerance);max_space=max(max_space,refinement)
   if passed:break
   if refinement<max_spatial_refinement:aa,bb,ii=spatial_refine(aa,bb,ii)
  if passed:leaves.append((start,end,clearance));continue
  if depth>=max_depth:failure=reason;break
  leftlo,lefthi,rightlo,righthi=bisect(a,b,1);mid=(start+end)/2;stack.extend([(rightlo,righthi,mid,end,depth+1),(leftlo,lefthi,start,mid,depth+1)])
 clearances=[c for _,_,c in leaves if c is not None];certified=failure is None and not stack
 return dict(certified=certified,valid=True if certified else None,min_clearance=min(clearances) if certified and clearances else None,clearance_semantics='conservative inter-ring coordinate-separation lower bound in normalized shape units; null for one ring or incomplete certificate',temporal_resolution=max((end-start)*window_seconds for start,end,_ in leaves) if certified else None,minimum_leaf_seconds=min((end-start)*window_seconds for start,end,_ in leaves) if leaves else None,certified_intervals=len(leaves),visited_intervals=visited,maximum_spatial_refinement=max_space,failure_code=failure,tolerance=tolerance,max_depth=max_depth,max_nodes=max_nodes,precision='serialized_float32_controls; outward_float64_intervals',decoder_version='canonical_tensor_bezier1',scope='continuous normalized canonical tensor-Bezier curves; fixed ring allocation/hierarchy; not independently evaluated SciPy/raster output or source-ring completeness',**input_hashes(coefficients,allocation,roles,parents,window_seconds))
