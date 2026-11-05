"""Local failure localization/repair around the unchanged certificate decoder."""
import numpy as np
from track1.numerics import wavelet_matrix,basis
from track1_lab import topology_certificate as tc

def localize(c,allocation,roles,parents,**kwargs):
 cert=tc.certify(c,allocation,roles,parents,**kwargs)
 if cert['certified']:return cert,None
 lo,hi,ids=tc.nets(c,allocation);stack=[(lo,hi,0.,1.,0)];visited=0
 while stack:
  a,b,start,end,depth=stack.pop();visited+=1;aa,bb,ii=a,b,ids
  for refinement in range(3):
   passed,clearance,reason=tc.interval_check(aa,bb,ii,roles,parents,kwargs.get('tolerance',1e-8))
   if passed:break
   if refinement<2:aa,bb,ii=tc.spatial_refine(aa,bb,ii)
  if passed:continue
  if depth<kwargs.get('max_depth',8) and visited<=kwargs.get('max_nodes',255):
   l,h,r,s=tc.bisect(a,b,1);mid=(start+end)/2;stack.extend([(r,s,mid,end,depth+1),(l,h,start,mid,depth+1)]);continue
  middle=tc.point_net(aa,bb,ii);mid=np.einsum('t,ntsc->nsc',[.125,.375,.375,.125],middle);axis=mid[:,-1]-mid[:,0];axis/=np.maximum(np.linalg.norm(axis,axis=1)[:,None],1e-30);dl=tc.down(aa[:,:,1:]-bb[:,:,:-1]);dh=tc.up(bb[:,:,1:]-aa[:,:,:-1]);lower,_=tc.dot_bounds(dl,dh,axis[:,None,None,:]);bad=np.flatnonzero(lower.min(axis=(1,2))<=kwargs.get('tolerance',1e-8));pairs=[];margin=float(lower.min())
  if reason=='nonlocal_hull_separation_unresolved':
   low=aa.min(axis=(1,2));high=bb.max(axis=(1,2));gap=np.maximum(tc.down(low[None]-high[:,None]),tc.down(low[:,None]-high[None])).max(2);valid=np.triu(np.ones(gap.shape,bool),1)
   for ring in np.unique(ii):
    at=np.flatnonzero(ii==ring)
    for x,y in zip(at,np.roll(at,-1)):valid[min(x,y),max(x,y)]=False
   indices=np.argwhere(valid&(gap<=kwargs.get('tolerance',1e-8)));pairs=indices[np.argsort(gap[indices[:,0],indices[:,1]])][:8].tolist() if len(indices) else [];bad=np.unique(np.array(pairs,int).ravel()) if pairs else np.arange(len(ii));margin=float(min((gap[x,y] for x,y in pairs),default=0.))
  if reason=='adjacent_piece_cones_unresolved':
   nxt=np.arange(len(ii))+1
   for ring in np.unique(ii):at=np.flatnonzero(ii==ring);nxt[at[-1]]=at[0]
   ax=mid[nxt,1]-mid[:,2];ax/=np.maximum(np.linalg.norm(ax,axis=1)[:,None],1e-30);_,left=tc.dot_bounds(tc.down(aa[:,:,:3]-bb[:,:,3:4]),tc.up(bb[:,:,:3]-aa[:,:,3:4]),ax[:,None,None,:]);right,_=tc.dot_bounds(tc.down(aa[nxt,:,1:]-bb[nxt,:,:1]),tc.up(bb[nxt,:,1:]-aa[nxt,:,:1]),ax[:,None,None,:]);margins=np.minimum((-left).min((1,2)),right.min((1,2)));bad=np.flatnonzero(margins<=kwargs.get('tolerance',1e-8));margin=float(margins.min())
  if not len(bad):bad=np.arange(len(ii))
  factor=2**refinement;piece_ids=sorted(set(int(i)//factor for i in bad));offsets=np.cumsum([0]+list(allocation));locations=[]
  for piece in piece_ids:
   ring=int(np.searchsorted(offsets[1:],piece,side='right'));locations.append(dict(ring=ring,piece=int(piece-offsets[ring])))
  return cert,dict(reason=reason,normalized_interval=[start,end],piece_locations=locations,hull_pairs=pairs,margin_proxy=margin,spatial_refinement=refinement,unresolved_is_not_proven_intersection=True)
 return cert,dict(reason=cert['failure_code'],piece_locations=[],unresolved_is_not_proven_intersection=True)

def rollback(c,anchor,allocation,roles,parents,kwargs):
 for amount in [1.,.75,.25,0.]:
  trial=np.array(c,float);trial[:,2:-1]*=amount;trial[0,2:-1]+=(1-amount)*np.asarray(anchor)[2:-1];trial=trial.astype(np.float32);cert=tc.certify(trial,allocation,roles,parents,**kwargs)
  if cert['certified']:return trial,cert,amount
 return None,None,None

def repair(c,anchor,allocation,roles,parents,retained_time,retained_h,weights=None,**kwargs):
 kwargs={'window_seconds':1.,'max_depth':8,'max_nodes':255,**kwargs};c=np.asarray(c,np.float32);cert,where=localize(c,allocation,roles,parents,**kwargs);attempts=[]
 if cert['certified']:return dict(coefficients=c,certificate=cert,method='unchanged',localization=None,attempts=[],geometry_covariance_available=False)
 base,base_cert,amount=rollback(c,anchor,allocation,roles,parents,kwargs);phi=basis(2*np.asarray(retained_time)-1);truth=np.asarray(retained_h)[:,2:-1];weights=np.ones(len(truth)) if weights is None else np.asarray(weights)
 def cost(x):return float(np.average(np.mean((phi@x[:,2:-1]-truth)**2,axis=1),weights=weights))
 best=base;best_cert=base_cert;bestcost=cost(base) if base is not None else float('inf');method='global_rollback' if base is not None else 'unresolved';current=c.copy();last=where;targets=set()
 for step in range(4):
  if not last or not last.get('piece_locations'):break
  for item in last['piece_locations']:
   for k in range(4):targets.add((item['ring'],(item['piece']+k)%allocation[item['ring']]))
  trial=current.astype(float);offset=2
  for ring,m in enumerate(allocation):
   w=wavelet_matrix(m);ctrl=np.einsum('ij,tjc->tic',w.T,trial[:,offset:offset+2*m].reshape(4,m,2));a=w.T@np.asarray(anchor)[offset:offset+2*m].reshape(m,2)
   for r,index in targets:
    if r==ring:ctrl[:,index]*=.5;ctrl[0,index]+=.5*a[index]
   trial[:,offset:offset+2*m]=np.einsum('ij,tjc->tic',w,ctrl).reshape(4,2*m);offset+=2*m
  trial=trial.astype(np.float32);test,local=localize(trial,allocation,roles,parents,**kwargs);value=cost(trial);attempts.append(dict(step=step,controls_changed=sorted([list(x) for x in targets]),certified=test['certified'],retained_shape_mse=value,failure=local))
  if test['certified'] and value<=bestcost:best=trial;best_cert=test;bestcost=value;method='local_projection';break
  current=trial;last=local
 return dict(coefficients=best,certificate=best_cert,method=method,global_amount=amount,localization=where,attempts=attempts,retained_shape_mse=bestcost if np.isfinite(bestcost) else None,geometry_covariance_available=False)
