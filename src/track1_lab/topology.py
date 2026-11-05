"""Sampled-time contour validity and minimal shape-only functional shrinkage."""
from collections import defaultdict
import json
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import LinearRing
from shapely import linearrings,is_simple
from track1.numerics import basis,wavelet_matrix
from refinement.kernels import periodic_design
from refinement.data import temporal_cases,holdout
from refinement.protocol import rank
from pipeline.common import ROOT
from track1.artifacts import save_index
from .common import config,path,allowed,report,LOG,identity
from .temporal import fit
W=wavelet_matrix();PHI=periodic_design(np.arange(128)/128);SYNTHESIS=PHI@W.T;CHECK_TIMES=np.unique(np.r_[np.linspace(0,1,17),(.5*(np.polynomial.legendre.leggauss(4)[0]+1))])

def validity(coefficients):
 shapes=(basis(2*CHECK_TIMES-1)@coefficients)[:,2:34].reshape(-1,16,2);curves=np.einsum('qn,tnc->tqc',SYNTHESIS,shapes);area=np.sum(curves[:,:,0]*np.roll(curves[:,:,1],-1,axis=1)-curves[:,:,1]*np.roll(curves[:,:,0],-1,axis=1),axis=1)
 return np.asarray(is_simple(linearrings(curves)),bool)&(area>1e-8)

def guard(coefficients,covariance,observed_h):
 original=np.asarray(coefficients,float);cov=np.asarray(covariance,float);valid=validity(original)
 if valid.all():return dict(coefficients=original.copy(),covariance=cov.copy(),alpha=1.,reference_index=None,guard_applied=False,valid=True)
 # Reference candidates are observed valid shapes, selected by distance to the fitted mean.
 order=np.argsort(np.sum((observed_h[:,2:34]-original[0,2:34])**2,axis=1));best=None
 for index in order[:min(6,len(order))]:
  reference=observed_h[index,2:34]
  for alpha in [.9,.75,.5,.25,0.]:
   c=original.copy();c[:,2:34]*=alpha;c[0,2:34]+=(1-alpha)*reference
   if validity(c).all():
    loss=float(np.mean((basis(2*np.linspace(0,1,17)-1)@(c-original))**2));candidate=dict(coefficients=c,alpha=alpha,reference_index=int(index),guard_applied=True,valid=True,distortion=loss)
    if best is None or candidate['distortion']<best['distortion']:best=candidate
    break
 if best is None:return dict(coefficients=original.copy(),covariance=cov.copy(),alpha=None,reference_index=None,guard_applied=False,valid=False)
 delta=best['coefficients']-original;outcov=cov.copy();outcov[2:34]*=best['alpha']**2;outcov[2:34]+=np.einsum('id,jd->dij',delta[:,2:34],delta[:,2:34]);best['covariance']=outcov;return best


def run(monitor):
 cfg=config();selected=json.loads((path('reports')/'derivative_candidate.json').read_text())['candidate']['options'];baseline=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text())['temporal']['options'];cal=json.loads((ROOT/'derived/track1_refined_v2/calibration/model.json').read_text());scale=np.asarray(cal['feature_scale']);records=[]
 for video in cfg['train']+cfg['validation']:
  for case in temporal_cases(allowed([video])):
   t=case['time'];h=case['h'];centre=[.25,.5,.75][int(rank(f"{case['entity_track_id']}:{case['window_end_us']}")[:8],16)%3]
   for protocol in ['full','gap100','endpoint100']:
    hidden=np.zeros(len(t),bool) if protocol=='full' else holdout(t,protocol,centre);keep=~hidden
    if keep.sum()<8 or (protocol!='full' and not hidden.any()):continue
    for method,o in [('v2_cubic',baseline),('slope_cubic',selected)]:
     result=fit(t[keep],h[keep],case['variance'][keep]*cal['kappa_cov'],case['prior'][keep],case['quality'][keep],o,scale);c=result['coefficients'];before=validity(c);fixed=guard(c,result['covariance'],h[keep]);query=t[hidden] if hidden.any() else t;target=h[hidden] if hidden.any() else h;prediction=basis(2*query-1)
     records.append(dict(video_id=video,split=case['split'],entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,method=method,invalid_before=bool(not before.all()),invalid_after=not fixed['valid'],guard_applied=fixed['guard_applied'],guard_alpha=fixed['alpha'],normalised_shape_rmse_before=float(np.sqrt(np.mean(((prediction@c-target)[:,2:34]/scale[2:34])**2))),normalised_shape_rmse_after=float(np.sqrt(np.mean(((prediction@fixed['coefficients']-target)[:,2:34]/scale[2:34])**2))),position_coefficients_unchanged=bool(np.array_equal(c[:,:2],fixed['coefficients'][:,:2]))))
  monitor.check();LOG.info('Functional topology audit %s',video)
 save_index(path('reports')/'functional_topology.parquet',records);summary={}
 for split in ['train','validation']:
  summary[split]={}
  for protocol in ['full','gap100','endpoint100']:
   summary[split][protocol]={}
   for method in ['v2_cubic','slope_cubic']:
    rs=[r for r in records if r['split']==split and r['protocol']==protocol and r['method']==method];summary[split][protocol][method]=dict(windows=len(rs),invalid_before_fraction=float(np.mean([r['invalid_before'] for r in rs])),invalid_after_fraction=float(np.mean([r['invalid_after'] for r in rs])),mean_shape_nrmse_before=float(np.mean([r['normalised_shape_rmse_before'] for r in rs])),mean_shape_nrmse_after=float(np.mean([r['normalised_shape_rmse_after'] for r in rs])),guarded_fraction=float(np.mean([r['guard_applied'] for r in rs])))
 report('functional_topology.json',dict(metrics=summary,check_times_seconds=CHECK_TIMES.tolist(),check='21 sampled physical times including endpoints and GL4; positive orientation and simple reconstructed spline.',covariance_limit='Guard scales shape covariance and adds coefficient-displacement outer products as model-discrepancy allowance; it requires fresh train-only calibration. Not a joint probabilistic guarantee.',limits=['This is sampled-time validity, not proof over every continuous time.','Shrinkage can suppress meaningful shape change; held-out shape distortion is reported and candidates are not automatically promoted.','Position coefficients and their physical derivatives are unchanged.'],state='experimental_not_finalized',**identity()))
