"""Annotated interaction-region geometry; 2D proximity never substitutes for physical contact truth."""
from collections import defaultdict
import json
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import Polygon,Point,LineString
from shapely.ops import unary_union
from shapely import make_valid
from scipy.spatial import cKDTree
from pipeline.common import ROOT
from track1.artifacts import load_shard,save_index
from track1.spatial import dominant_ring,resample
from refinement.kernels import periodic_design,nearest_curve,spatial_errors
from refinement.protocol import rank
from .geometry_contract import dominant_ring
from .common import config,path,allowed,split,report,pool,LOG,identity
from .spatial import fit as spatial_fit
from .geometry_contract import shape as contract_shape


def geometry(row,width,height):
 scale=np.array([width/row['annotation_width'],height/row['annotation_height']]);g,rings=contract_shape(row['contours_json'],scale)
 points=np.concatenate([resample(np.asarray(r['points'])*scale,min(1024,max(128,len(r['points'])))) for r in rings if len(r['points'])>=3])
 return g,points


def video(v):
 allowed([v]);sp=ROOT/'derived/track1_refined_v2/spatial';s=load_shard(sp/f'{v}.pt');rows=pq.read_table(sp/f'{v}.parquet').to_pylist();lookup={r['mask_uid']:r for r in rows};allm=pq.read_table(ROOT/f'canonical/v1/masks.{v}.parquet').to_pylist();masks={r['mask_uid']:r for r in allm};contact=pq.read_table(ROOT/f'canonical/v1/contacts.{v}.parquet').to_pylist();pairs=[]
 for r in sorted(contact,key=lambda r:rank(r['contact_uid'])):
  if r['source_kind']=='manual_sparse' and r['hand_mask_uid'] in lookup and r['target_mask_uid'] in lookup:pairs.append((r['hand_mask_uid'],r['target_mask_uid'],r['contact_uid'],'annotated_hand_target'))
  if len(pairs)>=64:break
 # Add same-frame non-target candidates to avoid a positive-only proximity diagnostic.
 byframe=defaultdict(list)
 for r in rows:
  if r['source_kind']=='manual_sparse':byframe[r['annotation_frame_uid']].append(r)
 for frame,rs in sorted(byframe.items(),key=lambda x:rank(x[0])):
  hands=[r for r in rs if 'hand' in r['entity_name'].lower()];objects=[r for r in rs if 'hand' not in r['entity_name'].lower()]
  for h in hands:
   if objects:
    o=min(objects,key=lambda r:np.linalg.norm(s['centre_native'][r['row_index']]-s['centre_native'][h['row_index']]))
    if not any(a==h['mask_uid'] and b==o['mask_uid'] for a,b,_,_ in pairs):pairs.append((h['mask_uid'],o['mask_uid'],frame+h['mask_uid'],'same_frame_nearest_object'))
  if len(pairs)>=96:break
 selection=json.loads((path('reports')/'spatial_candidate.json').read_text())['options'];cal=json.loads((ROOT/'derived/track1_refined_v2/calibration/model.json').read_text())['kappa_cov'];cache={};records=[];uncertainty_path=path('reports')/'uncertainty_candidate.json';uncertainty=json.loads(uncertainty_path.read_text()) if uncertainty_path.exists() else None
 def get(uid):
  if uid in cache:return cache[uid]
  r=lookup[uid];m=masks[uid];i=r['row_index'];g,points=geometry(m,r['width'],r['height']);ring,_,_=dominant_ring(m,np.diag([r['width']/m['annotation_width'],r['height']/m['annotation_height'],1.]));y=np.roll(resample(ring,128),-2*r['seam_shift'],axis=0);candidate=spatial_fit(y,selection,reliability=r['provenance_weight']*r['geometry_weight']);a,_=nearest_curve(y,s['controls'][i]);b,_=nearest_curve(y,candidate['controls'])
  if r['spatial_valid'] and (not candidate['valid'] or np.mean(b*b)>np.mean(a*a)):candidate=dict(controls=s['controls'][i],covariance=s['covariance'][i],valid=True)
  cache[uid]=dict(row=r,source_points=y,geometry=g,points=points,baseline=dict(controls=s['controls'][i],covariance=s['covariance'][i],valid=r['spatial_valid']),candidate=candidate)
  return cache[uid]
 for ua,ub,key,source in pairs:
  a,b=get(ua),get(ub);reference_gap=float(a['geometry'].distance(b['geometry']));radius=max(8.,.02*min(np.hypot(a['geometry'].bounds[2]-a['geometry'].bounds[0],a['geometry'].bounds[3]-a['geometry'].bounds[1]),np.hypot(b['geometry'].bounds[2]-b['geometry'].bounds[0],b['geometry'].bounds[3]-b['geometry'].bounds[1])));qa=a['points'][cKDTree(b['points']).query(a['points'])[0]<=reference_gap+radius];qb=b['points'][cKDTree(a['points']).query(b['points'])[0]<=reference_gap+radius]
  if not len(qa) or not len(qb):continue
  for method in ['baseline','candidate']:
   aa=a[method];bb=b[method];ca=periodic_design(np.arange(512)/512)@aa['controls'];cb=periodic_design(np.arange(512)/512)@bb['controls'];ga=make_valid(Polygon(ca));gb=make_valid(Polygon(cb));gap=float(ga.distance(gb));ea,va,da=spatial_errors(qa,aa);eb,vb,db=spatial_errors(qb,bb);ribbon=float(2.24*(np.sqrt(np.median(va)*cal)+np.sqrt(np.median(vb)*cal)))
   extra=dict(normal_coverage95_global_refit=None,normal_coverage95_local=None,local_mean_interval_width_px=None)
   if method=='candidate' and uncertainty is not None:
    from .uncertainty import variance_features
    xa=variance_features(a['source_points'],aa,qa);xb=variance_features(b['source_points'],bb,qb);coeff=np.asarray(uncertainty['local_coefficients']);la=xa@coeff;lb=xb@coeff;extra=dict(normal_coverage95_global_refit=float(np.mean(np.r_[abs(ea)<=1.96*np.sqrt(va*uncertainty['global_kappa']),abs(eb)<=1.96*np.sqrt(vb*uncertainty['global_kappa'])])),normal_coverage95_local=float(np.mean(np.r_[abs(ea)<=1.96*np.sqrt(la),abs(eb)<=1.96*np.sqrt(lb)])),local_mean_interval_width_px=float(np.mean(np.r_[3.92*np.sqrt(la),3.92*np.sqrt(lb)])))
   records.append(dict(video_id=v,split=split(v),pair_uid=key,source=source,hand_mask_uid=ua,target_mask_uid=ub,method=method,reference_gap_px=reference_gap,fitted_gap_px=gap,separation_absolute_error_px=abs(gap-reference_gap),contact_region_boundary_mean_px=float(np.mean(np.r_[da,db])),contact_region_boundary_p95_px=float(np.quantile(np.r_[da,db],.95)),normal_coverage95=float(np.mean(np.r_[abs(ea)<=1.96*np.sqrt(va*cal),abs(eb)<=1.96*np.sqrt(vb*cal)])),approx_ribbon_width_px=ribbon,both_spatial_valid=bool(aa['valid'] and bb['valid']),multi_component=bool(a['row']['component_count']>1 or b['row']['component_count']>1),**extra,**{f'false_proximity_{tol}px':bool(gap<=tol and reference_gap>tol) for tol in [2,5,10]},**{f'missed_proximity_{tol}px':bool(gap>tol and reference_gap<=tol) for tol in [2,5,10]},**{f'ambiguous_ribbon_{tol}px':bool(max(0,gap-ribbon)<=tol<=gap+ribbon) for tol in [2,5,10]}))
 save_index(path('reports')/f'contacts.{v}.parquet',records);return dict(video_id=v,pairs=len(records)//2)


def evaluate(monitor):
 cfg=config();pool(video,cfg['train']+cfg['validation'],monitor,workers=7);records=[]
 for v in cfg['train']+cfg['validation']:records+=pq.read_table(path('reports')/f'contacts.{v}.parquet').to_pylist()
 save_index(path('reports')/'contacts.parquet',records);metrics={}
 keys=['separation_absolute_error_px','contact_region_boundary_mean_px','contact_region_boundary_p95_px','normal_coverage95']+[f'{kind}_{tol}px' for kind in ['false_proximity','missed_proximity','ambiguous_ribbon'] for tol in [2,5,10]]
 for group in ['all','single_component','multi_component']:
  metrics[group]={}
  for method in ['baseline','candidate']:
   byvideo={}
   for v in cfg['train']+cfg['validation']:
    rs=[r for r in records if r['method']==method and r['video_id']==v and (group=='all' or r['multi_component']==(group=='multi_component'))]
    if rs:byvideo[v]={k:float(np.mean([r[k] for r in rs])) for k in keys}
   metrics[group][method]=dict(**{k:float(np.mean([r[k] for r in byvideo.values()])) for k in keys},by_video=byvideo)
 calibrated={};conditional={}
 for group in ['all','single_component','multi_component']:
  selected=[r for r in records if r['method']=='candidate' and (group=='all' or r['multi_component']==(group=='multi_component'))]
  calibrated[group]={k:float(np.mean([r[k] for r in selected if r.get(k) is not None])) for k in ['normal_coverage95_global_refit','normal_coverage95_local','local_mean_interval_width_px'] if any(r.get(k) is not None for r in selected)}
 for method in ['baseline','candidate']:
  conditional[method]={}
  for tol in [2,5,10]:
   selected=[r for r in records if r['method']==method];positive=[r for r in selected if r['reference_gap_px']<=tol];negative=[r for r in selected if r['reference_gap_px']>tol];conditional[method][str(tol)]=dict(reference_positive=len(positive),reference_negative=len(negative),miss_rate=float(np.mean([r[f'missed_proximity_{tol}px'] for r in positive])) if positive else None,false_positive_rate=float(np.mean([r[f'false_proximity_{tol}px'] for r in negative])) if negative else None)
 report('contact_geometry.json',dict(metrics=metrics,calibrated_candidate_bands=calibrated,conditional_proximity_rates=conditional,pair_evaluations=len(records)//2,proximity_definition='Source all-component/hole polygon separation <= 2,5,10 native pixels; not physical contact classification. Source manual contours are encoder inputs, not independently reviewed geometry.',uncertainty_note='v2 empirical kappa reused for a diagnostic only; candidate requires its own train-only calibration. Ribbon gap envelope ignores joint shape correlations and carries no certified contact probability.',test_videos_used=False,**identity()))
