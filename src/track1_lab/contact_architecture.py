"""Matched 2D interaction-separation ablation for single and compound spatial representations."""
from collections import defaultdict
import json
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import make_valid
from pipeline.common import ROOT
from track1.artifacts import load_shard,save_index
from track1.spatial import resample
from refinement.kernels import periodic_design
from .architecture import general_fit,source_geometry
from .common import config,path,allowed,split,pool,report,identity
from .working import directory
from .geometry_contract import select as select_geometry


def video(v):
 allowed([v]);raw=pq.read_table(path('reports')/f'contacts.{v}.parquet').to_pylist();pairs=[r for r in raw if r['method']=='baseline'];wanted={r[k] for r in pairs for k in ['hand_mask_uid','target_mask_uid']};masks={r['mask_uid']:r for r in pq.read_table(ROOT/f'canonical/v1/masks.{v}.parquet').to_pylist() if r['mask_uid'] in wanted};meta={r['mask_uid']:r for r in pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist() if r['mask_uid'] in wanted};old=load_shard(ROOT/f'derived/track1_refined_v2/spatial/{v}.pt');working=load_shard(directory('spatial')/f'{v}.pt');cache={};records=[]
 def get(uid):
  if uid in cache:return cache[uid]
  r=meta[uid];truth,rings=source_geometry(masks[uid]);scale=np.array([r['width']/masks[uid]['annotation_width'],r['height']/masks[uid]['annotation_height']]);main=max([x for x in rings if x['role']=='outer'],key=lambda x:abs(x['signed_area']));result={'truth':truth,'multi':r['component_count']>1}
  for method,state in [('v2_16',old),('working_16',working)]:result[method]=make_valid(Polygon(periodic_design(np.arange(512)/512)@state['controls'][r['row_index']]/scale))
  for method in ['single_32','compound_32']:
   chosen=[main] if method=='single_32' else select_geometry(rings,3);allocation=[32] if len(chosen)==1 else [24,8] if len(chosen)==2 else [16,8,8];pieces=[]
   for ring,m in zip(chosen,allocation):
    fitted=general_fit(resample(ring['points'],128),m=m);p=make_valid(Polygon(periodic_design(np.arange(512)/512,m=m)@fitted['controls']));pieces.append((ring,p))
   shape=Polygon()
   for ring,p in sorted(pieces,key=lambda x:x[0]['depth']):shape=shape.union(p) if ring['role']=='outer' else shape.difference(p)
   result[method]=shape
  cache[uid]=result;return result
 for r in pairs:
  a,b=get(r['hand_mask_uid']),get(r['target_mask_uid']);reference=float(a['truth'].distance(b['truth']))
  for method in ['v2_16','working_16','single_32','compound_32']:
   gap=float(a[method].distance(b[method]));records.append(dict(video_id=v,split=split(v),pair_uid=r['pair_uid'],method=method,reference_gap_annotation_px=reference,predicted_gap_annotation_px=gap,absolute_gap_error_px=abs(gap-reference),multi_component=a['multi'] or b['multi'],**{f'false_positive_{tol}px':bool(gap<=tol and reference>tol) for tol in [2,5,10]},**{f'missed_{tol}px':bool(gap>tol and reference<=tol) for tol in [2,5,10]}))
 save_index(path('reports')/f'architecture_contact.{v}.parquet',records);return dict(video_id=v,pairs=len(pairs))


def run(monitor):
 cfg=config();pool(video,cfg['train']+cfg['validation'],monitor,workers=5);records=[]
 for v in cfg['train']+cfg['validation']:records+=pq.read_table(path('reports')/f'architecture_contact.{v}.parquet').to_pylist()
 save_index(path('reports')/'architecture_contact.parquet',records);summary={}
 for group in ['all','single_component','multi_component']:
  summary[group]={}
  for method in ['v2_16','working_16','single_32','compound_32']:
   rs=[r for r in records if r['method']==method and r['split']=='validation' and (group=='all' or r['multi_component']==(group=='multi_component'))];byvideo={v:np.mean([r['absolute_gap_error_px'] for r in rs if r['video_id']==v]) for v in cfg['validation'] if any(r['video_id']==v for r in rs)};rates={}
   for tol in [2,5,10]:
    pos=[r for r in rs if r['reference_gap_annotation_px']<=tol];neg=[r for r in rs if r['reference_gap_annotation_px']>tol];rates[str(tol)]=dict(reference_positive=len(pos),reference_negative=len(neg),miss_rate=float(np.mean([r[f'missed_{tol}px'] for r in pos])) if pos else None,false_positive_rate=float(np.mean([r[f'false_positive_{tol}px'] for r in neg])) if neg else None)
   summary[group][method]=dict(pairs=len(rs),mean_separation_error_px=float(np.mean(list(byvideo.values()))) if byvideo else None,conditional_proximity_rates=rates)
 report('architecture_contact.json',dict(validation_metrics=summary,pairs=len(records)//4,limits=['Matched source-annotated pairs, not independent physical contact truth.','Distances use annotation pixels; manually annotated pilot frames use the native grid.','Compound representation uses a fixed total of 32 controls across at most three rings. Temporal loop correspondence and uncertainty calibration remain open.'],state='architecture_ablation_not_finalized',**identity()))
