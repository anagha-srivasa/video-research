"""Resumable mathematical training, full refit, validation and ingestion checks."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import argparse
import collections
import copy
import fcntl
import logging
import threading
import time
import numpy as np
from .data import *
from .fit import candidates,search_worker,choose_family,grouped_mean,mask_worker,detail_worker,calibration_worker,evaluate_worker,FAMILIES
from track1_retrain.common import guard,pool
LOG=logging.getLogger(__name__)

class Monitor:
 def __enter__(self):
  self.stop=threading.Event();self.error=None;self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def loop(self):
  import psutil,shutil
  while not self.stop.is_set():
   try:
    rss,available=guard();r=dict(unix=time.time(),pid=os.getpid(),tree_rss_gib=rss,available_gib=available,cpu_percent=psutil.cpu_percent(),disk_free_bytes=shutil.disk_usage(ROOT).free)
    with (R/'resources.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   except BaseException as e:self.error=e;return
   self.stop.wait(5)
 def check(self):
  if self.error:raise self.error
  guard()
 def __exit__(self,*args):self.stop.set();self.thread.join(6)

def search(monitor):
 base=baseline();trained={};summary=[]
 for n in CFG['models']:
  original=base[str(n)];trials=candidates(original['options']);jobs=pool(search_worker,[(v,n,trials,original['feature_scale']) for v in OLD_CFG['train']],monitor,workers=CFG['workers']);records=[r for j in jobs for r in j['records']];save(R/f'search.m{n}.parquet',records);save(R/f'search_attempts.m{n}.parquet',[r for j in jobs for r in j['attempts']]);options={k:v for k,v in original['options'].items() if k!='families'};families={};decisions=[]
  for family in FAMILIES:
   index,decision=choose_family(records,family);candidate=trials[index];families[family]={k:v for k,v in candidate.items() if k!='families'};families[family].update(candidate.get('families',{}).get(family,{}));decisions.append(decision)
  options['families']=families
  # Confirm shape in mask space on the same fixed training probes. Never choose on validation.
  admissible=next(d['admissible_trials'] for d in decisions if d['family']=='shape')
  top=sorted(admissible,key=lambda i:grouped_mean(records,'shape_mse',i))[:3]
  shape_choices=[copy.deepcopy(options)]
  for index in sorted(set([0]+top)):
   choice=copy.deepcopy(options);candidate=trials[index];local={k:v for k,v in candidate.items() if k!='families'};local.update(candidate.get('families',{}).get('shape',{}));choice['families']['shape']=local;shape_choices.append(choice)
  # The first mask candidate uses the safeguarded position/scale and baseline shape.
  baseline_shape={k:v for k,v in original['options'].items() if k!='families'};baseline_shape.update(original['options'].get('families',{}).get('shape',{}))
  baseline_choice=copy.deepcopy(options);baseline_choice['families']['shape']=baseline_shape;shape_choices.insert(0,baseline_choice)
  mask_jobs=pool(mask_worker,[(v,n,shape_choices,original['feature_scale']) for v in OLD_CFG['train']],monitor,workers=CFG['workers']);mask_records=[r for rs in mask_jobs for r in rs];save(R/f'mask_selection.m{n}.parquet',mask_records)
  mask_scores=[grouped_mean(mask_records,'iou',i) for i in range(len(shape_choices))];acceptable=[0]
  for i in range(1,len(shape_choices)):
   if all(grouped_mean([r for r in mask_records if r['video_id']==v],'iou',i)>=grouped_mean([r for r in mask_records if r['video_id']==v],'iou',0)*(1-CFG['maximum_training_family_regret']) for v in OLD_CFG['train']):acceptable.append(i)
  best=max(acceptable,key=lambda i:mask_scores[i])
  if mask_scores[best]<mask_scores[0]*(1+CFG['minimum_relative_improvement']):best=0
  options=shape_choices[best]
  # Baseline detail is always in the admissible search, along with a zero-detail core.
  details=[dict(count=0,strength=0.)]+[dict(count=c,strength=s) for c in [2,3] for s in [.003,.01,.1,1.]]
  d_jobs=pool(detail_worker,[(v,n,options,original['feature_scale'],details) for v in OLD_CFG['train']],monitor,workers=CFG['workers']);d_records=[r for j in d_jobs for r in j['records']];save(R/f'detail_selection.m{n}.parquet',d_records);di,dd=choose_family(d_records,'position');detail=details[di]
  ratios=pool(calibration_worker,[(v,n,options,original['feature_scale']) for v in OLD_CFG['train']],monitor,workers=CFG['workers']);quantiles=[]
  for protocol in ['all','endpoint100']:
   arrays=[]
   for job in ratios:
    values=[a for p,a in job['ratios'].items() if protocol=='all' or p==protocol]
    if values:arrays.append(np.concatenate(values))
   length=min(len(a) for a in arrays);balanced=np.concatenate([a[np.linspace(0,len(a)-1,length,dtype=int)] for a in arrays]);quantiles.append(np.nanquantile(balanced,.95,axis=0)/1.9599639845**2)
  multiplier=np.clip(np.nan_to_num(np.maximum(*quantiles),nan=1.,posinf=1e6),1e-4,1e6)
  trained[str(n)]=dict(options=options,detail=detail,feature_scale=original['feature_scale'],multipliers=multiplier.tolist(),selection='training participants only; per-participant/protocol 2% MSE regret guard; matched training mask confirmation',covariance_scope='post-selection empirical held-out training-source marginals; no independent physical calibration',detail_covariance_calibrated=False)
  summary.append(dict(model=n,family_decisions=decisions,mask_baseline_iou=mask_scores[0],mask_selected_iou=mask_scores[best],mask_choice=best,detail=detail,detail_decision=dd,trial_options=trials,training_videos=OLD_CFG['train']))
  LOG.info('M%d TRAINED families=%s detail=%s mask %.5f -> %.5f',n,{k:{x:y for x,y in v.items() if x in ['regularization','loss','variance_power']} for k,v in options['families'].items()},detail,mask_scores[0],mask_scores[best])
  atomic_json(R/'selection_progress.json',dict(models=summary))
 for source in BASES:atomic_json(D/source/'calibration/model.json',trained)
 atomic_json(R/'selection.json',dict(models=summary,selected=trained,validation_used_for_selection=False,validation_is_exploratory=True,independent_accuracy_validated=False))

def evaluate(monitor):
 model=json.loads((D/'v2/calibration/model.json').read_text());jobs=pool(evaluate_worker,[(v,n,model[str(n)]) for v in VIDEOS for n in CFG['models']],monitor,workers=CFG['workers']);records=[r for j in jobs for r in j['records']];attempts=[r for j in jobs for r in j['attempts']];save(R/'evaluation.parquet',records);save(R/'attempts.parquet',attempts);summary=[]
 for part in ['train','validation']:
  for n in CFG['models']:
   for protocol in ['single','gap100','endpoint100']:
    for method in sorted({r['method'] for r in records}):
     rs=[r for r in records if (r['split'],r['model'],r['protocol'],r['method'])==(part,n,protocol,method)];means=[]
     for v in sorted({r['video_id'] for r in rs}):
      local=[r for r in rs if r['video_id']==v];entry={}
      for key in ['position_mse','shape_mse','scale_mse','mask_iou','boundary_fscore','position_marginal95_coverage','position_diagonal_nll']:
       values=[r[key] for r in local if r.get(key) is not None];entry[key]=float(np.sqrt(np.mean(values))) if values and key.endswith('_mse') else float(np.mean(values)) if values else None
      means.append(entry)
     if means:summary.append(dict(split=part,model=n,protocol=protocol,method=method,probes=len(rs),**{k:float(np.mean([m[k] for m in means if m[k] is not None])) if any(m[k] is not None for m in means) else None for k in means[0]}))
 for row in summary:
  for family in FAMILIES:row[family+'_nrmse']=row.pop(family+'_mse')
 atomic_json(R/'evaluation.json',dict(summary=summary,rmse_note='Summary fields are mean-video normalized RMSE; per-probe parquet fields are MSE',attempts=len(attempts),supported=sum(r['supported'] for r in attempts),aggregation='mean of per-video RMSE, same probes for all methods',validation_exploratory=True,selection_already_locked=True,scientific_acceptance=False))

def build_job(job):
 from track1_retrain.temporal import build_worker
 source,v,n=job;root=D/source;report=R/source;report.mkdir(exist_ok=True)
 receipt=report/f'build_receipt.{v}.m{n}.json';fp=dict(model=digest(root/'calibration/model.json'),projection=digest(root/'appearance'/f'{v}.projected.npy'),episodes=digest(EPISODES[source]))
 if receipt.exists():
  old=json.loads(receipt.read_text())
  if old['inputs']==fp and all((ROOT/p).exists() and digest(ROOT/p)==h for p,h in old['outputs'].items()):return old['summary']
 result=build_worker((v,n,dict(root=str(root),reports=str(report),episodes=str(EPISODES[source]))))
 outputs={str(p.relative_to(ROOT)):digest(p) for p in (root/'functional').glob(f'{v}.m{n}.*')}
 atomic_json(receipt,dict(inputs=fp,outputs=outputs,summary=result));return result

def build(monitor):
 results=pool(build_job,[(source,v,n) for source in BASES for v in VIDEOS for n in CFG['models']],monitor,workers=CFG['workers']);atomic_json(R/'refit.json',dict(videos=results))

def verify(monitor):
 from .store import build_indices,verify_tensors,FinetunedDataset
 counts=build_indices();checks=verify_tensors(monitor)
 input_manifest=json.loads((R/'input_manifest.json').read_text())
 for p,h in input_manifest['dependencies'].items():assert digest(ROOT/p)==h,('Baseline/input modified',p)
 smoke=[]
 with FinetunedDataset(require_ready=False) as ds:
  for part in ['train','validation']:
   for length in [1,4,8,16]:
    rows=pq.read_table(D/'indices'/f'{part}.length{length}.parquet').to_pylist()
    for source in BASES:
     local=[r for r in rows if r['source']==source]
     if local:
      b=ds.get_sequence(local[0]);assert b['coefficients'].shape==(length,4,35);assert b['xi'].shape==(length,4,68);assert b['appearance'].shape==(length,4,2,128)
      smoke.append(dict(split=part,length=length,source=source))
 result=dict(checks=checks,indices=counts,smoke=smoke,baseline_hashes_unchanged=True,scientific_acceptance=False,track2_started=False)
 atomic_json(R/'verification.json',result)
 atomic_json(D/'status.json',dict(training_complete=True,numerical_ingestion_ready=True,ready_scope='Open Track 1 mathematical refit on prepared retrospective source observations',backbone_frozen=True,uncertainty_independently_validated=False,scientific_acceptance=False,finalized=False,track2_started=False))


def report():
 import html
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 ev=json.loads((R/'evaluation.json').read_text());selection=json.loads((R/'selection.json').read_text());v=json.loads((R/'verification.json').read_text());pca=json.loads((R/'pca.json').read_text())
 rows=[r for r in ev['summary'] if r['split']=='validation' and r['method'] in ['prepared_baseline','finetuned_detail']]
 fig,axes=plt.subplots(1,3,figsize=(15,4));labels=[f"M{n} {pr}" for n in [16,32] for pr in ['single','gap100','endpoint100']]
 for ax,key,title in zip(axes,['position_nrmse','shape_nrmse','mask_iou'],['Position NRMSE','Shape NRMSE','Source-mask IoU']):
  for offset,method in [(-.18,'prepared_baseline'),(.18,'finetuned_detail')]:
   values=[next(r[key] for r in rows if r['model']==n and r['protocol']==pr and r['method']==method) for n in [16,32] for pr in ['single','gap100','endpoint100']]
   ax.bar(np.arange(6)+offset,values,.36,label=method)
  ax.set_xticks(range(6),labels,rotation=55,ha='right');ax.set_title(title)
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(R/'quality.png',dpi=150);plt.close(fig)
 cells=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k])) if not isinstance(r[k],float) else format(r[k],".5f")}</td>' for k in ['model','protocol','method','position_nrmse','shape_nrmse','scale_nrmse','mask_iou','boundary_fscore'])+'</tr>' for r in rows)
 page='''<!doctype html><meta charset="utf-8"><title>Track 1 mathematical training</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px;line-height:1.5}td,th{padding:8px;border-bottom:1px solid #ccc}table{border-collapse:collapse}img{width:100%}</style><h1>Track 1 mathematical training and refinement</h1><p>Open development. Track 2 remains on hold. Parameters and PCA were fitted using training participants only. Validation has been used repeatedly in prior research and remains exploratory. These are withheld source-reconstruction tests; no independent physical motion/contact/identity or unseen-RGB segmentation claim is made.</p><img src="quality.png" alt="Paired validation reconstruction scores"><h2>Matched exploratory validation</h2><table><tr><th>Controls</th><th>Holdout</th><th>Method</th><th>Position NRMSE</th><th>Shape NRMSE</th><th>Scale NRMSE</th><th>Mask IoU</th><th>BF 2 annotation px</th></tr>'''+cells+'''</table><p>Same retained observations and targets for each method. Nearest/linear baselines and all attempted probes remain in the Parquet evidence. Missing/unsupported probes are reported separately. Shape errors require target compatibility; mask metrics retain topology-change targets. Position residuals remain endpoint-neutral. Separate local-CV kinematics remain authoritative for the declared source-position estimator.</p><p><a href="evaluation.json">Full metrics</a> · <a href="selection.json">Training selection</a> · <a href="pca.json">Train-only PCA</a> · <a href="verification.json">Numerical checks/support</a> · <a href="input_manifest.json">Input provenance</a></p>'''
 (R/'dashboard.html').write_text(page)
 resources=[json.loads(line) for line in (R/'resources.jsonl').read_text().splitlines()]
 atomic_json(R/'completion.json',dict(training_complete=True,training_method='finite regularization/reliability search and exact weighted float64 solves, train-only PCA and empirical covariance calibration',pca=pca,verification=v,peak_sampled_tree_rss_gib=max(r['tree_rss_gib'] for r in resources),minimum_sampled_available_gib=min(r['available_gib'] for r in resources),no_neural_backbone_finetuning=True,scientific_acceptance=False,track2_started=False))
 outputs={str(p.relative_to(ROOT)):digest(p) for p in D.rglob('*') if p.is_file() and not p.is_symlink() and p.name!='manifest.json'}
 atomic_json(D/'manifest.json',dict(outputs=outputs,source=source_identity(),prepared_manifest=digest(P/'manifest.json'),scientific_acceptance=False))

def source_identity():return {str(p.relative_to(ROOT)):digest(p) for p in list((ROOT/'src/track1_finetune').glob('*.py'))+[ROOT/'configs/track1_finetune.json']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','inputs','pca','search','evaluate','build','verify','report'],default='all');p.add_argument('--resume',action='store_true');a=p.parse_args()
 for path in [D,R,W]:path.mkdir(parents=True,exist_ok=True)
 lock=(W/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 functions=dict(inputs=check_inputs,pca=train_pca,search=search,evaluate=evaluate,build=build,verify=verify,report=report)
 stages=list(functions) if a.stage=='all' else [a.stage];identity=source_identity();started=time.monotonic()
 with Monitor() as monitor:
  for stage in stages:
   marker=W/f'{stage}.done.json'
   if a.resume and marker.exists():
    old=json.loads(marker.read_text())
    if old['source']==identity:LOG.info('RESUME completed stage %s',stage);continue
    raise ValueError('Code/config changed after checkpoint; explicit stage rerun required')
   atomic_json(R/'status.json',dict(state='running',stage=stage,pid=os.getpid(),updated_unix=time.time(),track2_started=False));LOG.info('BEGIN %s',stage)
   if stage in ['inputs','report']:functions[stage]()
   else:functions[stage](monitor)
   monitor.check();atomic_json(marker,dict(source=identity,completed_unix=time.time()));LOG.info('COMPLETE %s elapsed %.1fs',stage,time.monotonic()-started)
 atomic_json(R/'status.json',dict(state='complete' if a.stage=='all' else 'stage_complete',stage=a.stage,updated_unix=time.time(),track2_started=False))
if __name__=='__main__':main()
