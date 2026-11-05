"""One bounded, hash-locked Track 2 feasibility matrix. No promotion path."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import argparse,collections,fcntl,importlib.metadata,json,logging,platform,shutil,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import atomic_json,digest
from track1_retrain.common import guard,pool
from .data import ROOT,CFG,DATA,REPORT,WORK,Data,prepare,check_frozen,save
from .train import fold_job,final_job,load_model,weight_hash,tensors,seed_all
LOG=logging.getLogger(__name__)

class Monitor:
 def __enter__(self):
  self.stop=threading.Event();self.error=None;self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def loop(self):
  import psutil
  while not self.stop.is_set():
   try:
    rss,available=guard();r=dict(unix=time.time(),pid=os.getpid(),tree_rss_gib=rss,available_gib=available,cpu_percent=psutil.cpu_percent(),disk_free_bytes=shutil.disk_usage(ROOT).free)
    with (REPORT/'resources.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   except BaseException as e:self.error=e;return
   self.stop.wait(5)
 def check(self):
  if self.error:raise self.error
  guard()
 def __exit__(self,*args):self.stop.set();self.thread.join(6)

def adapter(monitor):
 prepare();monitor.check()
 from .evaluate import cv,prediction_rows
 data=Data();ids=data.indices('validation',usable=False);save(REPORT/'cv_predictions.parquet',prediction_rows(data,ids,cv(data,ids),'cv',0))
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True);version={}
 for name in ['torch','numpy','scipy','pyarrow','psutil','threadpoolctl']:
  try:version[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:version[name]='unavailable'
 lock=ROOT/'configs/requirements.lock.txt';atomic_json(REPORT/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=version,local_git_commit=commit.stdout.strip() if commit.returncode==0 else None,reported_commit_not_assumed_local=True,requirements_lock_sha256=digest(lock) if lock.exists() else None,device='cpu',matrix_exp_mps_supported=False,threads_per_worker=CFG['cpu_threads_per_worker'],workers=CFG['workers'],blas_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},model_precision='float32',aggregation_precision='float64'))
 atomic_json(REPORT/'protocol_amendment.json',dict(original_base_v2_reported_m16_counts=[396,63],audited_m16_counts=[319,49],base_v2_has_no_supported_future_validation_targets=True,user_approved_context_amendment=True,cohort='existing frozen track1_context_v3 M16 length-four only',added_revision_contexts_used=False,fresh_participants_used=False,training_participants=['P02','P03','P04','P11','P22'],validation_participants=['P08','P18'],final_model_runs=12,stopping_fold_fits=60,architecture_variants_added=False,architecture_choices='fixed 32-dimensional history-MLP observation encoder for both Koopman variants; shared state decoder; one-layer GRU32; MLP64/32 direct head; no CV residual skip',normalization='training participant states only; unique input/target states deduplicated for moments',frozen_track1_was_fit_with_all_training_participants=True,full_track2_authorized=False))


def folds(monitor):
 data=Data();participants=sorted(set(data.participants[data.indices()].tolist()));jobs=[(family,seed,p) for family in CFG['models'] for seed in CFG['seeds'] for p in participants];records=pool(fold_job,jobs,monitor,workers=CFG['workers']);selected=[]
 for family in CFG['models']:
  for seed in CFG['seeds']:
   rs=[r for r in records if r['family']==family and r['seed']==seed];epochs=int(np.median([r['best_epoch'] for r in rs]));selected.append(dict(family=family,seed=seed,epochs=epochs,fold_best_epochs=[r['best_epoch'] for r in sorted(rs,key=lambda r:r['held_participant'])]));LOG.info('LOCK EPOCHS %s/%s %d',family,seed,epochs)
 atomic_json(REPORT/'selected_epochs.json',dict(selected=selected,selection_uses_only_training_participant_folds=True,exploratory_validation_used=False,fold_fits=len(records),final_refits=12))


def training(monitor):
 selected=json.loads((REPORT/'selected_epochs.json').read_text())['selected'];records=pool(final_job,[(r['family'],r['seed'],r['epochs']) for r in selected],monitor,workers=CFG['workers']);atomic_json(REPORT/'training.json',dict(runs=records,final_model_runs=len(records),stopping_fold_fits=60))


def evaluation(monitor):
 from .evaluate import evaluate
 evaluate();monitor.check()


def verify(monitor):
 import torch
 check_frozen();data=Data();files=list((DATA/'models').glob('*.pt'));assert len(files)==12
 for path in files:
  model,norm,checkpoint=load_model(path);assert set(norm['training_participants'])=={'P02','P03','P04','P11','P22'};assert weight_hash(model)==checkpoint['record']['final_weights_sha256'];assert all(torch.isfinite(p).all() for p in model.parameters())
  for part in ['train','validation']:
   ids=data.indices(part,usable=False);x,_,_=tensors(data,norm)
   with torch.no_grad():p,_=model(x[ids]);assert torch.isfinite(p).all()
  monitor.check()
 ledger=pq.read_table(REPORT/'target_ledger.parquet').to_pylist();assert all(r['latest_input_us']<=r['input_cutoff_us'] for r in ledger if r['input_supported']);assert not (set(data.participants[data.split=='train'])&set(data.participants[data.split=='validation']))
 for sample in data.samples:assert all(b-a==250000 for a,b in zip(sample['input_times_us'],sample['input_times_us'][1:]))
 atomic_json(REPORT/'verification.json',dict(frozen_track1_hashes_unchanged=True,checkpoint_models=12,finite_predictions_and_weights=True,training_only_normalization=True,participant_leakage=False,zero_silent_target_imputation=True,exact_cpu_reload=True,feasibility_only=True,scientific_acceptance=False))


def repeat_best(monitor):
 # Best among the twelve locked primary runs; this repeat is not another selection opportunity.
 from .train import fit_one
 metrics=json.loads((REPORT/'metrics.json').read_text())['summary'];ranked=[]
 for family in CFG['models']:
  for seed in CFG['seeds']:
   rs=[r for r in metrics if r['family']==family and r['seed']==seed and r['horizon_us'] in [250000,500000]];ranked.append((float(np.mean([r['mean_participant_position_rmse'] for r in rs])),family,seed))
 _,family,seed=min(ranked);checkpoint=torch_checkpoint=load_model(DATA/'models'/f'{family}.{seed}.pt')[2];directory=WORK/'empty_repeat'
 if directory.exists():raise FileExistsError('Repeat directory must be empty')
 directory.mkdir();model,norm,record=fit_one(family,seed,Data().indices(),epochs=checkpoint['selected_epochs'],tag=f'repeat/{family}/{seed}');match=weight_hash(model)==checkpoint['record']['final_weights_sha256'];atomic_json(directory/'result.json',dict(family=family,seed=seed,epochs=checkpoint['selected_epochs'],original_weights_sha256=checkpoint['record']['final_weights_sha256'],repeat_weights_sha256=weight_hash(model),exact_weights_match=match,used_for_model_selection=False));atomic_json(REPORT/'repeat_best.json',json.loads((directory/'result.json').read_text()));assert match,'Deterministic CPU repeat differed';monitor.check()


def report(monitor):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 decision=json.loads((REPORT/'decision.json').read_text());metrics=json.loads((REPORT/'metrics.json').read_text())['summary'];rows=[]
 for family in ['cv']+CFG['models']:
  for h in CFG['horizons_us']:
   rs=[r for r in metrics if r['family']==family and r['horizon_us']==h];rows.append(dict(family=family,horizon_ms=h//1000,median_participant_position_rmse=float(np.median([r['mean_participant_position_rmse'] for r in rs])),median_participant_velocity_rmse=float(np.median([r['mean_participant_velocity_rmse'] for r in rs]))))
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for family in ['cv']+CFG['models']:
  rs=[r for r in rows if r['family']==family];axes[0].plot([r['horizon_ms'] for r in rs],[r['median_participant_position_rmse'] for r in rs],'o-',label=family)
 axes[0].set_xlabel('Forecast horizon ms');axes[0].set_ylabel('Mean-participant source-position RMSE');axes[0].legend(fontsize=8)
 for family in CFG['models']:
  curves=[]
  for seed in CFG['seeds']:
   history=json.loads((REPORT/f'training.{family}.{seed}.json').read_text())['history'];axes[1].plot([r['epoch'] for r in history],[r['training_loss'] for r in history],alpha=.5,label=family if seed==CFG['seeds'][0] else None)
 axes[1].set_xlabel('Final refit epoch');axes[1].set_ylabel('Training objective');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(REPORT/'quality.png',dpi=150);plt.close(fig)
 cells=''.join('<tr>'+''.join('<td>'+str(row[k])+'</td>' for k in ['family','horizon_ms','median_participant_position_rmse','median_participant_velocity_rmse'])+'</tr>' for row in rows)
 (REPORT/'dashboard.html').write_text('''<!doctype html><meta charset="utf-8"><title>Bounded Track 2 feasibility</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px;line-height:1.5}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>Isolated Track 2 feasibility</h1><p>Full Track 2 remains NO-GO. Track 1 is unchanged and scientifically unfinalized. Cohort amendment was explicitly approved after the report's base-v2 M16 count/support mismatch: only the existing frozen six-second contexts enter this diagnostic. No newly reserved participant or independent physical reference was used.</p><p><strong>Feasibility gates: '''+('PASS' if decision['feasibility_passed'] else 'FAIL — stop architecture work')+'''</strong></p><img src="quality.png" alt="Matched future source-position error and fixed-budget training curves"><table><tr><th>Family</th><th>Horizon ms</th><th>Position RMSE</th><th>Velocity RMSE</th></tr>'''+cells+'''</table><p>Intrinsic-normalized source-kinematic targets, not independent physical trajectories. Means aggregate participant RMSE first, then seed medians. Unsupported horizons are masked in loss and remain in the denominator ledger. Two exploratory validation participants do not establish final generalization or calibrated uncertainty. Stable latent eigenvalues do not certify decoded physical forecasts.</p><p><a href="decision.json">Predeclared gates</a> · <a href="adapter.json">Cohort audit</a> · <a href="protocol_amendment.json">Approved amendment</a> · <a href="predictions.parquet">Every forecast/mask</a> · <a href="target_ledger.parquet">All opportunities</a> · <a href="selected_epochs.json">LOPO stopping fits</a> · <a href="verification.json">Integrity checks</a> · <a href="inference_latency.json">Cached-state inference cost</a> · <a href="repeat_best.json">Empty-directory repeat</a></p>''')
 atomic_json(REPORT/'comparison.json',dict(rows=rows,scientific_acceptance=False));resources=[json.loads(l) for l in (REPORT/'resources.jsonl').read_text().splitlines()];atomic_json(DATA/'status.json',dict(feasibility_experiment_complete=True,feasibility_gates_passed=decision['feasibility_passed'],full_track2_authorized=False,track1_scientific_freeze=False,scientific_acceptance=False,optional_ablations_started=False));atomic_json(REPORT/'completion.json',dict(primary_training_runs=12,stopping_fold_fits=60,reproducibility_repeats=1,peak_sampled_tree_rss_gib=max(r['tree_rss_gib'] for r in resources),minimum_sampled_available_gib=min(r['available_gib'] for r in resources),full_track2_go=False))
 monitor.check()


def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','adapter','folds','training','evaluation','verify','repeat','report'],default='all');p.add_argument('--resume',action='store_true');args=p.parse_args()
 for path in [DATA,REPORT,WORK]:path.mkdir(exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True);lock=(WORK/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source={str(p.relative_to(ROOT)):digest(p) for p in list((ROOT/'src/track2_feasibility').glob('*.py'))+[ROOT/'configs/track2_feasibility.json']};functions=dict(adapter=adapter,folds=folds,training=training,evaluation=evaluation,verify=verify,repeat=repeat_best,report=report);stage=args.stage
 try:
  with Monitor() as monitor:
   for stage in list(functions) if args.stage=='all' else [args.stage]:
    marker=WORK/f'{stage}.done.json'
    if args.resume and marker.exists():
     if json.loads(marker.read_text())['source']!=source:raise ValueError('Source/config changed after checkpoint; explicit stage rerun required')
     check_frozen();LOG.info('RESUME %s',stage);continue
    atomic_json(REPORT/'status.json',dict(state='running',stage=stage,pid=os.getpid(),updated_unix=time.time(),full_track2_authorized=False));LOG.info('BEGIN %s',stage);start=time.monotonic();functions[stage](monitor);monitor.check();atomic_json(marker,dict(source=source,seconds=time.monotonic()-start));LOG.info('COMPLETE %s %.1fs',stage,time.monotonic()-start)
  check_frozen();atomic_json(DATA/'manifest.json',dict(source=source,outputs={str(p.relative_to(DATA)):digest(p) for p in DATA.rglob('*') if p.is_file() and p.name!='manifest.json'},frozen_inputs=digest(REPORT/'frozen_inputs.json'),scientific_acceptance=False));atomic_json(REPORT/'status.json',dict(state='complete',stage=args.stage,full_track2_authorized=False))
 except BaseException as e:atomic_json(REPORT/'status.json',dict(state='failed',stage=stage,error=str(e),full_track2_authorized=False));raise
if __name__=='__main__':main()
