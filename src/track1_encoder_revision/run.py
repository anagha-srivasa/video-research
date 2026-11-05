"""Execute report fixes without replacing baselines or opening fresh locked labels."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import argparse,collections,fcntl,hashlib,json,logging,subprocess,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json,digest
from track1_finetune.run import Monitor as BaseMonitor
from track1_retrain.common import pool
from . import experiments as exp
from .experiments import D,R,W,CFG,VIDEOS,worker,save
LOG=logging.getLogger(__name__)

class Monitor(BaseMonitor):
 def loop(self):
  import psutil,shutil
  from track1_retrain.common import guard
  while not self.stop.is_set():
   try:
    rss,available=guard();row=dict(unix=time.time(),pid=os.getpid(),tree_rss_gib=rss,available_gib=available,cpu_percent=psutil.cpu_percent(),disk_free_bytes=shutil.disk_usage(ROOT).free)
    with (R/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   except BaseException as e:self.error=e;return
   self.stop.wait(5)

def inputs(monitor):
 parent=json.loads((ROOT/'derived/track1_finetuned/manifest.json').read_text())
 for path,h in parent['outputs'].items():assert digest(ROOT/path)==h,path
 paths=[ROOT/'work/track1_weights/mobilenet_v3_small-047dcff4.pth',ROOT/'derived/track1_finetuned/pca_mean.npy',ROOT/'derived/track1_finetuned/pca_components.npy',ROOT/'src/track1_lab/topology_certificate.py',ROOT/'src/track1_remediation/motion.py',ROOT/'src/track1_finetune/store.py',ROOT/'derived/track1_finetuned/manifest.json']
 for v in VIDEOS:
  expected=pq.read_table(ROOT/f'canonical/v1/videos.{v}.parquet',columns=['sha256']).to_pylist()[0]['sha256'];video_path=ROOT/f'dataset/videos/{v}.MP4';assert digest(video_path)==expected,('Raw video changed',v);paths.extend([video_path,ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json'])
  paths.extend(ROOT/f'canonical/{generation}/{kind}.{v}.parquet' for generation,kind in [('v2','spatial_metadata'),('v2','masks'),('v1','frames')])
 for folder in ['A','B','adjudicated']:
  paths.extend((ROOT/f'references/track1/{folder}').glob('*.json'))
 hashes={str(p.relative_to(ROOT)):digest(p) for p in paths};atomic_json(R/'input_manifest.json',dict(hashes=hashes,report_pdf_sha256=digest(Path('/Users/anagha/Downloads/Track 1 Deep Repository Review_ Remaining Scope to Finalise the Encoder.pdf')),frozen_decoder_unchanged=True,finalized=False))
 monitor.check()

def phase(monitor):
 jobs=pool(worker,VIDEOS,monitor,workers=CFG['workers']);records=[r for j in jobs for r in j['records']];attempts=[r for j in jobs for r in j['attempts']];repairs=[r for j in jobs for r in j['repairs']];save(R/'phase.parquet',records);save(R/'attempts.parquet',attempts);save(R/'repair.parquet',repairs);summary=[]
 for part in ['train','validation']:
  for protocol in ['single','gap100','endpoint100']:
   for variant in exp.VARIANTS:
    rs=[r for r in records if (r['split'],r['protocol'],r['variant'])==(part,protocol,variant)];byvideo=collections.defaultdict(list)
    for r in rs:byvideo[r['video_id']].append(r)
    if byvideo:summary.append(dict(split=part,protocol=protocol,variant=variant,probes=len(rs),**{k:float(np.mean([np.mean([r[k] for r in vals]) for vals in byvideo.values()])) for k in ['iou','bf2','position_nrmse','certificate_pass']}))
 strata=[]
 for field in ['rings','entity_type','entity_track_id','target_topology_compatible']:
  groups=collections.defaultdict(list)
  for row in records:groups[(row['split'],row['variant'],row['protocol'],str(row[field]))].append(row)
  for key,local in groups.items():strata.append(dict(stratum=field,split=key[0],variant=key[1],protocol=key[2],value=key[3],probes=len(local),iou=float(np.mean([r['iou'] for r in local])),bf2=float(np.mean([r['bf2'] for r in local])),certificate_pass=float(np.mean([r['certificate_pass'] for r in local]))))
 save(R/'phase_strata.parquet',strata)
 choices=[]
 for variant in exp.VARIANTS:
  allowed=True
  for protocol in ['single','gap100','endpoint100']:
   for v in __import__('track1_retrain.common',fromlist=['CFG']).CFG['train']:
    a=[r for r in records if r['video_id']==v and r['protocol']==protocol and r['variant']=='A_discrete'];b=[r for r in records if r['video_id']==v and r['protocol']==protocol and r['variant']==variant]
    if not a or len(a)!=len(b):continue
    for key in ['iou','bf2','certificate_pass']:
     if np.mean([r[key] for r in b])+1e-12<np.mean([r[key] for r in a]):allowed=False
  train=[r for r in records if r['split']=='train' and r['variant']==variant];choices.append(dict(variant=variant,training_only_guard_passed=allowed,mean_train_iou=float(np.mean([r['iou'] for r in train])) if train else None))
 candidates=[r for r in choices if r['training_only_guard_passed']];choice=max(candidates,key=lambda r:r['mean_train_iou'])['variant'] if candidates else 'A_discrete'
 atomic_json(R/'phase.json',dict(summary=summary,selection=choices,development_choice=choice,baseline_promoted=False,proposed_endpoint_gain_gate_passed=False,all_proposed=len(attempts),supported=sum(r['supported'] for r in attempts),static_guard='retained mask IoU and BF2 may not decrease',validation_used_for_selection=False,oracle_fields_are_target_exposed_diagnostics=True,topology_oracle_note='Spatial-target oracle changes shape and topology together; no independent pure topology attribution is claimed',phase_covariance_calibrated=False))
 paired=[json.loads(r['comparison_json']) for r in repairs];global_pass=sum(x[0]['certificate_pass'] for x in paired);local_pass=sum(x[1]['certificate_pass'] for x in paired);both=[x for x in paired if x[0]['certificate_pass'] and x[1]['certificate_pass']];quality=dict(paired_certified_probes=len(both),mean_heldout_iou_delta=float(np.mean([x[1]['iou']-x[0]['iou'] for x in both])) if both else None,mean_heldout_bf2_delta=float(np.mean([x[1]['bf2']-x[0]['bf2'] for x in both])) if both else None,global_certified=global_pass,local_or_fallback_certified=local_pass,newly_recovered=sum(not x[0]['certificate_pass'] and x[1]['certificate_pass'] for x in paired))
 atomic_json(R/'repair.json',dict(quality=quality,probes=len(repairs),locally_selected=sum(r['selected_method']=='local_projection' for r in repairs),unchanged=sum(r['selected_method']=='unchanged' for r in repairs),global_fallback=sum(r['selected_method']=='global_rollback' for r in repairs),unresolved=sum(r['selected_method']=='unresolved' for r in repairs),global_and_local_same_probes=True,certificate_decoder_modified=False,geometry_covariance_available=False))

def events(monitor):
 from .diagnostics import event_worker,real_event_worker
 from .event_training import trial
 trials=pool(trial,[dict(noise_z=z,window=w,on_samples=n) for z in [2.5,3.5,5.] for w in [.08,.12,.16] for n in [1,2]],monitor,workers=CFG['workers']);eligible=[r for r in trials if r['false_alarms_per_quiet_minute']<=1.];selected=max(eligible,key=lambda r:(r['recall'],-r['false_alarms_per_quiet_minute'])) if eligible else min(trials,key=lambda r:r['false_alarms_per_quiet_minute']);atomic_json(R/'event_parameters.json',dict(parameters=selected['parameters'],trials=trials,selection_split='synthetic seeds 0-59 only',quiet_false_alarm_development_budget_per_minute=1.,final_threshold=False,validation_previously_exposed=True))
 jobs=pool(event_worker,range(90),monitor,workers=CFG['workers']);rows=[r for rs in jobs for r in rs];save(R/'event_synthetic.parquet',rows);summary=[]
 for part in ['synthetic_train','synthetic_validation']:
  for hop in [.125,.25]:
   for method in ['causal_evidence','cv','nearest','switching','innovation']:
    rs=[r for r in rows if (r['split'],r['hop_seconds'],r['method'])==(part,hop,method) and 'events' in r];n=sum(r['events'] for r in rs);quiet=sum(r['quiet_seconds'] for r in rs);delays=[x[0] for r in rs for x in r['detection_delay_intervals_seconds']];summary.append(dict(split=part,hop_seconds=hop,method=method,events=n,recall=sum(r['recalled'] for r in rs)/n if n else None,false_alarms_per_quiet_minute=sum(r['quiet_false_alarms'] for r in rs)*60/quiet if quiet else None,median_detection_delay_seconds=float(np.median(delays)) if delays else None))
 for seed in range(90):
  for hop in [.125,.25]:
   for horizon in [.25,.5]:
    a=next(r for r in rows if r['seed']==seed and r['hop_seconds']==hop and r.get('horizon')==horizon and r['method']=='switching');b=next(r for r in rows if r['seed']==seed and r['hop_seconds']==hop and r.get('horizon')==horizon and r['method']=='switching_plus_innovation');assert a['position_rmse_px']==b['position_rmse_px'] and a['velocity_rmse_px_s']==b['velocity_rmse_px_s']
 real=pool(real_event_worker,VIDEOS,monitor,workers=CFG['workers']);atomic_json(R/'events.json',dict(summary=summary,real_source_diagnostics=real,unchanged_switching_state_verified=True,thresholds_finalized=False,independent_reference=False,quiet_metric='unmatched detected onsets per quiet minute',position_and_250_500ms_continuation_bitwise_identical_with_or_without_event_channel=True))

def covariance(monitor):
 from .diagnostics import covariance_worker
 jobs=pool(covariance_worker,VIDEOS,monitor,workers=CFG['workers']);rows=[r for rs in jobs for r in rs];save(R/'covariance.parquet',rows);atomic_json(R/'covariance.json',dict(observations=len(rows),psd_violations=sum(r['minimum_eigenvalue']< -1e-8 for r in rows),full_control_correlation_propagated=True,centre_scale_cross_covariances_estimated=False,phase_identity_topology_uncertainty_included=False,geometry_after_repair_covariance_available=False,independent_coverage_calibrated=False))

def contexts(monitor):
 from .contexts import plan_worker,refit_worker
 plans=pool(plan_worker,VIDEOS,monitor,workers=CFG['workers']);atomic_json(R/'context_plan.json',dict(videos=plans,new_rgb_bytes=0,selection_uses_validation_predictions=False));results=pool(refit_worker,plans,monitor,workers=CFG['workers']);atomic_json(R/'contexts.json',dict(videos=results,new_rgb_bytes=0,source_graph_stability_is_not_physical_identity=True,appearance_features_pending=True,training_release_expanded=False))

def appearance(monitor):
 from .appearance import run
 run(monitor)

def references(monitor):
 from .references import audit
 atomic_json(R/'independent_references.json',audit());monitor.check()

def verify(monitor):
 from track1_lab.topology_certificate import input_hashes,evaluate_curves
 from .runtime import CompiledGeometry,RevisionDataset
 start=time.perf_counter();checks=0;compiled=None;meta=None;c=None
 for v in VIDEOS:
  rows=pq.read_table(D/f'{v}.parquet').to_pylist();a=np.load(D/f'{v}.geometry.npy',mmap_mode='r')
  assert len(rows)==len(a) and np.isfinite(a).all()
  for r in rows:
   assert r['latest_input_us']<=r['window_end_us'];assert len(r['retained_mask_uids'])>=8
   if r['certificate_valid']:
    fast=CompiledGeometry(a[r['row_index']],r,samples=128);t=r['window_start_us']+.37*r['window_seconds']*1e6;expected=evaluate_curves(a[r['row_index']],r['allocation'],r['roles'],.37,128);actual=fast.evaluate(t)
    for x,y in zip(expected,actual):np.testing.assert_allclose(x,y,atol=1e-11)
    checks+=1
    if compiled is None and r['source_complete']:compiled=fast;meta=r;c=np.array(a[r['row_index']])
  a._mmap.close();monitor.check()
 if compiled is None:raise ValueError('No complete certified pilot state')
 with RevisionDataset() as ds:
  fast=ds.geometry(meta['video_id'],meta['source'],meta['entity_track_id'],meta['window_end_us'],meta['protocol'],meta['variant'],128);fast.evaluate(meta['window_start_us'])
 # Separate offline compilation/proof from warm geometry evaluation and RGB upstream cost.
 timings=[];baseline=[];t=meta['window_start_us']+.37*meta['window_seconds']*1e6
 for _ in range(100):
  s=time.perf_counter();compiled.evaluate(t);timings.append((time.perf_counter()-s)*1000)
  s=time.perf_counter();evaluate_curves(c,meta['allocation'],meta['roles'],.37,128);baseline.append((time.perf_counter()-s)*1000)
 context_checks=0;context_appearance_states=0
 for path in D.glob('*.context_states.parquet'):
  v=path.name.split('.')[0];rs=pq.read_table(path).to_pylist();a=np.load(D/f'{v}.context_geometry.npy',mmap_mode='r');raw=np.load(D/f'{v}.context_coefficients.npy',mmap_mode='r');xi=np.load(D/f'{v}.context_xi.npy',mmap_mode='r');kc=np.load(D/f'{v}.context_kinematic_cov.npy',mmap_mode='r');app=np.load(D/f'{v}.context_appearance.npy',mmap_mode='r');assert np.isfinite(a).all() and np.isfinite(app).all()
  for r in rs:
   if r['valid']:
    from track1_lab.variants import collocation
    assert r['latest_input_us']<=r['window_end_us'];expected=collocation(raw[r['row_index'],:,:2*r['model']+3].astype(float),1.)['xi'];np.testing.assert_allclose(expected,xi[r['row_index'],:,:expected.shape[1]],rtol=1e-4,atol=2e-5)
   if r['kinematics_valid']:
    e=np.linalg.eigvalsh(kc[r['row_index']].astype(float));assert e[0]>=-1e-6*max(abs(e).max(),1e-20)
   if r['geometry_training_eligible']:
    assert r['source_complete'];c0=a[r['row_index'],:,:2*r['model']+3];CompiledGeometry(c0,r,128);context_checks+=1
   if r['appearance_valid']:assert r['valid'];context_appearance_states+=1
  a._mmap.close();app._mmap.close();raw._mmap.close();xi._mmap.close();kc._mmap.close()
 with RevisionDataset() as reader:
  for path in D.glob('*.context_length16.parquet'):
   for index in pq.read_table(path).to_pylist():
    batch=reader.get_context_sequence(index);assert batch['coefficients'].shape==(16,4,35) and batch['appearance'].shape==(16,4,2,128)
 hashes=json.loads((R/'input_manifest.json').read_text())['hashes']
 for p,h in hashes.items():assert digest(ROOT/p)==h,('Immutable input changed',p)
 for p in D.glob('*.shape_covariance.npy'):
  a=np.load(p,mmap_mode='r');assert np.isfinite(a).all();e=np.linalg.eigvalsh(a.astype(float));assert np.all(e[...,0]>=-1e-5*np.maximum(abs(e).max(-1),1e-20));a._mmap.close()
 atomic_json(R/'verification.json',dict(certificate_bindings_and_decoder_parity=checks,baseline_and_reference_hashes_unchanged=True,conditional_covariance_psd_checked=True,context_certificate_checks=context_checks,context_appearance_states=context_appearance_states,finite_tensors=True,scientific_acceptance=False))
 atomic_json(R/'inference_latency.json',dict(scope='warm normalized certified geometry decode for one sampled 128-point-per-ring state; excludes RGB decode, masks, appearance, fitting, and certification',warm_compiled_median_ms=float(np.median(timings)),warm_compiled_p95_ms=float(np.quantile(timings,.95)),existing_decoder_median_ms=float(np.median(baseline)),repeats=100,number_of_rings=len(meta['allocation']),deployment_end_to_end_fps_established=False))
 atomic_json(D/'status.json',dict(engineering_checks_passed=True,pilot_only=True,default_training_release_replaced=False,scientific_acceptance=False,finalized=False,track2_started=False))

def report(monitor):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 phase=json.loads((R/'phase.json').read_text());events=json.loads((R/'events.json').read_text());repair=json.loads((R/'repair.json').read_text());refs=json.loads((R/'independent_references.json').read_text());lat=json.loads((R/'inference_latency.json').read_text())
 fig,axes=plt.subplots(1,3,figsize=(15,4));rows=[r for r in phase['summary'] if r['split']=='validation'];x=np.arange(3)
 for variant in exp.VARIANTS:
  values=[next((r['iou'] for r in rows if r['variant']==variant and r['protocol']==p),np.nan) for p in ['single','gap100','endpoint100']];axes[0].plot(x,values,'o-',label=variant)
 axes[0].set_xticks(x,['single','gap100','endpoint100']);axes[0].set_title('Conditional source-mask IoU');axes[0].legend(fontsize=8)
 for method in ['cv','switching','innovation']:
  rs=[r for r in events['summary'] if r['split']=='synthetic_validation' and r['method']==method];axes[1].plot([r['false_alarms_per_quiet_minute'] for r in rs],[r['recall'] for r in rs],'o-',label=method)
 axes[1].set_xlabel('False onsets / quiet minute');axes[1].set_ylabel('Recall');axes[1].set_title('Exact synthetic diagnostic');axes[1].legend(fontsize=8)
 axes[2].bar(['Existing decode','Precompiled decode'],[lat['existing_decoder_median_ms'],lat['warm_compiled_median_ms']]);axes[2].set_ylabel('Median ms');axes[2].set_title('Geometry only; RGB upstream excluded');fig.tight_layout();fig.savefig(R/'revision.png',dpi=150);plt.close(fig)
 cells=''.join('<tr>'+''.join('<td>'+str(r[k])+'</td>' for k in ['protocol','variant','probes','iou','bf2','certificate_pass'])+'</tr>' for r in rows)
 (R/'dashboard.html').write_text('''<!doctype html><meta charset="utf-8"><title>Track 1 encoder report fixes</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px;line-height:1.5}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>Track 1 research-report implementation</h1><p>Open development. The finetuned M16 carrier and existing certificate decoder remain unchanged. These optional geometry states are selected holdout pilots, not a replacement full-timeline training release. Track 2 remains on hold.</p><img src="revision.png" alt="Phase ablation, event trade-off and geometry-only inference cost"><h2>Exploratory validation</h2><table><tr><th>Holdout</th><th>Variant</th><th>Supported probes</th><th>IoU</th><th>BF2</th><th>Certificate fraction</th></tr>'''+cells+'''</table><p>Same retained observations and target masks. Strict retained-mask guards reject fractional reparameterizations that reduce static IoU or BF2. Target-informed oracles are separate diagnostic columns and never fit inputs or select parameters. All unsupported probes remain in the attempt ledger. Covariance is conditional on supplied input blocks; phase, topology, identity and repaired geometry uncertainty are not calibrated.</p><p><a href="phase.json">Phase selection and denominators</a> · <a href="repair.json">Localized repair</a> · <a href="events.json">Causal event evidence</a> · <a href="covariance.json">Covariance scope</a> · <a href="contexts.json">Stable-context recovery</a> · <a href="independent_references.json">Reference readiness</a> · <a href="verification.json">Numerical checks</a> · <a href="inference_latency.json">Latency scope</a></p><p>Scientific acceptance and a final freeze still require accepted independent labels, genuinely fresh participants and a complete clean rebuild from raw sources. Machine reviews cannot satisfy those gates. Contact-aware allocation and learned association rejection remain reference-dependent.</p>''')
 atomic_json(R/'scientific_readiness.json',dict(independent_references=refs['accepted_usable'],fresh_locked_evaluation=False,full_clean_raw_rebuild=False,bounded_clean_rebuild_is_not_full_dataset_reproduction=True,contact_aware_allocation='blocked_on_independent_contact_geometry',association_calibrated_rejection='blocked_on_independent_identity_labels',conditional_covariance_only=True,pilot_geometry_only=True,finalization_authorized=False,scientific_acceptance=False,track2_started=False))
 monitor.check()

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','inputs','phase','events','covariance','contexts','appearance','references','verify','report'],default='all');p.add_argument('--resume',action='store_true');a=p.parse_args()
 for path in [D,R,W]:path.mkdir(parents=True,exist_ok=True)
 lock=(W/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 source={str(p.relative_to(ROOT)):digest(p) for p in list((ROOT/'src/track1_encoder_revision').glob('*.py'))+[ROOT/'configs/track1_encoder_revision.json']};functions=dict(inputs=inputs,phase=phase,events=events,covariance=covariance,contexts=contexts,appearance=appearance,references=references,verify=verify,report=report);stages=list(functions) if a.stage=='all' else [a.stage]
 try:
  with Monitor() as monitor:
   for stage in stages:
    marker=W/f'{stage}.done.json'
    if a.resume and marker.exists():
     if json.loads(marker.read_text())['source']!=source:raise ValueError('Source changed after checkpoint; rerun explicit stage')
     LOG.info('RESUME %s',stage);continue
    atomic_json(R/'status.json',dict(state='running',stage=stage,pid=os.getpid(),updated_unix=time.time(),scientific_acceptance=False));LOG.info('BEGIN %s',stage);start=time.monotonic();functions[stage](monitor);atomic_json(marker,dict(source=source,seconds=time.monotonic()-start));LOG.info('COMPLETE %s %.1fs',stage,time.monotonic()-start)
  outputs={str(p.relative_to(D)):digest(p) for p in D.glob('*') if p.is_file() and p.name!='manifest.json'};atomic_json(D/'manifest.json',dict(source=source,outputs=outputs,scientific_acceptance=False));atomic_json(R/'status.json',dict(state='complete',stage=a.stage,scientific_acceptance=False,track2_started=False))
 except BaseException as e:atomic_json(R/'status.json',dict(state='failed',stage=stage,error=str(e),scientific_acceptance=False));raise
if __name__=='__main__':main()
