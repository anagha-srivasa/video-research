"""One locked exploratory evaluation per seed, paired against constant velocity."""
import collections,json,time
import numpy as np
import torch
from .data import *
from .train import load_model,tensors,seed_all
from pipeline.common import atomic_json


def cv(data,ids):
 x=data.x[ids,-1];out=np.repeat(np.c_[x[:,140:144],x[:,:140]][:,None],3,axis=1).astype(np.float64)
 for hi,h in enumerate(CFG['horizons_us']):out[:,hi,:2]+=h/1e6*x[:,142:144]
 return out

def prediction_rows(data,ids,predictions,family,seed):
 rows=[]
 for j,i in enumerate(ids):
  sample=data.samples[i]
  for h,us in enumerate(CFG['horizons_us']):
   valid=bool(data.mask[i,h]);target=data.y[i,h].astype(float);pred=predictions[j,h].astype(float)
   if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite forecast')
   rows.append(dict(sequence_id=sample['sequence_id'],video_id=sample['video_id'],participant=sample['participant'],episode_id=sample['episode_id'],split=sample['split'],input_cutoff_us=sample['input_times_us'][-1],horizon_us=us,family=family,seed=seed,target_supported=valid,predicted_position=pred[:2].tolist(),predicted_velocity=pred[2:4].tolist(),target_position=target[:2].tolist() if valid else None,target_velocity=target[2:4].tolist() if valid else None,position_squared_error=float(np.sum((pred[:2]-target[:2])**2)) if valid else None,velocity_squared_error=float(np.sum((pred[2:4]-target[2:4])**2)) if valid else None,coarse_squared_error=float(np.mean((pred[4:]-target[4:])**2)) if valid else None,coordinate_mode='intrinsic_normalized_2D',physical_reference=False))
 return rows

def evaluate():
 data=Data();ids=data.indices('validation',usable=False);rows=prediction_rows(data,ids,cv(data,ids),'cv',0);spectra=[];latencies=[]
 files=[DATA/'models'/f'{family}.{seed}.pt' for family in CFG['models'] for seed in CFG['seeds']];lock=dict(models={p.name:digest(p) for p in files},config_sha256=digest(ROOT/'configs/track2_feasibility.json'),validation_participants=sorted(set(data.participants[ids].tolist())),validation_is_exploratory=True)
 if (REPORT/'evaluation_lock.json').exists():raise FileExistsError('Exploratory evaluation already locked; do not evaluate again')
 atomic_json(REPORT/'evaluation_lock.json',lock)
 for family in CFG['models']:
  for seed in CFG['seeds']:
   seed_all(seed);model,norm,checkpoint=load_model(DATA/'models'/f'{family}.{seed}.pt');x,_,_=tensors(data,norm)
   # Exactly one complete validation forward pass per final model/seed.
   with torch.no_grad():standardized,_=model(x[ids]);pred=standardized.numpy().astype(float)*np.array(norm['y_scale'])+np.array(norm['y_mean'])
   rows+=prediction_rows(data,ids,pred,family,seed)
   if family in ['koopman','stable_koopman']:
    generator=model.generator().detach().numpy().astype(float);eig=np.linalg.eigvals(generator);spectra.append(dict(family=family,seed=seed,spectral_abscissa=float(eig.real.max()),generator_norm=float(np.linalg.norm(generator,2))))
   # Timing uses a training history, includes normalization + model + denormalization.
   i=data.indices()[0];times=[];warm=data.x[i:i+1].copy();raw_scale=np.array(norm['x_scale'],np.float32);raw_mean=np.array(norm['x_mean'],np.float32);ym=np.array(norm['y_mean'],np.float32);ys=np.array(norm['y_scale'],np.float32)
   for _ in range(100):
    start=time.perf_counter();z=warm.copy();z[:,:,:154]=(z[:,:,:154]-raw_mean)/raw_scale
    with torch.no_grad():out=model(torch.from_numpy(z))[0].numpy()*ys+ym
    times.append((time.perf_counter()-start)*1000)
   latencies.append(dict(family=family,seed=seed,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),batch_size=1,three_horizons=True,input_is_cached_M16_state=True,includes_rgb_or_track1_encoding=False))
 save(REPORT/'predictions.parquet',rows);atomic_json(REPORT/'spectra.json',spectra);atomic_json(REPORT/'inference_latency.json',dict(records=latencies,device='cpu',transition_matrices_precompiled=True,precision='float32',upstream_rgb_masks_track1_excluded=True));summarize(rows,spectra)


def summarize(rows,spectra):
 per=[];summary=[]
 for family in ['cv']+CFG['models']:
  for seed in ([0] if family=='cv' else CFG['seeds']):
   for h in CFG['horizons_us']:
    for p in sorted({r['participant'] for r in rows}):
     values=[r for r in rows if r['family']==family and r['seed']==seed and r['horizon_us']==h and r['participant']==p and r['target_supported']]
     if not values:raise ValueError('Empty scoring participant/horizon')
     per.append(dict(family=family,seed=seed,horizon_us=h,participant=p,targets=len(values),position_rmse=float(np.sqrt(np.mean([r['position_squared_error'] for r in values],dtype=np.float64))),velocity_rmse=float(np.sqrt(np.mean([r['velocity_squared_error'] for r in values],dtype=np.float64))),maximum_position_error=float(np.sqrt(max(r['position_squared_error'] for r in values)))))
    local=[r for r in per if r['family']==family and r['seed']==seed and r['horizon_us']==h];summary.append(dict(family=family,seed=seed,horizon_us=h,mean_participant_position_rmse=float(np.mean([r['position_rmse'] for r in local])),mean_participant_velocity_rmse=float(np.mean([r['velocity_rmse'] for r in local])),targets=sum(r['targets'] for r in local)))
 save(REPORT/'participant_metrics.parquet',per);atomic_json(REPORT/'metrics.json',dict(summary=summary,participants=sorted({r['participant'] for r in per}),aggregation='participant RMSE first; then mean participant; final seed summary is median',independent_episode_count_not_sequence_count=True,validation_exploratory=True))
 rng=np.random.default_rng(20261105);gates=[];all_seed_position_pass=True;ratios={};ci={}
 for h in CFG['horizons_us']:
  base=next(r for r in summary if r['family']=='cv' and r['horizon_us']==h);values=[r for r in summary if r['family']=='stable_koopman' and r['horizon_us']==h];ratios[h]=float(np.median([r['mean_participant_position_rmse'] for r in values]))/max(base['mean_participant_position_rmse'],1e-12);velocity_ratio=float(np.median([r['mean_participant_velocity_rmse'] for r in values]))/max(base['mean_participant_velocity_rmse'],1e-12)
  participants=sorted({r['participant'] for r in per});pairs=[]
  for p in participants:
   cv_r=next(r['position_rmse'] for r in per if r['participant']==p and r['family']=='cv' and r['horizon_us']==h);stable=float(np.median([r['position_rmse'] for r in per if r['participant']==p and r['family']=='stable_koopman' and r['horizon_us']==h]));pairs.append((stable,cv_r))
  samples=[]
  for _ in range(CFG['bootstrap_repetitions']):
   selected=np.array(pairs)[rng.integers(0,len(pairs),len(pairs))];samples.append(float(selected[:,0].mean()/max(selected[:,1].mean(),1e-12)))
  ci[h]=np.quantile(samples,[.025,.975]).tolist();worst=max(r['mean_participant_position_rmse']/max(base['mean_participant_position_rmse'],1e-12) for r in values);all_seed_position_pass&=worst<=CFG['gates']['worst_seed_position_max_ratio'];gates.append(dict(horizon_us=h,median_position_ratio=ratios[h],participant_bootstrap_ratio95=ci[h],bootstrap_independent_units=len(pairs),ci_is_exploratory_not_independent_acceptance=True,median_velocity_ratio=velocity_ratio,worst_seed_position_ratio=worst))
 g250=ratios[250000]<=CFG['gates']['position_250_ratio'] or ci[250000][1]<1.;g500=ratios[500000]<=CFG['gates']['position_500_max_ratio'];g1000=ratios[1000000]<=CFG['gates']['position_1000_bounded_ratio'];velocity=all(g['median_velocity_ratio']<=CFG['gates']['velocity_max_ratio'] for g in gates);spectral=all(r['spectral_abscissa']<=CFG['gates']['stable_spectral_abscissa_tolerance'] for r in spectra if r['family']=='stable_koopman');passed=bool(g250 and g500 and g1000 and velocity and all_seed_position_pass and spectral)
 atomic_json(REPORT/'decision.json',dict(gates=gates,position_250_pass=bool(g250),position_500_pass=bool(g500),position_1000_bounded_pass=bool(g1000),velocity_pass=bool(velocity),seed_stability_pass=bool(all_seed_position_pass),stable_generator_spectral_pass=bool(spectral),feasibility_passed=passed,stop_architecture_work=not passed,stop_reason='Predeclared feasibility gates not all met' if not passed else None,full_track2_go=False,scientific_acceptance=False,optional_ablations_started=False,independent_physical_references=False))
