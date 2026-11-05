"""Train-only uncertainty calibration with held-out contour vertices/observations."""
from collections import defaultdict
import json,logging
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json,digest
from .artifacts import output_dir,load_shard,manifest,save_index
from .spatial import fit_spline
from .temporal import fit_function
from .numerics import basis

LOG=logging.getLogger('track1')

def coverage_metrics(error,variance):
    error=np.asarray(error);variance=np.maximum(np.asarray(variance),1e-20)
    z=error/np.sqrt(variance);bins=[]
    for lo,hi in zip(np.quantile(np.sqrt(variance),[0,.25,.5,.75]),np.quantile(np.sqrt(variance),[.25,.5,.75,1.])):
        mask=(np.sqrt(variance)>=lo)&(np.sqrt(variance)<=hi)
        bins.append(dict(sigma_min=float(lo),sigma_max=float(hi),n=int(mask.sum()),coverage95=float(np.mean(np.abs(z[mask])<=1.9599639845))))
    return dict(n=len(z),coverage68=float(np.mean(np.abs(z)<=1)),coverage95=float(np.mean(np.abs(z)<=1.9599639845)),
        gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*variance)+error*error/variance))),confidence_bins=bins)

def calibration_video(cfg,choice):
    v=choice['video_id'];spatial=output_dir(cfg,'spatial');state=load_shard(spatial/f'{v}.pt');rows=pq.read_table(spatial/f'{v}.parquet').to_pylist()
    results=[]
    for kind in cfg['provenance_priors']:
        candidates=[r for r in rows if r['spatial_valid'] and r['source_kind']==kind]
        if not candidates:continue
        chosen=np.linspace(0,len(candidates)-1,min(len(candidates),128),dtype=int)
        for j in chosen:
            row=candidates[j];y=state['boundary'][row['row_index']].astype(float)
            weights=np.ones(len(y));weights[::4]=0
            fit=fit_spline(y,cfg['spatial'],row['provenance_weight']*row['geometry_weight'],sample_weights=weights)
            for k in range(0,len(y),4):results.append(dict(video_id=v,mask_uid=row['mask_uid'],source_kind=kind,engineering_split=row['engineering_split'],visor_split=row['visor_split'],error=float(fit['residual_normal'][k]),variance=float(fit['normal_var'][k])))
    path=output_dir(cfg,'calibration')/f'{v}.parquet';save_index(path,results);manifest(cfg,'calibration',v,[path],dict(heldout_vertices=len(results)))
    return len(results)

def fit_spatial_calibration(cfg,choices):
    records=[]
    for c in choices:records+=pq.read_table(output_dir(cfg,'calibration')/f'{c["video_id"]}.parquet').to_pylist()
    train=[r for r in records if r['engineering_split']=='train' and r['visor_split']=='train' and r['source_kind']=='manual_sparse']
    ratio=np.array([r['error']**2/max(r['variance'],1e-20) for r in train])
    kappa=float(np.clip(np.quantile(ratio,.95)/1.9599639845**2,1e-4,1e6))
    evaluations={}
    for split in ['train','validation','test','source_visor_validation']:
        for kind in cfg['provenance_priors']:
            selected=[r for r in records if r['source_kind']==kind and (r['visor_split']=='val' if split=='source_visor_validation' else r['engineering_split']==split)]
            if selected:
                error=[r['error'] for r in selected];var=np.array([r['variance'] for r in selected])
                evaluations[split+'/'+kind]=dict(before=coverage_metrics(error,var),after=coverage_metrics(error,var*kappa))
    model=dict(kappa_cov=kappa,method='95th percentile squared held-out normal residual / Gaussian 95% quantile squared; manual masks only',
        training_vertices=len(train),training_videos=sorted({r['video_id'] for r in train}),heldout_vertex_stride=4,temporal_covariance_multiplier=1.,
        interpretation='Residual-scaled isotropic IRLS Gaussian approximation; calibration is contour representation error, not human-label uncertainty.',
        interpolation_evaluation='Held-out vertices from interpolated polygons diagnose fit coverage, not independent manual annotation accuracy.',evaluations=evaluations)
    atomic_json(output_dir(cfg,'calibration')/'model.json',model)
    LOG.info('Spatial covariance calibration learned kappa=%.6g from %d held-out manual vertices',kappa,len(train))
    return model

def temporal_calibration_cases(cfg,choice,limit=128):
    video=choice['video_id'];out=output_dir(cfg,'spatial');s=load_shard(out/f'{video}.pt');rows=pq.read_table(out/f'{video}.parquet').to_pylist()
    calibration=json.loads((output_dir(cfg,'calibration')/'model.json').read_text());by_track=defaultdict(list)
    for r in rows:
        if r['spatial_valid']:by_track[r['entity_track_id']].append(r)
    cases=[];count=0
    for track,rs in sorted(by_track.items()):
        if count>=limit:break
        if len(rs)<10:continue
        t=np.array([r['timestamp_us'] for r in rs])/1e6
        if t[-1]-t[0]<.65:continue
        end=np.floor(t[-1]*4)/4;sel=(t>=end-1)&(t<=end)
        local=np.flatnonzero(sel)
        if len(local)<10:continue
        ids=np.array([rs[i]['row_index'] for i in local]);ts=t[local];held=len(local)//2;keep=np.arange(len(local))!=held
        prov=np.array([rs[i]['provenance_weight'] for i in local]);quality=np.array([rs[i]['geometry_weight'] for i in local])
        variance=s['variance'][ids].astype(float)*calibration['kappa_cov']
        fit=fit_function(ts[keep],s['h'][ids[keep]].astype(float),variance[keep],prov[keep],cfg['temporal'],end-1,quality[keep])
        if not fit.valid:continue
        phi=basis(2*(ts[held]-(end-1))-1);prediction=phi@fit.coefficients
        error=s['h'][ids[held]]-prediction;var=np.einsum('i,dij,j->d',phi,fit.coefficient_cov,phi)
        baseline=s['h'][ids[np.flatnonzero(keep)[np.argmin(np.abs(ts[keep]-ts[held]))]]]
        for d in range(35):cases.append(dict(video_id=video,entity_track_id=track,engineering_split=rs[0]['engineering_split'],visor_split=rs[0]['visor_split'],feature=d,error=float(error[d]),variance=float(var[d]),observation_variance=float(variance[held,d]),baseline_error=float(s['h'][ids[held],d]-baseline[d])))
        count+=1
    path=output_dir(cfg,'calibration')/f'{video}.temporal.parquet';save_index(path,cases)
    return len(cases)

def fit_temporal_calibration(cfg,choices):
    rows=[]
    for c in choices:
        p=output_dir(cfg,'calibration')/f'{c["video_id"]}.temporal.parquet'
        if p.exists() and pq.read_metadata(p).num_rows:rows+=pq.read_table(p).to_pylist()
    train=[r for r in rows if r['engineering_split']=='train' and r['visor_split']=='train']
    if not train:raise ValueError('No valid train-only temporal calibration cases')
    kappa=float(np.clip(np.quantile([max(r['error']**2/1.9599639845**2-r['observation_variance'],0)/max(r['variance'],1e-20) for r in train],.95),1e-4,1e6))
    path=output_dir(cfg,'calibration')/'model.json';model=json.loads(path.read_text());model['temporal_covariance_multiplier']=kappa
    model['temporal_training_cases']=len(train);model['temporal_evaluations']={}
    for split in ['train','validation','test','source_visor_validation']:
        chosen=[r for r in rows if r['visor_split']=='val'] if split=='source_visor_validation' else [r for r in rows if r['engineering_split']==split]
        if chosen:
            error=np.array([r['error'] for r in chosen]);var=np.array([r['variance'] for r in chosen]);obs=np.array([r['observation_variance'] for r in chosen]);baseline=np.array([r['baseline_error'] for r in chosen])
            model['temporal_evaluations'][split]=dict(before=coverage_metrics(error,var+obs),after=coverage_metrics(error,var*kappa+obs),heldout_rmse=float(np.sqrt(np.mean(error**2))),nearest_observation_rmse=float(np.sqrt(np.mean(baseline**2))))
    model['temporal_evaluation_limit']='Held-out observation interpolation with neighbouring source-derived dense evidence; not independent dynamics prediction or human annotation accuracy.'
    atomic_json(path,model);LOG.info('Temporal covariance calibration learned kappa=%.6g from %d scalar cases',kappa,len(train))
    manifest(cfg,'calibration','model',[path],dict(spatial_kappa=model['kappa_cov'],temporal_kappa=kappa))
    return model
