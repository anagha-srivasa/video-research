"""Training-only calibration; validation is evaluated without refitting its targets."""
from collections import defaultdict
import json
import numpy as np
from pipeline.common import ROOT,atomic_json,digest
from track1.spatial import fit_spline
from track1.numerics import basis
from track1.calibration import coverage_metrics
from .protocol import config,folder,write_report,baseline_config,source_identity,LOG
from .data import spatial_cases,temporal_cases,holdout
from .kernels import spatial_fit,spatial_errors,temporal_fit
from .build import directory,manifest

def spatial_calibration():
    cfg=config();options=json.loads((folder('reports')/'spatial_selection.json').read_text())['options'];records=[]
    for case in spatial_cases(cfg['train']+cfg['validation']):
        y=case['y'] if options['samples']==128 else case['y'][::2];weights=np.ones(len(y));weights[::4]=0
        fit=spatial_fit(y,options,case['geometry_weight']*case['provenance_weight'],weights)
        if not fit['valid']:
            y0=case['y'][::2];w0=np.ones(64);w0[::4]=0;old=fit_spline(y0,baseline_config()['spatial'],case['geometry_weight']*case['provenance_weight'],sample_weights=w0)
            if not old['self_intersection']:fit=dict(controls=old['controls'],covariance=old['covariance'],valid=True)
        error,var,_=spatial_errors(case['y'][::8],fit)
        records.append(dict(video_id=case['video_id'],source_kind=case['source_kind'],split=case['split'],valid=fit['valid'],error=error,variance=var))
    train=[r for r in records if r['split']=='train' and r['source_kind']=='manual_sparse' and r['valid']]
    z2=np.concatenate([r['error']**2/r['variance'] for r in train]);kappa=float(np.quantile(z2,.95)/1.9599639845**2)
    evaluations={}
    for split in ['train','validation']:
        for kind in cfg['spatial_cases_per_kind']:
            rs=[r for r in records if r['split']==split and r['source_kind']==kind and r['valid']]
            if rs:evaluations[split+'/'+kind]=coverage_metrics(np.concatenate([r['error'] for r in rs]),np.concatenate([r['variance'] for r in rs])*kappa)
    model=dict(kappa_cov=kappa,feature_scale=[1.]*35,temporal_multipliers=[1.]*35,training_videos=cfg['train'],spatial_calibration_evaluations=evaluations,
        invalid_calibration_masks=sum(not r['valid'] for r in records),interpretation='Held-out contour representation error, conditional on usable geometry and fitted correspondence. Not human-label uncertainty.',test_used=False)
    atomic_json(directory('calibration')/'model.json',model);write_report('spatial_calibration.json',model)
    return model

def temporal_calibration():
    cfg=config();selected=json.loads((folder('reports')/'selection_locked.json').read_text());options=selected['temporal']['options'];scale=np.asarray(selected['feature_scale'])
    path=directory('calibration')/'model.json';model=json.loads(path.read_text());records=[]
    for case in temporal_cases(cfg['train']+cfg['validation']):
        centres=cfg['holdout_centres'];from .protocol import rank
        centre=centres[int(rank(f"{case['entity_track_id']}:{case['window_end_us']}")[:8],16)%len(centres)]
        for protocol in cfg['protocols']:
            hidden=holdout(case['time'],protocol,centre);keep=~hidden
            if not hidden.any() or keep.sum()<8:continue
            t=case['time'];h=case['h'];fit=temporal_fit(t[keep],h[keep],case['variance'][keep]*model['kappa_cov'],case['prior'][keep],case['quality'][keep],options,scale)
            phi=basis(2*t[hidden]-1);error=phi@fit['coefficients']-h[hidden];variance=np.maximum(np.einsum('ni,dij,nj->nd',phi,fit['covariance'],phi),1e-15)
            records.append(dict(split=case['split'],protocol=protocol,error=error,variance=variance))
    ratios={}
    for protocol in ['all']+cfg['protocols']:
        rs=[r for r in records if r['split']=='train' and (protocol=='all' or r['protocol']==protocol)]
        error=np.concatenate([r['error'] for r in rs]);variance=np.concatenate([r['variance'] for r in rs])
        ratios[protocol]=np.maximum(np.quantile(error**2/variance,.95,axis=0)/1.9599639845**2,1e-4)
    # Protect the endpoint uncertainty required by the functional-state contract.
    multipliers=np.maximum(ratios['all'],ratios['endpoint100']);evaluations={}
    for split in ['train','validation']:
        for protocol in cfg['protocols']:
            rs=[r for r in records if r['split']==split and r['protocol']==protocol];error=np.concatenate([r['error'] for r in rs]);var=np.concatenate([r['variance'] for r in rs])*multipliers
            evaluations[split+'/'+protocol]=coverage_metrics(error.ravel(),var.ravel())
    model.update(temporal_multipliers=multipliers.tolist(),feature_scale=scale.tolist(),temporal_calibration_evaluations=evaluations,
        temporal_calibration_policy='Per-feature empirical covariance scale fitted on train only; maximum of pooled and endpoint 95% requirements. Validation/test never fitted.')
    atomic_json(path,model);manifest('calibration','model',[path],[folder('reports')/'spatial_selection.json',folder('reports')/'selection_locked.json'],dict(kappa_cov=model['kappa_cov']))
    write_report('calibration.json',model);return model
