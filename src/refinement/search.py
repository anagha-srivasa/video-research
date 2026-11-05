"""Train/validation-only searches with fixed representation budgets and plateau stopping."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing,json,time,logging,itertools
import numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline
from track1.spatial import fit_spline
from track1.temporal import fit_function
from track1.numerics import basis
from track1.artifacts import save_shard
from .protocol import config,baseline_config,folder,source_identity,write_report,LOG,rank
from .data import spatial_cases,temporal_cases,holdout
from .kernels import spatial_fit,spatial_errors,temporal_fit

_SPATIAL=None;_TEMPORAL=None;_SCALE=None;_KAPPA=None

def initialize(kind,scale=None):
    global _SPATIAL,_TEMPORAL,_SCALE,_KAPPA
    cfg=config();_SCALE=None if scale is None else np.asarray(scale)
    from pipeline.common import ROOT
    marker=folder('reports')/'representation.json'
    model_path=ROOT/'derived/track1_refined_v2/calibration/model.json' if marker.exists() else ROOT/'derived/track1_calibration_v1/model.json'
    _KAPPA=json.loads(model_path.read_text())['kappa_cov']
    if kind=='spatial':_SPATIAL=spatial_cases(cfg['train']+cfg['validation'])
    else:_TEMPORAL=temporal_cases(cfg['train']+cfg['validation'])

def spatial_trial(options):
    totals=defaultdict(list);failed=defaultdict(list);cfg=baseline_config()
    for case in _SPATIAL:
        if case['source_kind']!='manual_sparse':continue
        y=case['y'][::2] if options['samples']==64 else case['y'];mask=np.ones(len(y));mask[::4]=0
        if options.get('baseline'):
            fit=fit_spline(y,cfg['spatial'],sample_weights=mask)
            fit=dict(controls=fit['controls'],covariance=fit['covariance'],valid=not fit['self_intersection'])
        else:fit=spatial_fit(y,options,sample_weights=mask)
        error,_,distance=spatial_errors(case['y'][::8],fit)
        key=(case['split'],case['video_id']);scale=max(np.sqrt(np.mean(np.sum((y-y.mean(0))**2,axis=1))),1)
        totals[key].append(dict(rmse=float(np.sqrt(np.mean(error**2))),distance=float(np.mean(distance)),normalised=float(np.sqrt(np.mean(error**2))/scale)))
        failed[key].append(not fit['valid'])
    result={}
    for split in ['train','validation']:
        by_video={v:dict(rmse=float(np.mean([r['rmse'] for r in rs])),distance=float(np.mean([r['distance'] for r in rs])),normalised=float(np.mean([r['normalised'] for r in rs])),invalid_fraction=float(np.mean(failed[(s,v)])),n=len(rs)) for (s,v),rs in totals.items() if s==split}
        result[split]=dict(by_video=by_video,rmse=float(np.mean([r['rmse'] for r in by_video.values()])),normalised=float(np.mean([r['normalised'] for r in by_video.values()])),invalid_fraction=float(np.mean([r['invalid_fraction'] for r in by_video.values()])))
    result['options']=options;result['objective']=result['validation']['normalised']*(1+10*result['validation']['invalid_fraction'])
    return result

def temporal_trial(options):
    totals=defaultdict(list);baseline=baseline_config();cfg=config()
    for case in _TEMPORAL:
        for protocol in cfg['protocols']:
            centres=cfg['holdout_centres'];centre=centres[int(rank(f"{case['entity_track_id']}:{case['window_end_us']}")[:8],16)%len(centres)]
            hidden=holdout(case['time'],protocol,centre);keep=~hidden
            if not hidden.any() or keep.sum()<8:continue
            t=case['time'];h=case['h'];var=case['variance']*_KAPPA
            if options.get('baseline'):
                fit=fit_function(t[keep],h[keep],var[keep],case['prior'][keep],baseline['temporal'],0,case['quality'][keep])
                coefficients=fit.coefficients;neff=fit.effective_samples
            else:
                fit=temporal_fit(t[keep],h[keep],var[keep],case['prior'][keep],case['quality'][keep],options,_SCALE)
                coefficients=fit['coefficients'];neff=fit['neff']
            prediction=basis(2*t[hidden]-1)@coefficients;error=prediction-h[hidden]
            totals[(case['split'],case['video_id'],protocol)].append(dict(mse=float(np.mean(error**2)),normalised_mse=float(np.mean((error/_SCALE)**2)),neff=float(neff)))
    result={}
    for split in ['train','validation']:
        protocols={}
        for protocol in cfg['protocols']:
            by_video={v:dict(rmse=float(np.sqrt(np.mean([r['mse'] for r in rs]))),normalised_rmse=float(np.sqrt(np.mean([r['normalised_mse'] for r in rs]))),valid_fraction=float(np.mean([r['neff']>=6 for r in rs])),n=len(rs)) for (s,v,p),rs in totals.items() if s==split and p==protocol}
            protocols[protocol]=dict(by_video=by_video,rmse=float(np.mean([r['rmse'] for r in by_video.values()])),normalised_rmse=float(np.mean([r['normalised_rmse'] for r in by_video.values()])),valid_fraction=float(np.mean([r['valid_fraction'] for r in by_video.values()])))
        result[split]=protocols
    weight=cfg['selection']['temporal_endpoint_weight'];score=(result['validation']['single']['normalised_rmse']+result['validation']['gap100']['normalised_rmse']+weight*result['validation']['endpoint100']['normalised_rmse'])/(2+weight)
    result.update(options=options,objective=score,minimum_valid_fraction=min(r['valid_fraction'] for r in result['validation'].values()))
    return result

def batch_trials(kind,options,monitor,scale=None,round_=0):
    cfg=config();results=[];start=time.monotonic()
    workers=min(cfg['workers'],10,len(options));function=spatial_trial if kind=='spatial' else temporal_trial
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(kind,scale)) as pool:
        jobs={pool.submit(function,opt):i for i,opt in enumerate(options)}
        while jobs:
            done,_=wait(jobs,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
            for future in done:
                number=jobs.pop(future);result=future.result();result.update(kind=kind,round=round_,trial=number,protocol_revision=cfg['protocol_revision'],elapsed_seconds=time.monotonic()-start);results.append(result)
                with (folder('reports')/'search_history.jsonl').open('a') as file:file.write(json.dumps(result)+'\n')
                LOG.info('%s round=%d trial=%d score=%.6g elapsed=%.1fs %s',kind,round_,number,result['objective'],time.monotonic()-start,result['options'])
    return results

def search_all(cfg,monitor,temporal_only=False):
    # Materialising only train data learns the feature scale; test arrays are not read.
    training=temporal_cases(cfg['train']);values=np.concatenate([r['h'][::3] for r in training]);centre=np.median(values,axis=0);scale=np.maximum(np.median(abs(values-centre),axis=0)*1.4826,.01)
    save_shard(folder('reports')/'feature_scale.pt',dict(scale=scale,centre=centre));del values,training
    spatial_options=[dict(baseline=True,samples=64,curvature=.001,loss='huber',reparameterize=0)]
    for samples,iterations,curvature,loss in itertools.product([64,128],[0,2,6],[.0001,.001],['l2','huber']):spatial_options.append(dict(samples=samples,reparameterize=iterations,curvature=curvature,loss=loss,irls=3,delta=1.5,parameter_step=.5))
    spatial_results=[json.loads((folder('reports')/'spatial_selection.json').read_text())] if temporal_only else batch_trials('spatial',spatial_options,monitor)
    allowed=[r for r in spatial_results if r['validation']['invalid_fraction']<=cfg['selection']['spatial_max_self_intersection_fraction']]
    if not allowed:allowed=spatial_results
    best_spatial=min(allowed,key=lambda r:r['objective'])
    candidates=[]
    for power,loss,regularization,floor in itertools.product([0,.5,1],['l2','huber','cauchy'],[0,.001,.1],[0,.01]):
        candidates.append(dict(variance_power=power,loss=loss,regularization=regularization,floor=floor,variance_clip=100,provenance_power=1,irls=3,delta=3.,robust_floor=.01))
    candidates=sorted(candidates,key=lambda r:rank(json.dumps(r,sort_keys=True)))[:48]
    candidates += [dict(variance_power=0,loss='l2',regularization=0,floor=0,variance_clip=0,provenance_power=0,irls=1),dict(baseline=True)]
    for penalty,power,regularization in itertools.product([[0,0,0,9],[0,0,1,9],[0,.01,1,36],[0,.1,1,100]],[0,.5],[.1,1,10]):
        candidates.append(dict(variance_power=power,loss='l2',regularization=regularization,floor=.01,variance_clip=100,provenance_power=0,mode_penalty=penalty,irls=1))
    temporal_results=batch_trials('temporal',candidates,monitor,scale.tolist())
    best_temporal=min(temporal_results,key=lambda r:r['objective']);history=[dict(round=0,best_objective=best_temporal['objective'])];plateau=0
    seen={json.dumps(r['options'],sort_keys=True) for r in temporal_results}
    for round_ in range(1,7):
        parent=best_temporal['options']
        if parent.get('baseline'):break
        proposals=[]
        for key,values in dict(variance_power=[max(0,parent['variance_power']-.25),min(1.5,parent['variance_power']+.25)],regularization=[parent['regularization']/3,parent['regularization']*3+.00001],robust_floor=[.001,.1],delta=[1.5,6.],provenance_power=[0,.5],variance_clip=[10,0]).items():
            for value in values:
                candidate=dict(parent,**{key:value});fingerprint=json.dumps(candidate,sort_keys=True)
                if fingerprint not in seen:proposals.append(candidate);seen.add(fingerprint)
        extra=batch_trials('temporal',proposals,monitor,scale.tolist(),round_);temporal_results+=extra
        new=min(temporal_results,key=lambda r:r['objective']);improvement=(best_temporal['objective']-new['objective'])/best_temporal['objective']
        plateau=plateau+1 if improvement<.002 else 0;best_temporal=new;history.append(dict(round=round_,best_objective=new['objective'],relative_improvement=improvement))
        if plateau>=cfg['selection']['plateau_rounds']:break
    selected=dict(**source_identity(),spatial=best_spatial,temporal=best_temporal,feature_scale=scale.tolist(),training_videos=cfg['train'],validation_videos=cfg['validation'],test_videos=cfg['test'],test_evaluated=False,
        temporal_rounds=history,stopped='validation plateau or completed bounded local search',fixed_budget=dict(spatial_controls=16,temporal_coefficients=4,state_channels=35),
        note='All candidate selection uses train/validation only. Published VISOR segmentation scores remain inapplicable to annotated-contour reconstruction.')
    write_report('selection_locked.json',selected);write_report('search_results.json',dict(spatial=spatial_results,temporal=temporal_results))
    if temporal_only:write_report('temporal_selection.json',selected['temporal'])
    LOG.info('Selection locked: spatial %.6g temporal %.6g',best_spatial['objective'],best_temporal['objective'])
