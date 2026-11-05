"""Sealed matched-input evaluation and independently checked ingestion gates."""
from collections import defaultdict
import json,time,tarfile
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator,CubicSpline
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import load_shard,save_index
from track1.numerics import basis
from track1.temporal import fit_function,collocation
from track1.calibration import coverage_metrics
from .protocol import config,baseline_config,folder,rank,write_report,source_identity,LOG
from .data import temporal_cases,spatial_cases,holdout
from .kernels import temporal_fit,spatial_fit,spatial_errors,periodic_design
from .build import directory
from .workflow import audit_baseline


def temporal_evaluation(monitor):
    cfg=config();selected=json.loads((folder('reports')/'selection_locked.json').read_text());model=json.loads((directory('calibration')/'model.json').read_text());scale=np.asarray(model['feature_scale']);records=[];uncertainty=defaultdict(list);capacity=[]
    for video in cfg['train']+cfg['validation']+cfg['test']:
        for case in temporal_cases([video]):
            t=case['time'];h=case['h'];centre=cfg['holdout_centres'][int(rank(f"{case['entity_track_id']}:{case['window_end_us']}")[:8],16)%len(cfg['holdout_centres'])]
            oracle=np.linalg.lstsq(basis(2*t-1),h,rcond=None)[0]
            capacity.append(dict(video_id=video,split=case['split'],mse=float(np.mean((basis(2*t-1)@oracle-h)**2)),observations=len(t)))
            for protocol in cfg['protocols']:
                hidden=holdout(t,protocol,centre);keep=~hidden
                if not hidden.any() or keep.sum()<8:continue
                x=t[keep];y=h[keep];target=h[hidden];q=t[hidden];phi=basis(2*q-1);var=case['variance'][keep]*model['kappa_cov']
                fit=temporal_fit(x,y,var,case['prior'][keep],case['quality'][keep],selected['temporal']['options'],scale)
                old=fit_function(x,y,var,case['prior'][keep],baseline_config()['temporal'],0,case['quality'][keep])
                near=np.abs(x[:,None]-q).argmin(axis=0)
                linear=np.stack([np.interp(q,x,y[:,j]) for j in range(35)],axis=1)
                # Explicit linear extrapolation at either end, using only retained observations.
                for edge,a,b in [(q<x[0],0,1),(q>x[-1],-2,-1)]:
                    linear[edge]=y[a]+(q[edge,None]-x[a])*(y[b]-y[a])/(x[b]-x[a])
                predictions=dict(refined=phi@fit['coefficients'],baseline=phi@old.coefficients,nearest=y[near],linear=linear,pchip=PchipInterpolator(x,y,axis=0,extrapolate=True)(q),cubic_spline=CubicSpline(x,y,axis=0,extrapolate=True)(q))
                for method,pred in predictions.items():
                    error=pred-target
                    records.append(dict(video_id=video,split=case['split'],entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,method=method,mse=float(np.mean(error**2)),normalised_mse=float(np.mean((error/scale)**2)),target_rows=len(q),input_rows=len(x)))
                error=predictions['refined']-target;variance=np.maximum(np.einsum('ni,dij,nj->nd',phi,fit['covariance'],phi)*np.asarray(model['temporal_multipliers']),1e-15)
                uncertainty[(case['split'],protocol)].append((error.ravel(),variance.ravel()))
        LOG.info('Temporal evaluation %s complete',video);monitor.check()
    save_index(folder('reports')/'temporal_cases.parquet',records);save_index(folder('reports')/'cubic_capacity.parquet',capacity)
    grouped=defaultdict(list)
    for r in records:grouped[(r['split'],r['protocol'],r['method'],r['video_id'])].append(r)
    by_video={key:dict(rmse=float(np.sqrt(np.mean([r['mse'] for r in rs]))),normalised_rmse=float(np.sqrt(np.mean([r['normalised_mse'] for r in rs]))),windows=len(rs),targets=sum(r['target_rows'] for r in rs)) for key,rs in grouped.items()}
    results={}
    for split in ['train','validation','test']:
        results[split]={}
        for protocol in cfg['protocols']:
            methods={}
            for method in predictions:
                vs={v:r for (s,p,m,v),r in by_video.items() if (s,p,m)==(split,protocol,method)}
                methods[method]=dict(rmse=float(np.mean([r['rmse'] for r in vs.values()])),normalised_rmse=float(np.mean([r['normalised_rmse'] for r in vs.values()])),by_video=vs)
            u=uncertainty[(split,protocol)];videos=sorted(methods['baseline']['by_video']);a=np.array([methods['baseline']['by_video'][v]['normalised_rmse'] for v in videos]);b=np.array([methods['refined']['by_video'][v]['normalised_rmse'] for v in videos]);rng=np.random.default_rng(cfg['seed']);ix=rng.integers(0,len(a),size=(5000,len(a)));gains=1-b[ix].mean(1)/a[ix].mean(1)
            results[split][protocol]=dict(methods=methods,relative_normalised_rmse_gain=float(1-b.mean()/a.mean()),paired_video_bootstrap_gain_95_interval=np.quantile(gains,[.025,.975]).tolist(),bootstrap_limit='Only whole videos are resampled; intervals are descriptive with two validation or four test videos.',refined_uncertainty=coverage_metrics(np.concatenate([r[0] for r in u]),np.concatenate([r[1] for r in u])))
    return dict(splits=results,capacity_note='All-observation least-squares cubic is an in-sample compression lower bound only; it sees targets and is not a held-out prediction baseline.',baseline_note='Legacy temporal fitter evaluated on exactly the same refined h, variance and retained observations. Isolates temporal fitting; not an end-to-end old-state comparison.',interpolator_note='Nearest, linear, PCHIP and CubicSpline retain all supplied knots. They have more parameters than the fixed four coefficients per channel. Endpoint scores include extrapolation; these interpolators are not forecasting models.')


def spatial_evaluation(monitor):
    import cv2
    from PIL import Image,ImageDraw
    from pipeline import Dataset
    from track1.evaluation import geometry_metrics
    from track1.spatial import fit_spline
    cfg=config();options=json.loads((folder('reports')/'spatial_selection.json').read_text())['options'];records=[];withheld=[];phi=periodic_design(np.arange(256)/256)
    for video in cfg['train']+cfg['validation']+cfg['test']:
        cases=[c for c in spatial_cases([video]) if c['source_kind']=='manual_sparse'];new=load_shard(directory('spatial')/f'{video}.pt');rows=pq.read_table(directory('spatial')/f'{video}.parquet').to_pylist();lookup={r['mask_uid']:r for r in rows}
        sheet=Image.new('RGB',(1440,660),(24,24,24));draw=ImageDraw.Draw(sheet)
        with Dataset() as ds:
            for number,case in enumerate(cases):
                y=case['y'];weights=np.ones(128);weights[::4]=0;fit=spatial_fit(y,options,sample_weights=weights)
                w=np.ones(64);w[::4]=0;old=fit_spline(y[::2],baseline_config()['spatial'],sample_weights=w);old=dict(controls=old['controls'],covariance=old['covariance'],valid=not old['self_intersection'])
                for name,f in [('baseline',old),('refined',fit)]:
                    e,_,_=spatial_errors(y[::8],f);withheld.append(dict(video_id=video,split=case['split'],mask_uid=case['mask_uid'],method=name,rmse=float(np.sqrt(np.mean(e**2))),valid=f['valid']))
                if number>=48:continue
                r=lookup[case['mask_uid']];target=ds.get_mask(r['mask_uid']);height,width=target.shape
                n,labels,stats,_=cv2.connectedComponentsWithStats(target.astype(np.uint8),connectivity=8);dominant=(labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))).astype(np.uint8) if n>1 else target
                panel=np.zeros((height,width,3),np.uint8);panel[target>0]=(65,65,65)
                for name,controls,valid,colour in [('baseline',case['baseline'],r['baseline_spatial_valid'],(255,90,70)),('refined',new['controls'][r['row_index']],r['spatial_valid'],(50,255,230))]:
                    curve=phi@controls*[width/r['width'],height/r['height']];polygon=np.rint(curve).astype(np.int32);fitted=np.zeros_like(target);cv2.fillPoly(fitted,[polygon],1)
                    metrics=geometry_metrics(target,fitted);dom=geometry_metrics(dominant,fitted)
                    records.append(dict(video_id=video,split=case['split'],mask_uid=case['mask_uid'],method=name,valid=bool(valid),**metrics,dominant_iou=dom['iou'],components=r['component_count'],holes=r['hole_count']))
                    cv2.polylines(panel,[polygon],True,colour,2)
                if number<6:
                    x=number%3*480;top=number//3*330;sheet.paste(Image.fromarray(panel).resize((480,270)),(x,top));draw.text((x+8,top+275),f'{video} | {r["entity_name"][:30]}',fill='white');draw.text((x+8,top+294),f'{r["timestamp_us"]/1e6:.3f}s  components={r["component_count"]} holes={r["hole_count"]}',fill='white');draw.text((x+8,top+313),'Gray GT mask / red baseline / cyan refined',fill='white')
        sheet.save(folder('reports')/f'geometry.{video}.jpg',quality=92);LOG.info('Spatial evaluation %s complete',video);monitor.check()
    save_index(folder('reports')/'spatial_geometry.parquet',records);save_index(folder('reports')/'spatial_withheld.parquet',withheld)
    from .visuals import render_visuals
    render_visuals()
    result={}
    for split in ['train','validation','test']:
        result[split]={}
        for method in ['baseline','refined']:
            vs={}
            for video in cfg[split]:
                rs=[r for r in records if r['video_id']==video and r['method']==method];ws=[r for r in withheld if r['video_id']==video and r['method']==method]
                vs[video]=dict(**{key:float(np.mean([r[key] for r in rs])) for key in ['iou','boundary_fscore_2px','symmetric_boundary_distance_px','dominant_iou']},withheld_normal_rmse_px=float(np.mean([r['rmse'] for r in ws])),withheld_invalid_fraction=float(np.mean([not r['valid'] for r in ws])),geometry_masks=len(rs),withheld_masks=len(ws))
            result[split][method]=dict(**{k:float(np.mean([v[k] for v in vs.values()])) for k in ['iou','boundary_fscore_2px','symmetric_boundary_distance_px','dominant_iou','withheld_normal_rmse_px','withheld_invalid_fraction']},by_video=vs)
    return dict(splits=result,note='Video-macro means; deterministic manual cases, including invalid fits, never filtered for a better score. Both curves use 256 rasterisation samples. Full-mask IoU penalises holes and disconnected components absent from dominant-contour representation. Withheld contour refits isolate compression generalisation, without production fallback.')


def evaluate_all(monitor):
    selection=folder('reports')/'selection_locked.json';calibration=directory('calibration')/'model.json';lock=folder('reports')/'evaluation_lock.json'
    identity=dict(selection_sha256=digest(selection),calibration_sha256=digest(calibration))
    if lock.exists():assert json.loads(lock.read_text())==identity,'Parameters changed after evaluation lock'
    else:atomic_json(lock,identity)
    temporal=temporal_evaluation(monitor);spatial=spatial_evaluation(monitor)
    for split in ['train','validation','test']:
        write_report(split+'_evaluation.json',dict(**identity,temporal=temporal['splits'][split],spatial=spatial['splits'][split],public_benchmark_parity=False,limits=[temporal['capacity_note'],temporal['baseline_note'],temporal['interpolator_note'],spatial['note']]))
    write_report('evaluation.json',dict(temporal=temporal,spatial=spatial,**identity))


def freeze():
    cfg=config();baseline_count=audit_baseline();report=folder('reports');
    visual=json.loads((report/'visual_review.json').read_text());assert visual['reviewed'] and all(digest(ROOT/p)==h for p,h in visual['images'].items())
    tests=json.loads((report/'tests.json').read_text());assert tests['passed']
    ev=json.loads((report/'test_evaluation.json').read_text());lock=json.loads((report/'evaluation_lock.json').read_text());assert lock['selection_sha256']==digest(report/'selection_locked.json') and lock['calibration_sha256']==digest(directory('calibration')/'model.json')
    counts=defaultdict(int);maximum_error=0.;appnorm=[]
    for v in cfg['train']+cfg['validation']+cfg['test']:
        for stage in ['spatial','appearance','functional']:
            marker=json.loads((directory(stage)/f'{v}.manifest.json').read_text());assert marker['completed']
            for p,h in {**marker['inputs'],**marker['outputs']}.items():assert digest(ROOT/p)==h,(p,'hash mismatch')
        s=load_shard(directory('spatial')/f'{v}.pt');rows=pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist();f=load_shard(directory('functional')/f'{v}.pt');meta=pq.read_table(directory('functional')/f'{v}.parquet').to_pylist();a=load_shard(directory('appearance')/f'{v}.pt')
        assert len(rows)==pq.read_metadata(ROOT/f'derived/track1_spatial_v1/{v}.parquet').num_rows
        assert all(not r['baseline_spatial_valid'] or r['spatial_valid'] for r in rows)
        for key in ['controls','covariance','h','variance']:assert np.isfinite(s[key]).all(),(v,key)
        assert f['coefficients'].shape==(len(meta),4,35) and f['coefficient_cov'].shape==(len(meta),35,4,4) and f['xi'].shape==(len(meta),4,68)
        for key,value in f.items():assert np.isfinite(value).all(),(v,key)
        assert len(a['valid'])==len(rows) and a['embedding'].shape==(len(rows),128)
        if a['valid'].any():appnorm.append(float(np.median(np.linalg.norm(a['embedding'][a['valid']].astype(float),axis=1))))
        for i in np.linspace(0,len(meta)-1,min(64,len(meta)),dtype=int):
            c=f['coefficients'][i].astype(float);nodes=collocation(c)
            for name in ['xi','xi_dot']:maximum_error=max(maximum_error,float(np.max(abs(nodes[name]-f[name][i]))))
            for name,d in [('endpoint',0),('endpoint_velocity',1),('endpoint_accel',2)]:maximum_error=max(maximum_error,float(np.max(abs((basis(1.,d)*2**d)@c-f[name][i]))))
        counts['observations']+=len(rows);counts['spatial_valid']+=sum(r['spatial_valid'] for r in rows);counts['rescued']+=sum(r['spatial_valid'] and not r['baseline_spatial_valid'] for r in rows);counts['windows']+=len(meta);counts['valid_windows']+=sum(r['valid'] for r in meta);counts['appearance_valid']+=int(a['valid'].sum())
    assert maximum_error<.01,maximum_error
    for name in ['train','validation','test']:
        index=pq.read_table(directory('indices')/f'{name}_index.parquet').to_pylist();assert all(r['video_id'] in cfg[name] and r['sequence_length']==16 for r in index)
        counts[name+'_sequences']=len(index)
    resources=[json.loads(line) for line in (report/'resources.jsonl').read_text().splitlines()];memory=dict(peak_process_tree_rss_gib=max(r['rss_gib'] for r in resources),minimum_system_available_gib=min(r['available_gib'] for r in resources),oom_observed=False,interpretation='Summed process RSS includes shared mapped pages; available memory measures system reserve. Resource guards abort at 20 GiB tree RSS or 5 GiB available.')
    scorecard=json.loads((report/'scorecard.json').read_text());assert scorecard['numerical_rebuild']['passed']
    result=dict(ready=True,ingestion_ready=True,scientific_acceptance_gates_passed=scorecard['all_five_scientific_acceptance_gates_passed'],scope='Annotated-contour reconstruction and fixed-budget temporal functional states; no forecasting or public VISOR segmentation parity claim',**source_identity(),counts=dict(counts),analytical_rebuild_max_absolute_error=maximum_error,verified_baseline_files=baseline_count,appearance_median_norm_by_video=appnorm,resources=memory,public_benchmark_parity=False,selection_stop=json.loads((report/'selection_locked.json').read_text())['stopped'])
    atomic_json(ROOT/'derived/track1_refined_v2/status.json',result);write_report('readiness.json',result)
    paths=list((ROOT/'src/refinement').glob('*.py'))+[ROOT/'configs/track1_refinement.yaml',ROOT/'tests/test_refinement.py',ROOT/'REFINEMENT.md',ROOT/'configs/track1_evaluation_brief.txt',ROOT/'work/refinement_visuals.py',ROOT/'work/refinement_capacity.py']+list((ROOT/'derived/track1_refined_v2').rglob('*'))+list(report.glob('*.json'))+list(report.glob('*.parquet'))+list(report.glob('geometry.*.jpg'))+list(report.glob('*.html'))+list(report.glob('*.png'))+list(report.glob('visual_overview.jpg'))
    paths=sorted({p for p in paths if p.is_file() and p.name not in ['freeze_manifest.json']})
    atomic_json(report/'freeze_manifest.json',dict(files={str(p.relative_to(ROOT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in paths},baseline_verified_files=baseline_count,**source_identity()))
    with tarfile.open(report/'implementation_snapshot.tar.gz','w:gz') as archive:
        for p in paths:
            if 'derived' not in p.relative_to(ROOT).parts:archive.add(p,arcname=str(p.relative_to(ROOT)))
    LOG.info('Refined v2 READY %s',result);return result
