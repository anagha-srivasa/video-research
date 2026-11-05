"""Twelve-family diagnostic scorecard. Proxy evidence never replaces reference truth."""
from collections import defaultdict
import json,html
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import norm
from scipy.spatial import cKDTree
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import load_shard,save_index
from track1.numerics import basis,wavelet_matrix
from track1.spatial import fit_spline
from .protocol import config,folder,rank,LOG
from .build import directory
from .data import temporal_cases,spatial_cases,holdout
from .kernels import spatial_fit,spatial_errors,periodic_design,temporal_fit,nearest_curve


def reliability(error,variance):
    e=np.asarray(error).ravel();v=np.maximum(np.asarray(variance).ravel(),1e-15);sigma=np.sqrt(v);z=e/sigma;out=dict(n=len(e),mean_scalar_nis=float(np.mean(z*z)),gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*v)+z*z))))
    for nominal in [.5,.68,.9,.95]:
        radius=norm.ppf((1+nominal)/2)*sigma;alpha=1-nominal;miss=np.maximum(abs(e)-radius,0)
        out[str(nominal)]=dict(coverage=float(np.mean(abs(e)<=radius)),mean_interval_width=float(np.mean(2*radius)),interval_score=float(np.mean(2*radius+2*miss/alpha)))
    if np.std(sigma)>0 and np.std(abs(e))>0:out['error_uncertainty_correlation']=float(np.corrcoef(sigma,abs(e))[0,1])
    order=np.argsort(sigma);cs=np.cumsum(e[order]**2)/np.arange(1,len(e)+1);steps=np.linspace(max(1,len(e)//20),len(e),20,dtype=int);out['risk_coverage']=[dict(retained=float(i/len(e)),rmse=float(np.sqrt(cs[i-1]))) for i in steps]
    out['risk_coverage_auc_mse']=float(np.mean(cs[steps-1]));out['statistical_limit']='Marginal normal-error diagnostic; correlated samples and approximate conditional covariance. Scalar NIS is not a full-state NEES validation.'
    return out


def representation(monitor):
    from shapely.geometry import LinearRing
    cfg=config();selection=json.loads((folder('reports')/'spatial_selection.json').read_text())['options'];model=json.loads((directory('calibration')/'model.json').read_text());w=wavelet_matrix();records=[];errors=defaultdict(list);max_inverse=0.;min_eigen=0.;negative=0;total=0;periodic=[];rate=[]
    for d in range(3):periodic.append(float(np.max(abs(periodic_design(np.array([0.,1.]),derivative=d)[0]-periodic_design(np.array([0.,1.]),derivative=d)[1]))))
    for v in cfg['train']+cfg['validation']+cfg['test']:
        s=load_shard(directory('spatial')/f'{v}.pt');rows=pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist();lookup={r['mask_uid']:r for r in rows}
        for begin in range(0,len(rows),4096):
            cov=s['covariance'][begin:begin+4096].astype(float);eig=np.linalg.eigvalsh(cov);negative+=int(np.sum(eig[:,0]<-1e-7));min_eigen=min(min_eigen,float(eig.min()));total+=len(cov)
        for case in spatial_cases([v]):
            r=lookup[case['mask_uid']];i=r['row_index'];y=case['y'];weights=np.ones(128);weights[::4]=0;fit=spatial_fit(y,selection,case['geometry_weight']*case['provenance_weight'],weights)
            if not fit['valid']:
                ww=np.ones(64);ww[::4]=0;old=fit_spline(y[::2],__import__('refinement.protocol',fromlist=['baseline_config']).baseline_config()['spatial'],case['geometry_weight']*case['provenance_weight'],sample_weights=ww)
                if not old['self_intersection']:fit=dict(controls=old['controls'],covariance=old['covariance'],valid=True)
            e,var,_=spatial_errors(y[::8],fit)
            if fit['valid']:errors[(case['split'],case['source_kind'])].append((e,var*model['kappa_cov']))
            p=s['controls'][i].astype(float);centre=s['centre_native'][i].astype(float);scale=float(s['scale_native'][i]);z=s['h'][i,2:34].astype(float).reshape(16,2);reconstructed=w.T@z*scale+centre
            max_inverse=max(max_inverse,float(np.max(abs(reconstructed-p))))
            curve=periodic_design(np.arange(256)/256)@p;simple=bool(LinearRing(curve).is_simple);signed=lambda a:float(np.sum(a[:,0]*np.roll(a[:,1],-1)-a[:,1]*np.roll(a[:,0],-1)))
            entity='hand' if 'hand' in r['entity_name'].lower() else 'object';objscale=max(float(np.sqrt(np.mean(np.sum((y-y.mean(0))**2,axis=1)))),1)
            records.append(dict(video_id=v,split=case['split'],source_kind=case['source_kind'],entity_type=entity,mask_uid=case['mask_uid'],small_object=bool(scale<32),spatial_valid=r['spatial_valid'],self_intersection=not simple,orientation_consistent=bool(signed(curve)*signed(y)>0),scale_native=scale,component_count=r['component_count'],hole_count=r['hole_count'],pose_valid=r['pose_valid']))
            for retained in [4,8,12,16]:
                zz=z.copy();zz[retained:]=0;pp=w.T@zz*scale+centre;distance,_=nearest_curve(y,pp)
                rate.append(dict(video_id=v,split=case['split'],source_kind=case['source_kind'],entity_type=entity,mask_uid=case['mask_uid'],coefficients_per_axis=retained,retained_energy=float(np.sum(zz*zz)/max(np.sum(z*z),1e-20)),normalised_contour_rmse=float(np.sqrt(np.mean(distance**2))/objscale)))
        monitor.check();LOG.info('Representation diagnostics %s',v)
    save_index(folder('reports')/'representation_cases.parquet',records);save_index(folder('reports')/'rate_distortion.parquet',rate)
    return dict(periodic_closure_derivative_max_error=periodic,periodicity_note='Basis invariants, not learned accuracy.',wavelet_inverse_max_native_px=max_inverse,spatial_covariance_matrices=total,spatial_covariance_psd_violations=negative,minimum_spatial_covariance_eigenvalue=min_eigen,uncertainty={s+'/'+k:reliability(np.concatenate([x[0] for x in rs]),np.concatenate([x[1] for x in rs])) for (s,k),rs in errors.items()},stratification_note='Hands/other objects, scale, source, participant/video and component counts are measured. Rigidity, deformability and visibility labels are unavailable; never inferred from fit error.')


def functional(monitor):
    cfg=config();model=json.loads((directory('calibration')/'model.json').read_text());options=json.loads((folder('reports')/'selection_locked.json').read_text())['temporal']['options'];scale=np.asarray(model['feature_scale']);r2s=[];overlaps=[];robust=[];yield_records=[];min_eigen=0.;negative=0;total=0
    for v in cfg['train']+cfg['validation']+cfg['test']:
        s=load_shard(directory('spatial')/f'{v}.pt');sr=pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist();lookup={(r['entity_track_id'],r['timestamp_us']):r for r in sr};f=load_shard(directory('functional')/f'{v}.pt');rows=pq.read_table(directory('functional')/f'{v}.parquet').to_pylist();fm={(r['entity_track_id'],r['window_end_us']):r for r in rows};eligible=[r for r in rows if r['span_seconds']>=.6 and r['max_gap_seconds']<=.35]
        yield_records.append(dict(video_id=v,attempted_windows=len(rows),valid_windows=sum(r['valid'] for r in rows),coverage_eligible_windows=len(eligible),valid_eligible_windows=sum(r['valid'] for r in eligible),eligibility='Observed span >=0.6s and max uncovered gap <=0.35s, independent of conditioning/effective-sample validity.'))
        for begin in range(0,len(rows),1024):
            eig=np.linalg.eigvalsh(f['coefficient_cov'][begin:begin+1024].astype(float));negative+=int(np.sum(eig[:,:,0]<-1e-6));min_eigen=min(min_eigen,float(eig.min()));total+=eig.shape[0]*eig.shape[1]
        cases=temporal_cases([v]);camera=json.loads((directory('spatial')/f'{v}.camera.json').read_text());k=np.asarray(camera['intrinsic_native']);width,height=camera['native_size'];diag=np.hypot(width,height)
        for case_number,case in enumerate(cases):
            t=case['time'];h=case['h'];r=fm.get((case['entity_track_id'],case['window_end_us']))
            if r is None:continue
            c=f['coefficients'][r['row_index']].astype(float);pred=basis(2*t-1)@c;error=pred-h;ss=np.sum((h-h.mean(0))**2,axis=0);good=ss>len(h)*scale**2*1e-4;r2=1-np.sum(error**2,axis=0)[good]/ss[good]
            # Irregular first differences are observation-derived proxies, not independent truth.
            dt=np.diff(t);velocity=np.diff(h,axis=0)/dt[:,None];predv=basis(2*(t[:-1]+t[1:])/2-1,1)@c*2;verror=predv-velocity
            centre_error=error[:,:2]*[k[0,0],k[1,1]];normalised_centre=float(np.sqrt(np.mean(np.sum(centre_error**2,axis=1)))/diag)
            residual_spectrum=np.fft.rfft((error/scale),axis=0);energy=np.sum(abs(residual_spectrum)**2,axis=1);high=float(energy[len(energy)//2:].sum()/max(energy.sum(),1e-20))
            r2s.append(dict(video_id=v,split=case['split'],entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],nonconstant_channels=int(good.sum()),median_in_sample_r2=float(np.median(r2)) if len(r2) else None,position_frame_diagonal_nrmse=normalised_centre,velocity_observation_proxy_nrmse=float(np.sqrt(np.mean((verror/scale)**2))),high_frequency_residual_energy_fraction=high,reference='Sample-index FFT is not a physical-Hz event test. Observed states, often interpolated; velocity and frequency are proxies. In-sample R2 is not held-out generalisation. Pixel error uses local intrinsic-plane focal scale; not a world-motion metric.'))
            previous=fm.get((r['entity_track_id'],r['window_end_us']-250000))
            if previous and previous['valid'] and r['valid']:
                a=f['coefficients'][previous['row_index']].astype(float);q=np.linspace(.25,1,7);pa=basis(2*q-1);pb=basis(2*(q-.25)-1);da=(pa@a-pb@c)/scale;dv=(basis(2*q-1,1)@a-basis(2*(q-.25)-1,1)@c)*2/scale
                overlaps.append(dict(video_id=v,split=case['split'],entity_track_id=r['entity_track_id'],window_end_us=r['window_end_us'],position_disagreement_nrmse=float(np.sqrt(np.mean(da**2))),velocity_disagreement_nrmse=float(np.sqrt(np.mean(dv**2))),handoff_position_nrmse=float(np.sqrt(np.mean(da[-1]**2))),handoff_velocity_nrmse=float(np.sqrt(np.mean(dv[-1]**2)))))
            # Fixed diagnostics on train/validation; no parameter selection from corruption outcomes.
            if case['split']=='test' or case_number>=16:continue
            target=holdout(t,'endpoint100');base=~target
            if not target.any() or base.sum()<8:continue
            rng=np.random.default_rng(int(rank(str(case['window_end_us'])+case['entity_track_id'])[:8],16));u=rng.random(len(t))
            for power in [0.,1.]:
                oo=dict(options,variance_power=power);ff=temporal_fit(t[base],h[base],case['variance'][base]*model['kappa_cov'],case['prior'][base],case['quality'][base],oo,scale);ee=basis(2*t[target]-1)@ff['coefficients']-h[target]
                robust.append(dict(video_id=v,split=case['split'],entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],corruption='variance_weighting_ablation',fraction=power,normalised_rmse=float(np.sqrt(np.mean((ee/scale)**2))),mean_std_normalised=float(np.mean(np.sqrt(np.maximum(np.diagonal(ff['covariance'],axis1=1,axis2=2),0)))),effective_samples=ff['neff']))
            for corruption in ['dropout','outlier']:
                for amount in [0.,.1,.25,.5]:
                    keep=base&(u>=amount) if corruption=='dropout' else base.copy()
                    if keep.sum()<8:continue
                    yy=h[keep].copy()
                    if corruption=='outlier':yy[u[keep]<amount]+=3*scale
                    fit=temporal_fit(t[keep],yy,case['variance'][keep]*model['kappa_cov'],case['prior'][keep],case['quality'][keep],options,scale);ph=basis(2*t[target]-1);e=ph@fit['coefficients']-h[target];vv=np.einsum('ni,dij,nj->nd',ph,fit['covariance'],ph)*np.asarray(model['temporal_multipliers'])
                    robust.append(dict(video_id=v,split=case['split'],entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],corruption=corruption,fraction=amount,normalised_rmse=float(np.sqrt(np.mean((e/scale)**2))),mean_std_normalised=float(np.mean(np.sqrt(np.maximum(vv,0))/scale)),effective_samples=fit['neff']))
        monitor.check();LOG.info('Functional diagnostics %s',v)
    for name,data in [('functional_fidelity',r2s),('overlap_consistency',overlaps),('robustness',robust),('yield',yield_records)]:save_index(folder('reports')/(name+'.parquet'),data)
    valid=sum(r['valid_windows'] for r in yield_records);attempted=sum(r['attempted_windows'] for r in yield_records);eligible=sum(r['coverage_eligible_windows'] for r in yield_records);ev=sum(r['valid_eligible_windows'] for r in yield_records)
    return dict(overlap_pairs=len(overlaps),overlap_position_disagreement_nrmse=float(np.median([r['position_disagreement_nrmse'] for r in overlaps])) if overlaps else None,overlap_velocity_disagreement_nrmse=float(np.median([r['velocity_disagreement_nrmse'] for r in overlaps])) if overlaps else None,overlap_covariance_consistency='Unverified: shared-frame cross-window covariance is not stored. Marginal covariance sums would assume false independence.',median_in_sample_r2=float(np.median([r['median_in_sample_r2'] for r in r2s if r['median_in_sample_r2'] is not None])),median_position_frame_diagonal_nrmse=float(np.median([r['position_frame_diagonal_nrmse'] for r in r2s])),valid_windows=valid,attempted_windows=attempted,all_attempted_yield=valid/attempted,coverage_eligible_windows=eligible,valid_coverage_eligible_windows=ev,coverage_eligible_yield=ev/max(eligible,1),temporal_covariance_psd_violations=negative,temporal_covariance_matrices=total,minimum_temporal_covariance_eigenvalue=min_eigen,true_velocity_accuracy='Unverified: needs independently reviewed high-frame-rate trajectories, not interpolated masks.',fast_event_retention='Unverified: needs independently annotated motion/contact onset and offset.')


def scorecard(monitor):
    rebuild=numerical_rebuild(monitor);spatial=representation(monitor);temporal=functional(monitor);appearance=appearance_pooling(monitor);silhouette=silhouette_details(monitor);evaluation=json.loads((folder('reports')/'evaluation.json').read_text());cfg=config();test=evaluation['spatial']['splits']['test']['refined'];unc=spatial['uncertainty']['test/manual_sparse']['0.95']['coverage'];gates=[dict(name='1 Data integrity',status='supported',evidence='Frozen canonical/v1 checks and participant/video disjoint splits; baseline SHA audit repeated at freeze.',limits='Independent inter-annotator agreement unavailable.'),dict(name='2 Spatial geometry',status='pass' if test['iou']>=.85 and test['boundary_fscore_2px']>=.85 else 'below proposed gate',evidence=f"Reserved manual mask mean IoU={test['iou']:.4f}, strict 2px BF={test['boundary_fscore_2px']:.4f}",limits='Dominant-ring model omits holes/components. Strict 2px tolerance differs from scale-aware BF gate.'),dict(name='3 Temporal fidelity',status='partially verified',evidence=f"Median nonconstant-channel in-sample R2={temporal['median_in_sample_r2']:.4f}; held-out RMSE and overlap disagreement reported.",limits='True endpoint velocity, world-motion preservation and independent fast-event retention unverified.'),dict(name='4 Uncertainty reliability',status='partial' if .92<=unc<=.98 else 'coverage outside proposed gate',evidence=f"Manual held-out contour 95% coverage={unc:.4f}; spatial PSD violations={spatial['spatial_covariance_psd_violations']}; temporal PSD violations={temporal['temporal_covariance_psd_violations']}",limits='Marginal empirical representation uncertainty, not joint Gaussian NEES or human-label uncertainty.'),dict(name='5 Downstream readiness',status='unverified identity gate',evidence=f"All attempted yield={temporal['all_attempted_yield']:.4f}; coverage-eligible yield={temporal['coverage_eligible_yield']:.4f}.",limits='Provisional annotation chains lack independent identity GT; IDF1/HOTA cannot be certified.')]
    families={
      'A Integrity':dict(status='existing verified evidence plus split audit',artifacts=['dataset/metadata/source_integrity.json','canonical/v1/status.json','reports/track1_readiness.json'],missing=['Independent annotation agreement']),
      'B Camera':dict(status='intrinsic-normalised 2D; no scene-depth stabilization',artifacts=['reports/track1_camera.*.json'],missing=['Static-background stabilization/foreground preservation reference','Metric scale and static-object truth']),
      'C Silhouette':dict(status='measured manual reconstruction',artifacts=['spatial_geometry.parquet'],missing=['Occlusion/contact-specific labels']),
      'D Spline':dict(status='measured numerical and geometric checks',artifacts=['representation_cases.parquet','spatial_withheld.parquet']),
      'E Wavelet':dict(status='measured inverse and rate-distortion',artifacts=['rate_distortion.parquet']),
      'F Uncertainty':dict(status='marginal calibration measured; multivariate reference validation unavailable',artifacts=['scorecard.json','calibration.json'],missing=['Independent full-state/reference annotation covariance']),
      'G Identity':dict(status='unverified against independent identity ground truth',artifacts=[],missing=['IDF1/HOTA ground-truth identity trajectories and visible track seconds']),
      'H Temporal':dict(status='held-out and in-sample diagnostics measured',artifacts=['temporal_cases.parquet','functional_fidelity.parquet','cubic_capacity.parquet'],missing=['True high-frame-rate velocity/acceleration references']),
      'I Continuity':dict(status='shared-time disagreement measured',artifacts=['overlap_consistency.parquet'],missing=['Shared-frame cross-window covariance for statistical consistency','Independent short-event labels']),
      'J Robustness':dict(status='deterministic dropout/outlier stress curves measured',artifacts=['robustness.parquet'],missing=['Real occlusion/viewpoint ground truth']),
      'K Appearance':dict(status='frozen CNN and train-only PCA measured',artifacts=['derived/track1_refined_v2/appearance/projection.json'],missing=['Independent foreground leakage/background replacement and identity retrieval labels']),
      'L System':dict(status='schema, analytic derivatives, PSD, hashes, memory and tests gated at freeze',artifacts=['readiness.json','resources.jsonl','tests.json'])}
    result=dict(evaluation_brief_sha256=digest(ROOT/'configs/track1_evaluation_brief.txt'),proposed_thresholds_are_public_benchmarks=False,all_five_scientific_acceptance_gates_passed=False,gates=gates,families=families,spatial=spatial,temporal=temporal,appearance=appearance,silhouette=silhouette,numerical_rebuild=rebuild,meaning='Ingestion-ready numerical artifacts do not imply independent scientific acceptance for Track 2. Missing ground truth is an explicit gate, never replaced by a reconstruction proxy.')
    enrich_scorecard(result);atomic_json(folder('reports')/'scorecard.json',result);dashboard(result);plots(result);return result


def dashboard(result):
    report=folder('reports');geom=pq.read_table(report/'spatial_geometry.parquet').to_pylist();rep={r['mask_uid']:r for r in pq.read_table(report/'representation_cases.parquet').to_pylist()};rows=[]
    for r in geom:
        q=rep.get(r['mask_uid'],{});rows.append(dict(video=r['video_id'],split=r['split'],entity=q.get('entity_type','unknown'),provenance='manual_sparse',method=r['method'],iou=r['iou'],bf=r['boundary_fscore_2px'],distance=r['symmetric_boundary_distance_px'],valid=r['valid']))
    esc=html.escape;body=''.join('<tr><td>'+esc(g['name'])+'</td><td>'+esc(g['status'])+'</td><td>'+esc(g['evidence'])+'</td><td>'+esc(g['limits'])+'</td></tr>' for g in result['gates'])
    data=json.dumps(rows).replace('</','<\\/');page='''<!doctype html><meta charset="utf-8"><title>Track 1 evaluation</title><style>body{font:15px system-ui;max-width:1400px;margin:36px auto;padding:0 20px;background:#10151c;color:#e2e8f0}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #344154}select{margin:10px;padding:8px}a{color:#7dd3fc}.note{color:#bec8d8}button{padding:8px}</style><h1>Track 1 evaluation</h1><p>Annotated-contour reconstruction and fixed-budget temporal states. Proposed engineering gates; no public VISOR segmentation parity claim.</p><h2>Scientific acceptance gates</h2><table><thead><tr><th>Gate</th><th>Status</th><th>Measured evidence</th><th>Limit</th></tr></thead><tbody>'''+body+'''</tbody></table><p class="note">Independent identity, true velocity, fast events and cross-window covariance remain unverified. Ingestion readiness is reported separately.</p><h2>Manual silhouette reconstruction</h2><p>Participant/video, entity type and method filters below apply to measured manual masks. Visibility and rigidity labels are unavailable.</p><div id="filters"></div><p id="summary"></p><table><thead><tr><th>Video</th><th>Split</th><th>Entity</th><th>Method</th><th>IoU</th><th>BF 2px</th><th>Boundary px</th><th>Valid</th></tr></thead><tbody id="rows"></tbody></table><h2>Full evidence</h2><p><a href="scorecard.json">12-family scorecard</a> · <a href="evaluation.json">Held-out evaluation</a> · <a href="readiness.json">Ingestion gates</a> · <a href="quality.png">Quality plots</a></p><script>const data='''+data+''';const keys=['video','split','entity','method'];let filters={};for(const key of keys){const s=document.createElement('select');s.setAttribute('aria-label',key);for(const v of ['All '+key,...new Set(data.map(r=>r[key]))]){const o=document.createElement('option');o.textContent=v;s.append(o)}s.onchange=()=>{filters[key]=s.selectedIndex?s.value:null;render()};document.querySelector('#filters').append(s)}function render(){const rs=data.filter(r=>keys.every(k=>!filters[k]||r[k]===filters[k]));const mean=k=>rs.reduce((s,r)=>s+r[k],0)/Math.max(rs.length,1);document.querySelector('#summary').textContent=rs.length+' mask/method rows; sample mean IoU '+mean('iou').toFixed(4)+', BF '+mean('bf').toFixed(4)+' (official report uses video-macro means).';const el=document.querySelector('#rows');el.replaceChildren();for(const r of rs){const tr=document.createElement('tr');for(const v of [r.video,r.split,r.entity,r.method,r.iou.toFixed(4),r.bf.toFixed(4),r.distance.toFixed(3),r.valid]){const td=document.createElement('td');td.textContent=v;tr.append(td)}el.append(tr)}}render();</script>'''
    test=json.loads((report/'test_evaluation.json').read_text());cards=[('Geometry',f"IoU {test['spatial']['refined']['iou']:.3f}; scale BF {result['silhouette']['splits']['test']['refined']['boundary_fscore_scale']:.3f}"),('Functional fit',f"Median in-sample R² {result['temporal']['median_in_sample_r2']:.3f}; held-out tests linked below"),('Uncertainty',f"Manual 95% coverage {result['spatial']['uncertainty']['test/manual_sparse']['0.95']['coverage']:.3f}; marginal calibration"),('Tracking','IDF1 / HOTA unverified; provisional chains'),('Continuity',f"Overlap state disagreement {result['temporal']['overlap_position_disagreement_nrmse']:.3f} normalized; joint covariance unavailable"),('Reliability',f"Eligible yield {result['temporal']['coverage_eligible_yield']:.2%}; all-attempted yield {result['temporal']['all_attempted_yield']:.2%}")]
    panels='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">'+''.join('<div style="padding:16px;border:1px solid #344154;border-radius:8px"><strong>'+esc(name)+'</strong><p>'+esc(value)+'</p></div>' for name,value in cards)+'</div>'
    page=page.replace('<h2>Scientific acceptance gates</h2>',panels+'<h2>Scientific acceptance gates</h2>')
    (report/'dashboard.html').write_text(page)


def appearance_pooling(monitor):
    import duckdb
    from pipeline import Dataset
    from track1.spatial import soft_masks
    cfg=config();cal=json.loads((directory('calibration')/'model.json').read_text());records=[]
    for v in cfg['train']+cfg['validation']+cfg['test']:
        s=load_shard(directory('spatial')/f'{v}.pt');rows=pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist();manual=sorted([r for r in rows if r['source_kind']=='manual_sparse' and r['spatial_valid']],key=lambda r:rank(r['mask_uid']))[:48]
        with duckdb.connect() as db:
            db.execute('SET threads=1');holes={uid:([r for r in json.loads(rings) if r['role']=='hole'],aw,ah) for uid,rings,aw,ah in db.execute('SELECT mask_uid,contours_json,annotation_width,annotation_height FROM read_parquet(?) WHERE hole_count>0',[str(ROOT/f'canonical/v1/masks.{v}.parquet')]).fetchall()}
        with Dataset() as ds:
            for r in manual:
                i=r['row_index'];width,height=r['width'],r['height'];grid=np.stack(np.meshgrid((np.arange(14)+.5)*width/14,(np.arange(8)+.5)*height/8),axis=-1).reshape(-1,2);sigma=np.sqrt(s['normal_variance'][i]*cal['kappa_cov']);core,halo=soft_masks(periodic_design(np.arange(64)/64)@s['controls'][i],sigma,grid,tau=max(width/448,1.))
                if r['mask_uid'] in holes:
                    rings,aw,ah=holes[r['mask_uid']]
                    for ring in rings:
                        p=np.asarray(ring['points'])*[width/aw,height/ah];_,hh=soft_masks(p,np.full(len(p),max(float(np.median(sigma)),1.)),grid,tau=max(width/448,1.));core*=1-hh
                target=ds.get_mask(r['mask_uid']);mh,mw=target.shape;xy=np.clip((grid*[mw/width,mh/height]).astype(int),[0,0],[mw-1,mh-1]);fg=target[xy[:,1],xy[:,0]]>0;mass=float(core.sum());precision=float(core[fg].sum()/mass) if mass>1e-4 else None
                records.append(dict(video_id=v,split=__import__('refinement.protocol',fromlist=['split']).split(v),mask_uid=r['mask_uid'],entity_type='hand' if 'hand' in r['entity_name'].lower() else 'object',core_mass=mass,feature_cell_foreground_precision=precision,foreground_cell_coverage=float(np.mean(core[fg]>=.5)) if fg.any() else None,has_foreground_feature_cells=bool(fg.any())))
        monitor.check()
    save_index(folder('reports')/'appearance_pooling.parquet',records);good=[r for r in records if r['feature_cell_foreground_precision'] is not None]
    return dict(samples=len(records),usable_core_samples=len(good),mean_feature_cell_foreground_precision=float(np.mean([r['feature_cell_foreground_precision'] for r in good])),empty_foreground_grid_samples=sum(not r['has_foreground_feature_cells'] for r in records),limit='Geometry-only foreground membership at actual 8x14 CNN cell centres with source-hole exclusion. CNN receptive fields span background; this does not certify neural background contamination or semantic identity retrieval.')


def silhouette_details(monitor):
    import cv2
    from pipeline import Dataset
    cfg=config();geometry=pq.read_table(folder('reports')/'spatial_geometry.parquet').to_pylist();records=[];phi=periodic_design(np.arange(256)/256)
    for v in cfg['train']+cfg['validation']+cfg['test']:
        new=load_shard(directory('spatial')/f'{v}.pt');old=load_shard(ROOT/f'derived/track1_spatial_v1/{v}.pt');lookup={r['mask_uid']:r for r in pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist()};rs=[r for r in geometry if r['video_id']==v and r['method']=='refined']
        with Dataset() as ds:
            for r in rs:
                meta=lookup[r['mask_uid']];target=ds.get_mask(r['mask_uid']).astype(np.uint8);h,w=target.shape;ys,xs=np.nonzero(target);scale=max(float(np.hypot(xs.max()-xs.min()+1,ys.max()-ys.min()+1)),1);tol=max(1.,.01*scale);bs=target-cv2.erode(target,np.ones((3,3),np.uint8));dist_s=cv2.distanceTransform(1-bs,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);area=int(target.sum());centroid=np.array([xs.mean(),ys.mean()])
                for method,state in [('baseline',old),('refined',new)]:
                    polygon=np.rint(phi@state['controls'][meta['row_index']]*[w/meta['width'],h/meta['height']]).astype(np.int32);fit=np.zeros_like(target);cv2.fillPoly(fit,[polygon],1);bf=fit-cv2.erode(fit,np.ones((3,3),np.uint8));df=cv2.distanceTransform(1-bf,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);d1=dist_s[bf>0];d2=df[bs>0];precision=float(np.mean(d1<=tol)) if len(d1) else 0.;recall=float(np.mean(d2<=tol)) if len(d2) else 0.;yy,xx=np.nonzero(fit);c=np.array([xx.mean(),yy.mean()]) if len(xx) else np.array([np.inf,np.inf]);farea=int(fit.sum());intersection=int(np.sum((fit>0)&(target>0)));all_dist=np.r_[d1,d2]
                    records.append(dict(video_id=v,split=r['split'],mask_uid=r['mask_uid'],method=method,boundary_tolerance_px=tol,object_bbox_diagonal_px=scale,dice=2*intersection/max(area+farea,1),boundary_precision=precision,boundary_recall=recall,boundary_fscore_scale=2*precision*recall/max(precision+recall,1e-20),assd_normalised=float((d1.mean()+d2.mean())/2/scale),hd95_normalised=float(np.quantile(all_dist,.95)/scale),relative_area_error=abs(farea-area)/max(area,1),centroid_error_normalised=float(np.linalg.norm(c-centroid)/scale)))
        monitor.check()
    save_index(folder('reports')/'silhouette_details.parquet',records);result={}
    for split in ['train','validation','test']:
        result[split]={}
        for method in ['baseline','refined']:
            keys=['dice','boundary_precision','boundary_recall','boundary_fscore_scale','assd_normalised','hd95_normalised','relative_area_error','centroid_error_normalised'];vs=[{k:float(np.mean([r[k] for r in records if r['video_id']==v and r['method']==method])) for k in keys} for v in cfg[split]];result[split][method]={k:float(np.mean([r[k] for r in vs])) for k in keys}
    return dict(splits=result,tolerance='max(1 annotation pixel, 1% of source full-mask bounding-box diagonal). Mean of per-video means. Full-mask topology retained in reference.')


def plots(result):
    import os
    os.environ['MPLCONFIGDIR']=str(ROOT/'work/mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True);ev=json.loads((folder('reports')/'test_evaluation.json').read_text());protocols=['single','gap100','endpoint100'];x=np.arange(3)
    for i,method in enumerate(['baseline','refined','nearest','linear']):axes[0,0].bar(x+(i-1.5)*.2,[ev['temporal'][p]['methods'][method]['normalised_rmse'] for p in protocols],.2,label=method)
    axes[0,0].set(xticks=x,xticklabels=protocols,ylabel='Train-standardised RMSE',title='Reserved videos: temporal holdouts');axes[0,0].legend(fontsize=8)
    axes[0,1].bar(['Baseline','Refined'],[ev['spatial'][m]['withheld_normal_rmse_px'] for m in ['baseline','refined']],color=['#b46341','#207b83']);axes[0,1].set(ylabel='Native pixels',title='Reserved videos: withheld normal error')
    for source in ['manual_sparse','dense_gt_endpoint','dense_interpolated']:
        r=result['spatial']['uncertainty']['test/'+source];axes[0,2].plot([.5,.68,.9,.95],[r[str(q)]['coverage'] for q in [.5,.68,.9,.95]],'o-',label=source)
    axes[0,2].plot([.5,1],[.5,1],'k--');axes[0,2].set(xlabel='Nominal coverage',ylabel='Measured coverage',title='Reserved spatial uncertainty');axes[0,2].legend(fontsize=8)
    rate=pq.read_table(folder('reports')/'rate_distortion.parquet').to_pylist()
    for source in ['manual_sparse','dense_gt_endpoint','dense_interpolated']:
        axes[1,0].plot([4,8,12,16],[np.mean([r['normalised_contour_rmse'] for r in rate if r['split']=='test' and r['source_kind']==source and r['coefficients_per_axis']==k]) for k in [4,8,12,16]],'o-',label=source)
    axes[1,0].set(xlabel='db2 coefficients per axis',ylabel='Contour RMSE / object RMS radius',title='Reserved rate–distortion');axes[1,0].legend(fontsize=8)
    stress=pq.read_table(folder('reports')/'robustness.parquet').to_pylist()
    for kind in ['dropout','outlier']:
        axes[1,1].plot([0,.1,.25,.5],[np.mean([r['normalised_rmse'] for r in stress if r['corruption']==kind and r['fraction']==q]) for q in [0,.1,.25,.5]],'o-',label=kind)
    axes[1,1].set(xlabel='Corrupted input fraction',ylabel='Endpoint normalised RMSE',title='Train/validation stress diagnostics');axes[1,1].legend()
    resources=[json.loads(line) for line in (folder('reports')/'resources.jsonl').read_text().splitlines()];axes[1,2].plot([r['rss_gib'] for r in resources],label='Tree RSS (shared pages counted)');axes[1,2].plot([r['available_gib'] for r in resources],label='System available');axes[1,2].set(xlabel='Logged sample (multiple processes)',ylabel='GiB',title='Resource monitoring');axes[1,2].legend(fontsize=8)
    fig.suptitle('Track 1 v2: matched-input evidence, fixed 16-control / 4-coefficient budget');fig.savefig(folder('reports')/'quality.png',dpi=160);plt.close(fig)


def numerical_rebuild(monitor):
    import duckdb
    from track1.spatial import dominant_ring,resample
    cfg=config();selected=json.loads((folder('reports')/'selection_locked.json').read_text());cal=json.loads((directory('calibration')/'model.json').read_text());spatial_samples=0;temporal_samples=0;max_c=0.;max_cov=0.
    for v in cfg['train']+cfg['validation']+cfg['test']:
        s=load_shard(directory('spatial')/f'{v}.pt');old=load_shard(ROOT/f'derived/track1_spatial_v1/{v}.pt');rows=pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist();ids=np.linspace(0,len(rows)-1,24,dtype=int);wanted=[rows[i]['mask_uid'] for i in ids]
        with duckdb.connect() as db:
            db.execute('SET threads=1');geometry={uid:(rings,aw,ah) for uid,rings,aw,ah in db.execute('SELECT mask_uid,contours_json,annotation_width,annotation_height FROM read_parquet(?) WHERE mask_uid IN (SELECT unnest(?))',[str(ROOT/f'canonical/v1/masks.{v}.parquet'),wanted]).fetchall()}
        for i in ids:
            r=rows[i];rings,aw,ah=geometry[r['mask_uid']];ring,_,_=dominant_ring(dict(contours_json=rings),np.diag([r['width']/aw,r['height']/ah,1.]));y=np.roll(resample(ring,128),-2*r['seam_shift'],axis=0);fit=spatial_fit(y,selected['spatial']['options'],r['provenance_weight']*r['geometry_weight']);p=fit['controls'];cov=fit['covariance'];a,_=nearest_curve(y,old['controls'][i]);b,_=nearest_curve(y,p)
            if r['baseline_spatial_valid'] and (not fit['valid'] or np.mean(b*b)>np.mean(a*a)):p=old['controls'][i].astype(float);cov=old['covariance'][i].astype(float)
            assert np.array_equal(p.astype(np.float32),s['controls'][i]),(v,i,'spatial controls')
            assert np.array_equal(cov.astype(np.float32),s['covariance'][i]),(v,i,'spatial covariance')
            spatial_samples+=1
        tracks=defaultdict(list)
        for r in rows:
            if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
        f=load_shard(directory('functional')/f'{v}.pt');meta=pq.read_table(directory('functional')/f'{v}.parquet').to_pylist()
        for i in np.linspace(0,len(meta)-1,48,dtype=int):
            r=meta[i];rs=[q for q in tracks[r['entity_track_id']] if r['window_start_us']<=q['timestamp_us']<=r['window_end_us']];local=np.array([q['row_index'] for q in rs]);t=np.array([(q['timestamp_us']-r['window_start_us'])/1e6 for q in rs]);fit=temporal_fit(t,s['h'][local].astype(float),s['variance'][local].astype(float)*cal['kappa_cov'],np.array([q['provenance_weight'] for q in rs]),np.array([q['geometry_weight'] for q in rs]),selected['temporal']['options'],np.asarray(cal['feature_scale']));c=fit['coefficients'].astype(np.float32);cov=(fit['covariance']*np.asarray(cal['temporal_multipliers'])[:,None,None]).astype(np.float32)
            max_c=max(max_c,float(abs(c-f['coefficients'][i]).max()));max_cov=max(max_cov,float(abs(cov-f['coefficient_cov'][i]).max()));assert np.array_equal(c,f['coefficients'][i]),(v,i,'coefficients');assert np.array_equal(cov,f['coefficient_cov'][i]),(v,i,'coefficient covariance');temporal_samples+=1
        monitor.check()
    result=dict(passed=True,spatial_full_source_refits=spatial_samples,temporal_full_observation_refits=temporal_samples,float32_coefficients_and_covariances_exact=True,max_temporal_coefficient_error=max_c,max_temporal_covariance_error=max_cov,scope='Deterministic stratified row samples across every video, including invalid windows; not a full-store second build.')
    atomic_json(folder('reports')/'numerical_rebuild.json',result);return result


def enrich_scorecard(result):
    geometry=result['silhouette']['splits']['test']['refined'];result['gates'][1].update(status='below proposed boundary gate',evidence=f"Reserved full-mask IoU={json.loads((folder('reports')/'test_evaluation.json').read_text())['spatial']['refined']['iou']:.4f}; strict 2px BF={json.loads((folder('reports')/'test_evaluation.json').read_text())['spatial']['refined']['boundary_fscore_2px']:.4f}; scale-aware BF={geometry['boundary_fscore_scale']:.4f}.",limits='Scale-aware tolerance is max(1px, 1% source bounding-box diagonal). Both boundary scores remain below 0.85. Dominant rings omit holes/components.')
    result['gates'][2]['status']='below proposed R2 gate; true velocity unverified'
    result['families']['K Appearance']['geometry_proxy_precision']=result['appearance']['mean_feature_cell_foreground_precision'];result['families']['K Appearance']['limits']=f"Foreground feature-cell precision is below proposed 0.95; {result['appearance']['empty_foreground_grid_samples']}/{result['appearance']['samples']} masks have no foreground cell at this grid. This is not an independent CNN receptive-field leakage measurement."
    stress=pq.read_table(folder('reports')/'robustness.parquet').to_pylist();weighting={}
    for power in [0.,1.]:
        byvideo={}
        for v in config()['train']+config()['validation']:
            rs=[r for r in stress if r['video_id']==v and r['corruption']=='variance_weighting_ablation' and r['fraction']==power]
            if rs:byvideo[v]=float(np.sqrt(np.mean([r['normalised_rmse']**2 for r in rs])))
        weighting[str(power)]=byvideo
    result['essential_ablations']={
        'uniform_vs_confidence_weighting':dict(status='measured with identical retained observations and selected regularization; no retuning',endpoint_normalised_rmse_by_video=weighting,limits='Uniform here removes inverse-variance weighting only; both retain identical temporal coverage/provenance weights. Selected exponent is 0.25, between these controls.'),
        'direct_contours_vs_spline_wavelet':dict(status='rate-distortion measured',limits='Source has 128 2D samples; full spline has 16 2D controls and invertible 16-coefficient db2 per axis. Coordinate-count budget differs by 8x; retained-coefficient sweep reports this explicitly.'),
        'raw_vs_canonical':dict(status='unverified scene-motion ablation',limits='Intrinsic-normalized depth-free coordinates do not remove camera translation. No independent static/foreground motion references; normalization alone cannot certify nuisance-motion removal.'),
        'point_vs_propagated_uncertainty':dict(status='marginal coverage, NLL, risk-coverage and interval scores measured',limits='Deterministic fits are identical; uncertainty adds selective-rejection evidence, not an accuracy improvement or independent joint NEES proof.')}
    path=folder('reports')/'compression_limit.json'
    if path.exists():result['compression_limit']=json.loads(path.read_text())
    return result
