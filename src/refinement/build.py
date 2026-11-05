"""Version-two shards; frozen version-one sources are never overwritten."""
from collections import defaultdict
import json,time,logging
import hashlib
import numpy as np
import pyarrow.parquet as pq
import duckdb
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import load_shard,save_shard,save_index,memory_guard
from track1.spatial import dominant_ring,resample,wavelet_state
from track1.numerics import wavelet_matrix,basis
from track1.camera import undistort
from track1.temporal import collocation
from .protocol import config,folder,source_identity,LOG
from .kernels import spatial_fit,spatial_errors,nearest_curve,periodic_design,temporal_fit

def directory(stage):
    path=ROOT/'derived/track1_refined_v2'/stage;path.mkdir(parents=True,exist_ok=True);return path

def manifest(stage,video,outputs,inputs,metrics):
    atomic_json(directory(stage)/f'{video}.manifest.json',dict(**source_identity(),stage=stage,video_id=video,completed=True,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},outputs={str(p.relative_to(ROOT)):digest(p) for p in outputs},metrics=metrics))

_SPATIAL_CONTEXT={}

def spatial_video(video,start_index=0,stop_index=None,chunk=False):
    cfg=config();memory_guard(cfg);start=time.monotonic();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True)
    selected=json.loads((folder('reports')/'spatial_selection.json').read_text());options=selected['options']
    if options.get('baseline'):raise ValueError('No spatial refinement was selected')
    base=ROOT/'derived/track1_spatial_v1'
    if _SPATIAL_CONTEXT.get('video')!=video:
        _SPATIAL_CONTEXT.clear();old=load_shard(base/f'{video}.pt');rows=pq.read_table(base/f'{video}.parquet').to_pylist()
        camera=json.loads((base/f'{video}.camera.json').read_text())
        with duckdb.connect() as db:
            db.execute('SET threads=1');values=db.execute('SELECT mask_uid,contours_json,annotation_width,annotation_height FROM read_parquet(?) WHERE timestamp_us IS NOT NULL AND geometry_status!=?',
                [str(ROOT/f'canonical/v1/masks.{video}.parquet'),'quarantined']).fetchall()
        _SPATIAL_CONTEXT.update(video=video,old=old,rows=rows,camera=camera,geometry={uid:(rings,w,h) for uid,rings,w,h in values});values=None
    old=_SPATIAL_CONTEXT['old'];all_rows=_SPATIAL_CONTEXT['rows'];camera=_SPATIAL_CONTEXT['camera'];geometry=_SPATIAL_CONTEXT['geometry'];k=np.asarray(camera['intrinsic_native']);dist=np.asarray(camera['distortion'])
    stop_index=min(stop_index or len(all_rows),len(all_rows));rows=[dict(row) for row in all_rows[start_index:stop_index]]
    arrays={key:np.empty((len(rows),)+value.shape[1:],value.dtype) for key,value in old.items()};fallback=0;rescued=0;new_invalid=0;phi=periodic_design(np.arange(64)/64);phi1=periodic_design(np.arange(64)/64,derivative=1);w=wavelet_matrix();mean_phi=phi.mean(0)
    for i,row in enumerate(rows,start_index):
        if (i-start_index)%1000==0:memory_guard(cfg);LOG.info('%s refined spatial %d/%d elapsed=%.1fs',video,i,stop_index,time.monotonic()-start)
        rings,aw,ah=geometry[row['mask_uid']];transform=np.diag([row['width']/aw,row['height']/ah,1.]);ring,_,_=dominant_ring(dict(contours_json=rings),transform)
        y=resample(ring,options['samples']);y=np.roll(y,-row['seam_shift']*(options['samples']//64),axis=0)
        fit=spatial_fit(y,options,row['provenance_weight']*row['geometry_weight']);p=fit['controls'];cov=fit['covariance']
        old_error,_=nearest_curve(y,old['controls'][i]);new_error,_=nearest_curve(y,p)
        use_old=bool(row['spatial_valid'] and (not fit['valid'] or np.mean(new_error**2)>np.mean(old_error**2)))
        if use_old:p=old['controls'][i].astype(float);cov=old['covariance'][i].astype(float);fallback+=1
        valid=bool(row['spatial_valid'] or fit['valid'])
        if valid and not row['spatial_valid']:rescued+=1
        if not valid:new_invalid+=1
        centre=old['centre_native'][i].astype(float);scale=float(old['scale_native'][i]);area=np.pi*scale*scale
        z,zcov=wavelet_state(p,cov,centre,scale);centre_n=undistort(centre[None],k,dist)[0]
        h=np.r_[centre_n,z.reshape(-1),np.log(scale/np.sqrt(k[0,0]*k[1,1]))]
        centre_var=float(mean_phi@cov@mean_phi);curve=phi@p;gx=(np.roll(curve[:,1],-1)-np.roll(curve[:,1],1))/2;gy=(np.roll(curve[:,0],1)-np.roll(curve[:,0],-1))/2
        ax=phi.T@gx;ay=phi.T@gy;log_var=float((ax@cov@ax+ay@cov@ay)/(4*max(area*area,1)))
        shape_var=np.repeat(np.diag(zcov),2)+np.repeat((w@np.ones(16))**2*centre_var/scale**2,2)+z.reshape(-1)**2*log_var
        variance=np.r_[centre_var/k[0,0]**2,centre_var/k[1,1]**2,shape_var,log_var]
        tangent=phi1@p;normal=np.stack([tangent[:,1],-tangent[:,0]],axis=1);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        delta=old['boundary'][i]-curve
        values=dict(controls=p,covariance=cov,boundary=old['boundary'][i],normals=normal,normal_variance=np.maximum(np.einsum('qi,ij,qj->q',phi,cov,phi),0),residual_normal=np.sum(delta*normal,axis=1),h=h,variance=variance,world_bearing=old['world_bearing'][i],centre_native=centre,scale_native=np.array(scale))
        for key,value in values.items():arrays[key][i-start_index]=value
        row['baseline_spatial_valid']=row['spatial_valid'];row['spatial_valid']=valid;row['refinement_fallback']=use_old;row['refinement_steps']=fit['iterations'];row['baseline_spatial_rmse_px']=row['spatial_rmse_px'];row['spatial_rmse_px']=float(np.sqrt(np.mean((old_error if use_old else new_error)**2)));row['fit_error_metric']='nearest_curve_distance_native_px'
        row['quality_flags']=[f for f in row['quality_flags'] if f!='spline_self_intersection_or_nonfinite']
        if not valid:row['quality_flags'].append('spline_self_intersection_or_nonfinite')
    if chunk:
        out=folder('cache')/'spatial_chunks';out.mkdir(exist_ok=True);stem=f'{video}.{start_index:08d}'
    else:out=directory('spatial');stem=video
    tensor=out/f'{stem}.pt';index=out/f'{stem}.parquet';save_shard(tensor,arrays);save_index(index,rows)
    metrics=dict(observations=len(rows),valid=len(rows)-new_invalid,rescued=rescued,fallback=fallback,elapsed_seconds=time.monotonic()-start,start_index=start_index,stop_index=stop_index)
    if chunk:
        atomic_json(out/f'{stem}.json',dict(completed=True,video_id=video,**metrics,tensor_sha256=digest(tensor),index_sha256=digest(index),selection_sha256=digest(folder('reports')/'spatial_selection.json'),kernel_sha256=digest(ROOT/'src/refinement/kernels.py'),builder_sha256=digest(ROOT/'src/refinement/build.py')))
        return metrics
    camera_path=out/f'{video}.camera.json';atomic_json(camera_path,camera)
    manifest('spatial',video,[tensor,index,camera_path],[base/f'{video}.pt',base/f'{video}.parquet',ROOT/f'canonical/v1/masks.{video}.parquet',folder('reports')/'spatial_selection.json'],metrics)
    return metrics

def assemble_spatial(video):
    chunks=sorted((folder('cache')/'spatial_chunks').glob(f'{video}.*.json'));meta=[json.loads(p.read_text()) for p in chunks]
    assert meta and meta[0]['start_index']==0
    for a,b in zip(meta,meta[1:]):assert a['stop_index']==b['start_index']
    n=pq.read_metadata(ROOT/f'derived/track1_spatial_v1/{video}.parquet').num_rows;assert meta[-1]['stop_index']==n
    first=load_shard(chunks[0].with_suffix('.pt'));arrays={key:np.empty((n,)+value.shape[1:],value.dtype) for key,value in first.items()};rows=[]
    for path,info in zip(chunks,meta):
        assert info['tensor_sha256']==digest(path.with_suffix('.pt')) and info['index_sha256']==digest(path.with_suffix('.parquet'))
        assert info['selection_sha256']==digest(folder('reports')/'spatial_selection.json') and info['kernel_sha256']==digest(ROOT/'src/refinement/kernels.py') and info['builder_sha256']==digest(ROOT/'src/refinement/build.py')
        data=load_shard(path.with_suffix('.pt'))
        for key,value in data.items():arrays[key][info['start_index']:info['stop_index']]=value
        rows+=pq.read_table(path.with_suffix('.parquet')).to_pylist()
    out=directory('spatial');tensor=out/f'{video}.pt';index=out/f'{video}.parquet';save_shard(tensor,arrays);save_index(index,rows)
    camera=out/f'{video}.camera.json';atomic_json(camera,json.loads((ROOT/f'derived/track1_spatial_v1/{video}.camera.json').read_text()))
    metrics={key:sum(r[key] for r in meta) for key in ['observations','valid','rescued','fallback','elapsed_seconds']}
    manifest('spatial',video,[tensor,index,camera],[ROOT/f'derived/track1_spatial_v1/{video}.pt',ROOT/f'derived/track1_spatial_v1/{video}.parquet',folder('reports')/'spatial_selection.json'],metrics)
    return metrics

def functional_video(video):
    cfg=config();memory_guard(cfg);start=time.monotonic();model=json.loads((directory('calibration')/'model.json').read_text());options=json.loads((folder('reports')/'selection_locked.json').read_text())['temporal']['options'];scale=np.asarray(model['feature_scale'])
    state=load_shard(directory('spatial')/f'{video}.pt');rows=pq.read_table(directory('spatial')/f'{video}.parquet').to_pylist();appearance=load_shard(directory('appearance')/f'{video}.pt')
    tracks=defaultdict(list)
    for row in rows:
        if row['spatial_valid']:tracks[row['entity_track_id']].append(row)
    with duckdb.connect() as db:end_us=db.execute('SELECT duration_us FROM read_parquet(?)',[str(ROOT/f'canonical/v1/videos.{video}.parquet')]).fetchone()[0]
    arrays=defaultdict(list);index=[]
    for number,(track,rs) in enumerate(sorted(tracks.items())):
        if number%100==0:memory_guard(cfg)
        ids=np.array([r['row_index'] for r in rs]);times=np.array([r['timestamp_us'] for r in rs]);prior=np.array([r['provenance_weight'] for r in rs]);quality=np.array([r['geometry_weight'] for r in rs])
        for end in range(max(1000000,((int(times[0])+249999)//250000)*250000),min(int(times[-1])+1000000,end_us)//250000*250000+1,250000):
            begin=end-1000000;lo=np.searchsorted(times,begin);hi=np.searchsorted(times,end,side='right')
            if hi==lo:continue
            local=ids[lo:hi];t=(times[lo:hi]-begin)/1e6;var=state['variance'][local].astype(float)*model['kappa_cov']
            fit=temporal_fit(t,state['h'][local].astype(float),var,prior[lo:hi],quality[lo:hi],options,scale)
            c=fit['coefficients'];cov=fit['covariance']*np.asarray(model['temporal_multipliers'])[:,None,None]
            ell=np.stack([basis(1.,d)*2**d for d in range(3)]);endpoint=ell@c;ecov=np.einsum('ai,dij,aj->da',ell,cov,ell);cross=np.einsum('i,dij,j->d',ell[0],cov,ell[1]);nodes=collocation(c)
            appvalid=appearance['valid'][local].astype(bool);app=appearance['embedding'][local][appvalid].astype(float).mean(0) if appvalid.any() else np.zeros(128)
            values=dict(coefficients=c,coefficient_cov=cov,endpoint=endpoint[0],endpoint_velocity=endpoint[1],endpoint_accel=endpoint[2],endpoint_cov=ecov,endpoint_cross_cov=cross,appearance=app.astype(np.float16),collocation_h=nodes['h'],collocation_velocity=nodes['velocity'],collocation_accel=nodes['acceleration'],xi=nodes['xi'],xi_dot=nodes['xi_dot'])
            for key,value in values.items():arrays[key].append(np.asarray(value,np.float16 if key=='appearance' else np.float32))
            maxgap=float(max(np.diff(np.r_[0,t,1])));span=float(t[-1]-t[0]);valid=len(t)>=4 and fit['neff']>=6 and span>=.6 and maxgap<=.35 and fit['condition']<1e10
            residual=state['h'][local]-basis(2*t-1)@c;rmse=float(np.sqrt(np.sum(fit['weights']*residual**2)/max(fit['weights'].sum(),1e-20)))
            index.append(dict(video_id=video,row_index=len(index),entity_track_id=track,window_start_us=begin,window_end_us=end,window_seconds=1.,hop_seconds=.25,canonical_mode='intrinsic_normalised_2d',effective_samples=fit['neff'],max_gap_seconds=maxgap,span_seconds=span,weighted_fit_rmse=rmse,condition_score=fit['condition'],solver='cholesky',manual_fraction=float(np.mean([r['source_kind']=='manual_sparse' for r in rs[lo:hi]])),interpolation_fraction=float(np.mean([r['source_kind']=='dense_interpolated' for r in rs[lo:hi]])),valid=bool(valid),appearance_valid=bool(appvalid.any()),camera_pose_fraction=float(np.mean([r['pose_valid'] for r in rs[lo:hi]])),quality_flags=[] if valid else ['insufficient_effective_samples_span_or_gap'],action_split=rs[0]['action_split'],visor_split=rs[0]['visor_split'],engineering_split=rs[0]['engineering_split']))
    out=directory('functional');tensor=out/f'{video}.pt';table=out/f'{video}.parquet';save_shard(tensor,{k:np.stack(v) for k,v in arrays.items()});save_index(table,index)
    metrics=dict(windows=len(index),valid=sum(r['valid'] for r in index),elapsed_seconds=time.monotonic()-start)
    manifest('functional',video,[tensor,table],[directory('spatial')/f'{video}.pt',directory('appearance')/f'{video}.pt',directory('calibration')/'model.json',folder('reports')/'selection_locked.json'],metrics)
    return metrics
