"""Per-video CPU stages: source-aware tracks, spatial shards and rolling functions."""
from collections import defaultdict,Counter
import json,time,logging
import numpy as np
import duckdb
from pipeline.common import ROOT,atomic_json
from .artifacts import output_dir,save_shard,load_shard,save_index,manifest,memory_guard
from .tracks import assign_tracks
from .camera import intrinsics,undistort,require_validated
from .spatial import dominant_ring,resample,align_seam,fit_spline,wavelet_state
from .temporal import fit_function,collocation
from .numerics import spline_basis,wavelet_matrix

LOG=logging.getLogger('track1')

def split_for(cfg,video):
    return next(k for k,vs in cfg['engineering_splits'].items() if video in vs)

def observations(cfg,video):
    with duckdb.connect() as db:
        db.execute('SET threads=1')
        columns='m.video_id,m.mask_uid,m.timestamp_us,m.source_kind,m.entity_name_open,m.class_id,m.interpolation_id,m.annotation_frame_uid,m.geometry_status,m.component_count,m.hole_count,m.area_px,m.centroid_x,m.centroid_y,m.bbox_xmin,m.bbox_ymin,m.bbox_xmax,m.bbox_ymax,m.contours_json,m.quality_flags,m.exhaustive_status'
        cursor=db.execute(f"SELECT {columns},a.subsequence,a.T_annotation_to_native,a.native_frame_uid,a.alignment_status,a.alignment_error_us,a.alignment_limit_us,a.transform_valid FROM read_parquet(?) m JOIN read_parquet(?) a USING(annotation_frame_uid)",[str(ROOT/cfg['input']/f'masks.{video}.parquet'),str(ROOT/cfg['input']/f'annotation_frames.{video}.parquet')])
        names=[x[0] for x in cursor.description];rows=[dict(zip(names,r)) for r in cursor.fetchall()]
    eligible=[];excluded=[]
    for row in rows:
        reason='unresolved_timestamp' if row['timestamp_us'] is None else 'quarantined_geometry' if row['geometry_status']=='quarantined' else 'annotation_transform_invalid' if not row['transform_valid'] else None
        if reason is None and row['alignment_error_us'] is not None and row['alignment_error_us']>=row['alignment_limit_us']-1000:reason='outside_strict_half_local_frame_interval'
        if reason:excluded.append(dict(mask_uid=row['mask_uid'],reason=reason));continue
        row['bbox']=np.array([row[k] for k in ['bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax']])
        t=np.asarray(row['T_annotation_to_native']).reshape(3,3);row['bbox']*=np.array([t[0,0],t[1,1],t[0,0],t[1,1]])
        eligible.append(row)
    return assign_tracks(eligible,cfg['spatial']['max_gap_seconds']),excluded

def spatial_video(cfg,choice):
    video=choice['video_id'];logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    memory_guard(cfg);require_validated(video);start=time.monotonic();rows,excluded=observations(cfg,video)
    for row in rows:
        if not row['selected_observation']:excluded.append(dict(mask_uid=row['mask_uid'],reason='duplicate_track_timestamp_prefer_manual_endpoint'))
    rows=[r for r in rows if r['selected_observation']]
    rows.sort(key=lambda r:(r['entity_track_id'],r['timestamp_us'],r['mask_uid']))
    raw_camera=json.loads((ROOT/f'dataset/camera/{video}.json').read_text())['camera']
    with duckdb.connect() as db:
        db.execute('SET threads=1')
        width,height=db.execute('SELECT native_width,native_height FROM read_parquet(?)',[str(ROOT/cfg['input']/f'videos.{video}.parquet')]).fetchone()
        pose_rows=db.execute('SELECT native_frame_uid,T_camera_to_world FROM read_parquet(?) WHERE pose_valid',[str(ROOT/cfg['input']/f'camera_frames.{video}.parquet')]).fetchall()
    poses={};duplicates=set()
    for key,value in pose_rows:
        if key in poses:duplicates.add(key)
        else:poses[key]=np.asarray(value).reshape(4,4)
    for key in duplicates:poses.pop(key,None)
    k,dist=intrinsics(raw_camera,width,height)
    arrays=defaultdict(list);index=[];previous=None;track=None;failures=Counter()
    for i,row in enumerate(rows):
        if i%2000==0:memory_guard(cfg);LOG.info('%s spatial %d/%d elapsed=%.1fs',video,i,len(rows),time.monotonic()-start)
        try:
            transform=np.asarray(row['T_annotation_to_native']).reshape(3,3)
            ring,components,holes=dominant_ring(row,transform)
            y=resample(ring,cfg['spatial']['samples'])
            if track!=row['entity_track_id']:previous=None
            y,shift=align_seam(y,previous);previous=y;track=row['entity_track_id']
            qgeom=.85 if row['geometry_status']=='repaired_for_rasterisation' else .95 if row['geometry_status']=='clipped_for_rasterisation' else 1.
            prior=cfg['provenance_priors'][row['source_kind']]
            fit=fit_spline(y,cfg['spatial'],prior*qgeom)
            p=fit['controls'];cov=fit['covariance'];area=row['area_px']*abs(np.linalg.det(transform[:2,:2]));scale=np.sqrt(area/np.pi)
            centre=(transform@np.array([row['centroid_x'],row['centroid_y'],1.]))[:2]
            normalized=undistort(p,k,dist);centre_n=undistort(centre[None],k,dist)[0]
            scale_n=scale/np.sqrt(k[0,0]*k[1,1])
            # Shape is dimensionless; geometry channels remain camera-normalised.
            z,zcov=wavelet_state(p,cov,centre,scale,cfg['spatial']['wavelet'])
            h=np.r_[centre_n,z.reshape(-1),np.log(max(scale_n,1e-12))]
            phi_mean=spline_basis().mean(0)
            centre_var=float(phi_mean@cov@phi_mean)
            curve=fit['curve'];gx=(np.roll(curve[:,1],-1)-np.roll(curve[:,1],1))/2;gy=(np.roll(curve[:,0],1)-np.roll(curve[:,0],-1))/2
            phi=spline_basis()
            ax=phi.T@gx;ay=phi.T@gy;logscale_var=float((ax@cov@ax+ay@cov@ay)/(4*max(area*area,1)))
            w=wavelet_matrix()
            shape_var=np.repeat(np.diag(zcov),2)+np.repeat((w@np.ones(len(p)))**2*centre_var/scale**2,2)+z.reshape(-1)**2*logscale_var
            variance=np.r_[centre_var/k[0,0]**2,centre_var/k[1,1]**2,shape_var,logscale_var]
            bearing=np.r_[centre_n,1.];bearing/=np.linalg.norm(bearing)
            pose=poses.get(row['native_frame_uid']);world=pose[:3,:3]@bearing if pose is not None else np.full(3,np.nan)
            valid=not fit['self_intersection'] and np.isfinite(h).all()
            flags=list(row['quality_flags'])
            if components>1:flags.append('dominant_component_only')
            if holes:flags.append('spline_outer_ring_does_not_encode_holes')
            if not valid:flags.append('spline_self_intersection_or_nonfinite')
            array_values=dict(controls=p,covariance=cov,boundary=y,normals=fit['normals'],normal_variance=fit['normal_var'],
                residual_normal=fit['residual_normal'],h=h,variance=variance,world_bearing=world,centre_native=centre,scale_native=np.array(scale))
            for key,value in array_values.items():arrays[key].append(np.asarray(value,np.float32))
            index.append(dict(video_id=video,row_index=len(index),mask_uid=row['mask_uid'],entity_track_id=row['entity_track_id'],timestamp_us=row['timestamp_us'],
                native_frame_uid=row['native_frame_uid'],annotation_frame_uid=row['annotation_frame_uid'],subsequence=row['subsequence'],interpolation_id=row['interpolation_id'],
                source_kind=row['source_kind'],entity_name=row['entity_name_open'],class_id=row['class_id'],identity_evidence=row['identity_evidence'],geometry_status=row['geometry_status'],
                component_count=components,hole_count=holes,seam_shift=shift,spatial_rmse_px=fit['rmse'],condition_score=fit['condition'],solver=fit['solver'],
                spatial_valid=bool(valid),pose_valid=pose is not None,canonical_mode='intrinsic_normalised_2d',world_bearing_valid=pose is not None,
                provenance_weight=prior,geometry_weight=qgeom,quality_flags=flags,action_split=choice['action_split'],visor_split=choice['visor_split'],engineering_split=split_for(cfg,video),
                width=width,height=height))
        except (ValueError,np.linalg.LinAlgError) as error:
            failures[str(error)]+=1;excluded.append(dict(mask_uid=row['mask_uid'],reason='spatial_failure:'+str(error)))
    out=output_dir(cfg,'spatial');tensor=out/f'{video}.pt';table=out/f'{video}.parquet';bad=out/f'{video}.excluded.parquet'
    save_shard(tensor,{k:np.stack(v) for k,v in arrays.items()});save_index(table,index);save_index(bad,excluded)
    summary=dict(observations=len(index),excluded=len(excluded),spatial_valid=sum(r['spatial_valid'] for r in index),tracks=len({r['entity_track_id'] for r in index}),failures=dict(failures),elapsed_seconds=time.monotonic()-start)
    camera_meta=out/f'{video}.camera.json'
    atomic_json(camera_meta,dict(intrinsic_native=k.tolist(),distortion=dist.tolist(),camera_source=raw_camera,native_size=[width,height],
        T_native_to_camera=[[raw_camera['width']/width,0,0],[0,raw_camera['height']/height,0],[0,0,1]],validation_report=f'reports/track1_camera.{video}.json',
        geometry_scale='reconstruction_units',metric_3d=False,h_channels='centroid intrinsic-normalised 2D (2), centred scale-normalised db2 controls (32), log apparent intrinsic scale (1)',
        covariance_assumption='Isotropic x/y spline-coordinate residual model; delta-method centroid/area and diagonal temporal feature variances. Centre/scale cross-correlations approximated. Raw controls and full 16x16 coefficient covariance retained.'))
    manifest(cfg,'spatial',video,[tensor,table,bad,camera_meta],summary);LOG.info('%s spatial COMPLETE %s',video,summary)
    return summary

def functional_video(cfg,choice):
    import pyarrow.parquet as pq
    video=choice['video_id'];start=time.monotonic();out=output_dir(cfg,'functional');spatial=output_dir(cfg,'spatial')
    state=load_shard(spatial/f'{video}.pt');rows=pq.read_table(spatial/f'{video}.parquet').to_pylist()
    appearance=load_shard(output_dir(cfg,'appearance')/f'{video}.pt')
    calibration=json.loads((output_dir(cfg,'calibration')/'model.json').read_text())
    by_track=defaultdict(list)
    for row in rows:
        if row['spatial_valid']:by_track[row['entity_track_id']].append(row)
    data=defaultdict(list);index=[];step=round(cfg['temporal']['hop_seconds']*1e6);duration=round(cfg['temporal']['seconds']*1e6)
    with duckdb.connect() as db:end_us=db.execute('SELECT duration_us FROM read_parquet(?)',[str(ROOT/cfg['input']/f'videos.{video}.parquet')]).fetchone()[0]
    for number,(track,observations_) in enumerate(sorted(by_track.items())):
        if number%200==0:memory_guard(cfg);LOG.info('%s temporal tracks %d/%d',video,number,len(by_track))
        ids=np.array([r['row_index'] for r in observations_]);times=np.array([r['timestamp_us'] for r in observations_])
        weights=np.array([r['provenance_weight'] for r in observations_]);quality=np.array([r['geometry_weight'] for r in observations_])
        variance=state['variance'][ids]*calibration['kappa_cov']
        for end in range(max(duration,((int(times[0])+step-1)//step)*step),min(int(times[-1])+duration,end_us)//step*step+1,step):
            begin=end-duration;lo=np.searchsorted(times,begin);hi=np.searchsorted(times,end,side='right')
            if hi==lo:continue
            local=ids[lo:hi];t=times[lo:hi]/1e6
            fit=fit_function(t,state['h'][local].astype(float),variance[lo:hi].astype(float),weights[lo:hi],cfg['temporal'],begin/1e6,quality[lo:hi])
            fit.coefficient_cov*=calibration['temporal_covariance_multiplier']
            fit.endpoint_cov*=calibration['temporal_covariance_multiplier']
            fit.endpoint_cross_cov*=calibration['temporal_covariance_multiplier']
            nodes=collocation(fit.coefficients,cfg['temporal']['seconds'])
            appvalid=appearance['valid'][local].astype(bool);app=appearance['embedding'][local][appvalid].astype(float).mean(0) if appvalid.any() else np.zeros(128)
            values=dict(coefficients=fit.coefficients,coefficient_cov=fit.coefficient_cov,endpoint=fit.endpoint,endpoint_velocity=fit.endpoint_velocity,
                endpoint_accel=fit.endpoint_accel,endpoint_cov=fit.endpoint_cov,endpoint_cross_cov=fit.endpoint_cross_cov,
                appearance=app.astype(np.float16),collocation_h=nodes['h'],collocation_velocity=nodes['velocity'],collocation_accel=nodes['acceleration'],xi=nodes['xi'],xi_dot=nodes['xi_dot'])
            for key,value in values.items():data[key].append(np.asarray(value,np.float16 if key=='appearance' else np.float32))
            flags=[]
            if not fit.valid:flags.append('insufficient_effective_samples_span_or_gap')
            if not appvalid.any():flags.append('no_usable_appearance_core')
            index.append(dict(video_id=video,row_index=len(index),entity_track_id=track,window_start_us=begin,window_end_us=end,window_seconds=cfg['temporal']['seconds'],hop_seconds=cfg['temporal']['hop_seconds'],
                canonical_mode='intrinsic_normalised_2d',effective_samples=fit.effective_samples,max_gap_seconds=fit.max_gap_seconds,span_seconds=fit.span_seconds,
                weighted_fit_rmse=fit.weighted_fit_rmse,condition_score=fit.condition_score,solver=fit.solver,manual_fraction=float(np.mean([r['source_kind']=='manual_sparse' for r in observations_[lo:hi]])),
                interpolation_fraction=float(np.mean([r['source_kind']=='dense_interpolated' for r in observations_[lo:hi]])),valid=fit.valid,appearance_valid=bool(appvalid.any()),
                camera_pose_fraction=float(np.mean([r['pose_valid'] for r in observations_[lo:hi]])),quality_flags=flags,
                action_split=choice['action_split'],visor_split=choice['visor_split'],engineering_split=split_for(cfg,video)))
    tensor=out/f'{video}.pt';table=out/f'{video}.parquet'
    save_shard(tensor,{k:np.stack(v) for k,v in data.items()});save_index(table,index)
    summary=dict(windows=len(index),valid=sum(r['valid'] for r in index),elapsed_seconds=time.monotonic()-start)
    manifest(cfg,'functional',video,[tensor,table],summary);LOG.info('%s functional COMPLETE %s',video,summary)
    return summary
