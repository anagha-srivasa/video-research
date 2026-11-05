"""Synthetic event evidence and real fitted-contour covariance diagnostics."""
import collections,json,time
import numpy as np
import pyarrow.parquet as pq
from track1_lab.kinematics import truth,KINDS
from track1_reference.events import event_stream,interval_scores
from track1_remediation.motion import switching,predict
from track1_lab.causal_kinematics import estimate
from .events import InnovationEvidence
from .covariance import propagate,independent_input_covariance
from .experiments import ROOT,D,R,CFG,VIDEOS,save,rank


def event_worker(seed):
 from .event_training import episode_inputs
 t,p,variance,targets,quiet,kind,corruption,onset,duration=episode_inputs(seed);records=[]
 parameters=json.loads((R/'event_parameters.json').read_text())['parameters'] if (R/'event_parameters.json').exists() else {}
 for hop in [.125,.25]:
  cutoffs=np.arange(1.,4.001,hop);events,_=event_stream(t,p,variance,cutoffs);detector=InnovationEvidence(**parameters);last=-np.inf;bank_active=False;bank_events=[];latched=[];continuations=[];velocity_error=[]
  for cutoff in cutoffs:
   for i in np.flatnonzero((t>last)&(t<=cutoff)):detector.update(t[i],p[i],variance[i],cutoff)
   last=cutoff;latched.extend(e for e in detector.drain(cutoff) if e['kind']=='motion_onset' and e['evidence_time']>=cutoffs[0])
   try:bank=switching(t,p,variance,cutoff,noise_px=12.)
   except ValueError:continue
   vel=bank['weights']@np.array([b[1] for b,c in bank['models']]);now=np.linalg.norm(vel)>=100
   if now and not bank_active:bank_events.append(dict(decision_time=float(cutoff),evidence_time=float(cutoff),kind='motion_onset'))
   bank_active=now
   nearest=p[np.flatnonzero(t<=cutoff)[-1]];cv=estimate(t,p,variance,cutoff)
   if not cv.support_valid:continue
   actual,actual_v=truth(np.array([cutoff]),kind,onset,duration)
   for method,v in [('nearest',np.zeros(2)),('cv',cv.velocity),('switching',vel),('switching_plus_innovation',vel)]:velocity_error.append(dict(method=method,error=float(np.sum((v-actual_v[0])**2))))
   for horizon in [.25,.5]:
    target,_=truth(np.array([cutoff+horizon]),kind,onset,duration);point=predict(bank,horizon)[0]
    for method,prediction in [('nearest',nearest),('cv',cv.position+horizon*cv.velocity),('switching',point),('switching_plus_innovation',point)]:continuations.append(dict(method=method,horizon=horizon,mse=float(np.sum((prediction-target[0])**2))))
  events['innovation']=latched;events['switching']=bank_events
  for method,detections in events.items():
   scores=interval_scores(detections,targets,quiet);records.append(dict(seed=seed,split='synthetic_train' if seed<60 else 'synthetic_validation',kind=kind,corruption=corruption,hop_seconds=hop,method=method,**scores))
  for method in ['nearest','cv','switching','switching_plus_innovation']:
   vs=[r['error'] for r in velocity_error if r['method']==method]
   for horizon in [.25,.5]:
    errors=[r['mse'] for r in continuations if r['method']==method and r['horizon']==horizon]
    records.append(dict(seed=seed,split='synthetic_train' if seed<60 else 'synthetic_validation',kind=kind,corruption=corruption,hop_seconds=hop,method=method,horizon=horizon,position_rmse_px=float(np.sqrt(np.mean(errors))) if errors else None,velocity_rmse_px_s=float(np.sqrt(np.mean(vs))) if vs else None,position_estimator_modified=False))
 return records


def covariance_worker(video):
 from track1_lab.architecture import general_fit
 from track1.spatial import resample
 from track1_lab.geometry_contract import normalize,select
 import pyarrow.parquet as pq
 rows=pq.read_table(ROOT/f'derived/track1_context_v3/spatial/{video}.parquet').to_pylist();chosen=sorted([r for r in rows if r['source_kind']=='manual_sparse'],key=lambda r:rank(r['mask_uid']))[:8];wanted={r['mask_uid'] for r in chosen};source={}
 for batch in pq.ParquetFile(ROOT/f'canonical/v2/masks.{video}.parquet').iter_batches(batch_size=256,columns=['mask_uid','contours_json','annotation_width','annotation_height','centroid_x','centroid_y','area_px']):
  for r in batch.to_pylist():
   if r['mask_uid'] in wanted:source[r['mask_uid']]=r
 result=[];full=[]
 camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{video}.camera.json').read_text());intrinsics=np.asarray(camera['intrinsic_native']);distortion=np.asarray(camera['distortion'])
 from track1.camera import undistort
 for r in chosen:
  m=source[r['mask_uid']];native=np.array([r['width']/m['annotation_width'],r['height']/m['annotation_height']]);pitch=float(native.max());rings=select(normalize(m['contours_json']),3);allocation=[32] if len(rings)==1 else [24,8] if len(rings)==2 else [16,8,8];controls=[];covariances=[]
  for ring,n in zip(rings,allocation):
   f=general_fit(resample(np.array(ring['points'])*native,128),m=n);controls.append(f['controls']);covariances.append(f['covariance']*pitch**2)
  centre=np.array([m['centroid_x'],m['centroid_y']])*native;scale=np.sqrt(m['area_px']*native.prod()/np.pi);joint=independent_input_covariance(covariances,np.eye(2)*pitch**2,pitch**2);position_jacobian=np.column_stack([(undistort((centre+np.eye(2)[j]*.001)[None],intrinsics,distortion)[0]-undistort((centre-np.eye(2)[j]*.001)[None],intrinsics,distortion)[0])/.002 for j in range(2)]);propagated=propagate(controls,centre,scale,joint,position_jacobian);cov=propagated['covariance'];e=np.linalg.eigvalsh(cov);diag=np.diag(cov);correlation=cov/np.sqrt(np.maximum(diag[:,None]*diag[None,:],1e-30));off=correlation-np.diag(np.diag(correlation));result.append(dict(video_id=video,mask_uid=r['mask_uid'],allocation=allocation,row_index=len(full),minimum_eigenvalue=float(e.min()),maximum_abs_offdiagonal_correlation=float(abs(off).max()),scope='full conditional Jacobian covariance from refitted control covariances; independent centre/scale blocks assumed',coordinate_mode='intrinsic_normalized_2D_position_and_dimensionless_shape_log_scale',position_jacobian_method='central finite difference of distortion-corrected native point, step .001 px',input_cross_covariances_estimated=False,physical_uncertainty_calibrated=False,geometry_after_projection_available=False));full.append(cov)
 save(R/f'covariance.{video}.parquet',result)
 if full:
  from track1_finetune.data import array
  array(D/f'{video}.shape_covariance.npy',full)
 return result


def real_event_worker(video):
 rows=pq.read_table(ROOT/f'derived/track1_context_v3/spatial/{video}.parquet',columns=['entity_track_id','timestamp_us','centre_native','mask_uid','source_kind']).to_pylist();groups=collections.defaultdict(dict)
 for r in rows:groups[r['entity_track_id']][r['timestamp_us']]=r
 values=[]
 for track,local in sorted(groups.items()):
  parameters=json.loads((R/'event_parameters.json').read_text())['parameters'] if (R/'event_parameters.json').exists() else {}
  detector=InnovationEvidence(**parameters)
  for t,r in sorted(local.items()):
   state=detector.update(t/1e6,r['centre_native'],[1.,1.]);events=detector.drain(t/1e6);values.append(dict(video_id=video,entity_track_id=track,timestamp_us=t,mask_uid=r['mask_uid'],source_kind=r['source_kind'],**state,events_json=json.dumps(events),coordinate_mode='native image-plane source centroid; not independent physical motion',independent_reference=False))
 save(D/f'{video}.event_evidence.parquet',values);return dict(video_id=video,observations=len(values),events=sum(len(json.loads(r['events_json'])) for r in values),independent_reference=False)
