"""Conditional fixed-association NIS fit and explicitly silver-reference evaluation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from types import SimpleNamespace
from scipy.optimize import minimize
from pipeline.common import ROOT, atomic_json, digest


def replay(archives, q, floor, output=False):
    """Replay centroid measurements with fixed recorded associations; no association retuning."""
    score=0.;count=0;states={};nis_total=0.
    for episode,values in archives:
        tracks={}
        for row in values:
            t,tid=row[:2];z=row[2:4];r=row[4:8].reshape(2,2)+np.eye(2)*floor**2
            if tid not in tracks:
                mean=np.r_[z,row[10:12]];cov=np.zeros((4,4));cov[:2,:2]=r+np.eye(2)*1e-4;cov[2:,2:]=np.eye(2)*.01
            else:
                old,mean,cov=tracks[tid];dt=t-old
                if dt<=0:raise ValueError('Measurement times must increase per ID')
                eye=np.eye(2);f=np.block([[eye,dt*eye],[np.zeros((2,2)),eye]])
                process=q*np.block([[dt**3/3*eye,dt**2/2*eye],[dt**2/2*eye,dt*eye]])
                mean=f@mean;cov=f@cov@f.T+process;innovation=z-mean[:2];s=cov[:2,:2]+r
                sign,logdet=np.linalg.slogdet(s)
                if sign<=0:raise ValueError('Nonpositive innovation covariance')
                nis=float(innovation@np.linalg.solve(s,innovation));nis_total+=nis;score+=logdet+nis;count+=1
                gain=np.linalg.solve(s,cov[:2,:]).T;mean=mean+gain@innovation
                a=np.eye(4);a[:,:2]-=gain;cov=a@cov@a.T+gain@r@gain.T
            tracks[tid]=(t,mean,cov)
            if output:states[(episode,int(round(t*1e6)),int(tid))]=(mean.copy(),cov.copy())
    replay.last_nis=nis_total/max(count,1)
    return (score/max(count,1),count,states)


def replay_batched(archives,q,floor):
    """Same fixed-association objective as replay, batched over independent IDs per time."""
    score=0.;count=0;nis_total=0.
    for episode,values in archives:
        if not len(values):continue
        if np.any(np.diff(values[:,0])<0):raise ValueError('Chronological observations required')
        size=int(values[:,1].max())+1
        if size>100000:raise ValueError('Unbounded recorded IDs')
        means=np.zeros((size,4));covs=np.zeros((size,4,4));last=np.full(size,np.nan)
        boundaries=np.r_[0,np.flatnonzero(np.diff(values[:,0]))+1,len(values)]
        for begin,end in zip(boundaries[:-1],boundaries[1:]):
            batch=values[begin:end];ids=batch[:,1].astype(int)
            if len(set(ids))!=len(ids) or np.any(ids<0):raise ValueError('Unique nonnegative IDs per timestamp required')
            t=batch[0,0];z=batch[:,2:4];r=batch[:,4:8].reshape(-1,2,2)+np.eye(2)*floor**2
            seen=np.isfinite(last[ids]);old_ids=ids[seen]
            if seen.any():
                dt=t-last[old_ids]
                if np.any(dt<=0):raise ValueError('Increasing time required per ID')
                f=np.broadcast_to(np.eye(4),(len(dt),4,4)).copy();f[:,0,2]=dt;f[:,1,3]=dt
                process=np.zeros_like(f)
                for axis in range(2):
                    process[:,axis,axis]=q*dt**3/3;process[:,axis,axis+2]=process[:,axis+2,axis]=q*dt**2/2;process[:,axis+2,axis+2]=q*dt
                mean=(f@means[old_ids,:,None])[...,0];cov=f@covs[old_ids]@f.transpose(0,2,1)+process
                innovation=z[seen]-mean[:,:2];innovation_cov=cov[:,:2,:2]+r[seen]
                sign,logdet=np.linalg.slogdet(innovation_cov)
                if np.any(sign<=0):raise ValueError('Nonpositive innovation covariance')
                nis=np.einsum('bi,bi->b',innovation,np.linalg.solve(innovation_cov,innovation[...,None])[...,0]);nis_total+=float(nis.sum());score+=float((logdet+nis).sum());count+=len(dt)
                gain=np.linalg.solve(innovation_cov,cov[:,:2,:]).transpose(0,2,1);mean+=(gain@innovation[...,None])[...,0]
                a=np.broadcast_to(np.eye(4),cov.shape).copy();a[:,:,:2]-=gain
                cov=a@cov@a.transpose(0,2,1)+gain@r[seen]@gain.transpose(0,2,1)
                means[old_ids]=mean;covs[old_ids]=cov
            fresh=ids[~seen]
            if len(fresh):
                means[fresh]=np.c_[z[~seen],batch[~seen,10:12]]
                covs[fresh,:2,:2]=r[~seen]+np.eye(2)*1e-4;covs[fresh,2:,2:]=np.eye(2)*.01
            last[ids]=t
    replay_batched.last_nis=nis_total/max(count,1)
    return score/max(count,1),count,{}


def box_iou(first,second):
    first=np.asarray(first,float);second=np.asarray(second,float)
    intersection=np.maximum(0,np.minimum(first[2:],second[2:])-np.maximum(first[:2],second[:2])).prod()
    union=np.maximum(0,first[2:]-first[:2]).prod()+np.maximum(0,second[2:]-second[:2]).prod()-intersection
    return float(intersection/union) if union>0 else 0.


def bind_reference_window(states,supports,episode,timestamp,centroid,bbox,visible_ordinal):
    if visible_ordinal>5:return None
    candidates=[]
    for (ep,t,tid),(mean,cov) in states.items():
        if ep!=episode or t!=timestamp:continue
        support=supports.get((ep,t,tid))
        if support is None or support[2]<3:continue
        shape,box,hits=support;delta=centroid-mean[:2]
        distance=float(delta@np.linalg.solve(shape,delta));iou=box_iou(box,bbox)
        if distance<=6.25 or iou>=.20:candidates.append((distance,-iou,tid))
    return min(candidates)[2] if candidates else None


def bind_keyframe_reference(states,supports,episode,timestamp,centroid,bbox,visible_ordinal):
    if visible_ordinal>5:return None
    bbox=np.asarray(bbox,float);extent=bbox[2:]-bbox[:2]
    if np.any(extent<=0):return None
    candidates=[]
    for (ep,t,tid),(mean,cov) in states.items():
        if ep!=episode or t!=timestamp:continue
        support=supports.get((ep,t,tid))
        if support is None or support[2]<3:continue
        shape,box,hits=support
        if np.any(mean[:2]<bbox[:2]-.15*extent) or np.any(mean[:2]>bbox[2:]+.15*extent):continue
        ellipse_box_area=float(np.prod(4.5*np.sqrt(np.diag(shape))))
        if ellipse_box_area>3.5*extent.prod():continue
        distance=float(np.sum(((mean[:2]-centroid)/extent)**2))
        candidates.append((distance,tid))
    return min(candidates)[1] if candidates else None


def review_ready(report):
    files=list((report/'responses').glob('*.json'))
    return len(files)==300 and all(json.loads(path.read_text()).get('label') is not None for path in files)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort',default='reports/track1_fmr/stage2_cohort_v5')
    ap.add_argument('--review',default='reports/track1_fmr/stage2_vlm_review_v2_support')
    ap.add_argument('--reuse-fit',type=Path);ap.add_argument('--tag',default='stage2_silver_calibration_v3');ap.add_argument('--wait-for-review',action='store_true');args=ap.parse_args()
    cohort=ROOT/args.cohort;review=ROOT/args.review;out=ROOT/'reports/track1_fmr'/args.tag
    keyframe=(cohort/'protocol.json').is_file() and json.loads((cohort/'protocol.json').read_text()).get('kind')=='keyframe_v2'
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    if not args.wait_for_review and not review_ready(review):
        raise ValueError('Complete all 300 reviews, or use --wait-for-review')
    out.mkdir(exist_ok=False);manifest=json.loads((review/'manifest.json').read_text())
    metadata=json.loads((ROOT/'reports/track1_belief/anchor_data.json').read_text())['episodes']
    held=set(manifest['participants']);dev=[];val=[];hashes={};shapes={};ledger_hashes={}
    for meta in metadata:
        path=cohort/(meta['episode_id']+'.observations.npz');hashes[str(path.relative_to(ROOT))]=digest(path)
        is_validation=meta['video_id'].split('_')[0] in held
        with np.load(path) as z:
            values=z['values'].copy()
            if is_validation:
                required=['support_shapes','support_boxes','observed_counts']
                if any(key not in z for key in required):raise ValueError('Causal per-frame support required from v5 cohort')
                for row,shape,box,hits in zip(values,z['support_shapes'],z['support_boxes'],z['observed_counts'],strict=True):
                    key=(meta['episode_id'],int(round(row[0]*1e6)),int(row[1]))
                    shapes[key]=(shape.copy(),box.copy(),int(hits))
        (val if is_validation else dev).append((meta['episode_id'],values))
    if not args.wait_for_review and not review_ready(review):
        raise ValueError('Complete all 300 reviews, or use --wait-for-review to fit dev while review runs')
    atomic_json(out/'dev_fit_protocol.json',dict(observation_hashes=hashes,review_manifest_sha256=digest(review/'manifest.json'),
        response_hashes={str(p.relative_to(ROOT)):digest(p) for p in (review/'responses').glob('*.json')},
        dev_excludes_participants=sorted(held),reference_kind='local_vlm_silver',human_adjudicated=False,
        scope='Offline centroid filter, fixed recorded associations, no DIS replay; does not calibrate live Pipeline',
        reference_state_comparable='support-consistent prompt, not independently verified',reference_definition_warning='Model estimates of foreground-mask centroids and boxes remain imperfect; no human adjudication.',
        binding=('First five visible timestamps, >=3 observed hits, centre inside reference bbox padded 15 percent; 2.25sigma ellipse bbox area <=3.5 times reference bbox area; nearest normalized centroid, first eligible ID fixed' if keyframe else 'First five visible timestamps, confirmed observation (>=3 hits), 2.5sigma containment OR bbox IoU>=.20; first eligible ID then fixed, no later reinitialization'),
        ledger_hashes=ledger_hashes,reused_fit=None if args.reuse_fit is None else str(args.reuse_fit),
        code_sha256=digest(Path(__file__))))
    evaluations=0
    def objective(log):
        nonlocal evaluations
        score=(replay_batched if keyframe else replay)(dev,*np.exp(log))[0];evaluations+=1
        if evaluations%5==0:print(f'NIS objective evaluation {evaluations}, score={score:.5f}',flush=True)
        return score
    if args.reuse_fit is None:
        fitted=minimize(objective,np.log([.001,.005]),method='Nelder-Mead',
                        bounds=[(np.log(1e-4),np.log(5.0)),(np.log(1e-5),np.log(.1))],options={'maxiter':80,'xatol':.05,'fatol':.01})
    else:
        prior=args.reuse_fit if args.reuse_fit.is_absolute() else ROOT/args.reuse_fit
        prior_protocol=json.loads((prior.parent/'dev_fit_protocol.json').read_text())
        if (not keyframe and prior_protocol['observation_hashes']!=hashes) or prior_protocol['dev_excludes_participants']!=sorted(held):
            raise ValueError('Reused fit has different inputs or participant split')
        fit=json.loads(prior.read_text());parameters=np.asarray(fit['parameters'],float)
        if parameters.shape!=(2,) or not np.isfinite(parameters).all() or np.any(parameters<=0):raise ValueError('Invalid reused fit')
        if keyframe:
            fitted=minimize(objective,np.log(parameters),method='Nelder-Mead',bounds=[(np.log(1e-4),np.log(5.0)),(np.log(1e-5),np.log(.1))],options={'maxiter':80,'xatol':.05,'fatol':.01})
        else:
            fitted=SimpleNamespace(x=np.log(parameters),fun=fit['objective'],success=fit['success'],message='Reused identical development-only fit')
        atomic_json(out/'reused_fit_provenance.json',dict(path=str(prior),sha256=digest(prior),
            prior_protocol_sha256=digest(prior.parent/'dev_fit_protocol.json'),initializer_only_refit_new_dev=keyframe))
    atomic_json(out/'dev_fit.json',dict(parameters=np.exp(fitted.x).tolist(),objective=float(fitted.fun),
        success=bool(fitted.success),evaluations=evaluations,live_pipeline_calibrated=False))
    while not review_ready(review):
        print('Dev NIS fit complete; waiting for all 300 locked model reviews',flush=True);time.sleep(30)
    labels=[]
    for path in sorted((review/'responses').glob('*.json')):
        row=json.loads(path.read_text());label=row['label']
        if label is not None and label['visible'] and label['confidence']>=.8:labels.append(row)
    atomic_json(out/'evaluation_protocol.json',dict(dev_fit_sha256=digest(out/'dev_fit.json'),
        review_manifest_sha256=digest(review/'manifest.json'),
        response_hashes={str(p.relative_to(ROOT)):digest(p) for p in (review/'responses').glob('*.json')},
        reference_kind='local_vlm_silver',human_adjudicated=False,reference_state_comparable="intended_foreground_support_alignment_only"))
    q,floor=map(float,np.exp(fitted.x));dev_replay=replay_batched if keyframe else replay;dev_replay(dev,q,floor);dev_nis=dev_replay.last_nis
    _,_,states=replay(val,q,floor,True);bound={};scores=[];rows=[];eligible=0;attempts={};binding_times={}
    visible_ordinals={}
    visible=[json.loads(path.read_text()) for path in sorted((review/'responses').glob('*.json'))]
    for item in sorted(visible,key=lambda x:(x['task']['episode_id'],x['task']['timestamp_us'])):
        label=item['label']
        if label is None or not label['visible']:continue
        task=item['task'];episode=task['episode_id'];ts=task['timestamp_us']
        visible_ordinals[episode]=visible_ordinals.get(episode,0)+1
        if label['confidence']<.8:continue
        z=np.array([label['x'],label['y']]);eligible+=1
        if episode not in bound and visible_ordinals[episode]<=5:
            attempts[episode]=attempts.get(episode,0)+1
            candidate=(bind_keyframe_reference if keyframe else bind_reference_window)(states,shapes,episode,ts,z,label['bbox'],visible_ordinals[episode])
            if candidate is not None:bound[episode]=candidate;binding_times[episode]=ts
        state=states.get((episode,ts,bound.get(episode)))
        if state is None:continue
        mean,cov=state;delta=z-mean[:2];nees=float(delta@np.linalg.solve(cov[:2,:2],delta));scores.append(nees)
        rows.append(dict(episode_id=episode,timestamp_us=ts,track_id=bound[episode],nees=nees,
                         error_px=float(np.linalg.norm(delta*[1920,1080]))))
    episode_ids=sorted(set(task['episode_id'] for task in manifest['tasks']))
    for episode in episode_ids:bound.setdefault(episode,None)
    checks=dict(binding_episode_rate=sum(tid is not None for tid in bound.values())/len(episode_ids)>=.85,
        evaluable_frames=len(scores)>=(255 if keyframe else 250),dev_fit_interior=bool(fitted.success) and 1e-4*(1+1e-6)<q<5.0*(1-1e-6),
        silver_nees=bool(scores) and 1.6<=float(np.mean(scores))<=2.6,
        silver_joint_95=bool(scores) and float(np.mean(np.array(scores)<=5.991464547))>=.88)
    atomic_json(out/'result.json',dict(process_noise=float(q),measurement_floor=float(floor),optimizer_success=bool(fitted.success),
        optimizer_message=str(fitted.message),dev_mean_nis=dev_nis,q_acc_boundary_hit=not (1e-4*(1+1e-6)<q<5.0*(1-1e-6)),q_acc_bounds=[1e-4,5.0],binding_attempts=attempts,binding_times=binding_times,qualification_checks=checks,silver_diagnostic_passed=all(checks.values()),bound_ids=bound,visible_high_confidence_labels=eligible,evaluable_fixed_id_frames=len(scores),
        silver_mean_2d_nees=float(np.mean(scores)) if scores else None,
        silver_joint_95_coverage=float(np.mean(np.array(scores)<=5.991464547)) if scores else None,
        independent_human_nees=None,independent_identity_idf1=None,human_adjudicated=False,
        stage2_promoted=False,live_pipeline_calibrated=False,reference_state_comparable="intended_foreground_support_alignment_only",per_frame=rows,
        limitation='Model centroid errors and incomplete target identity truth prevent human-reference qualification or scene IDF1'))


if __name__=='__main__':main()
