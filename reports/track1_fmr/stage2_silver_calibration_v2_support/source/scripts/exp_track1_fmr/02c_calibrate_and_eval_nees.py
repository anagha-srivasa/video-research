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
    score=0.;count=0;states={}
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
                score+=logdet+innovation@np.linalg.solve(s,innovation);count+=1
                gain=np.linalg.solve(s,cov[:2,:]).T;mean=mean+gain@innovation
                a=np.eye(4);a[:,:2]-=gain;cov=a@cov@a.T+gain@r@gain.T
            tracks[tid]=(t,mean,cov)
            if output:states[(episode,int(round(t*1e6)),int(tid))]=(mean.copy(),cov.copy())
    return (score/max(count,1),count,states)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort',default='reports/track1_fmr/stage2_cohort_v2')
    ap.add_argument('--review',default='reports/track1_fmr/stage2_vlm_review_v1')
    ap.add_argument('--reuse-fit',type=Path);ap.add_argument('--tag',default='stage2_silver_calibration_v1');ap.add_argument('--wait-for-review',action='store_true');args=ap.parse_args()
    cohort=ROOT/args.cohort;review=ROOT/args.review;out=ROOT/'reports/track1_fmr'/args.tag
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    if not args.wait_for_review and len(list((review/'responses').glob('*.json')))<300:
        raise ValueError('Complete all 300 reviews, or use --wait-for-review')
    out.mkdir(exist_ok=False);manifest=json.loads((review/'manifest.json').read_text())
    metadata=json.loads((ROOT/'reports/track1_belief/anchor_data.json').read_text())['episodes']
    held=set(manifest['participants']);dev=[];val=[];hashes={};shapes={};ledger_hashes={}
    for meta in metadata:
        path=cohort/(meta['episode_id']+'.observations.npz');hashes[str(path.relative_to(ROOT))]=digest(path)
        with np.load(path) as z:values=z['values'].copy()
        is_validation=meta['video_id'].split('_')[0] in held
        (val if is_validation else dev).append((meta['episode_id'],values))
        if is_validation:
            ledger=cohort/(meta['episode_id']+'.tubes.parquet');ledger_hashes[str(ledger.relative_to(ROOT))]=digest(ledger)
            for row in pq.read_table(ledger,columns=['track_id','shape_scatter']).to_pylist():
                key=(meta['episode_id'],int(row['track_id']));shape=np.asarray(row['shape_scatter']).reshape(2,2)
                if key in shapes and not np.array_equal(shapes[key],shape):raise ValueError('Binding requires immutable birth shape, not future changing support')
                shapes[key]=shape
    if not args.wait_for_review and len(list((review/'responses').glob('*.json')))<300:
        raise ValueError('Complete all 300 reviews, or use --wait-for-review to fit dev while review runs')
    atomic_json(out/'dev_fit_protocol.json',dict(observation_hashes=hashes,review_manifest_sha256=digest(review/'manifest.json'),
        response_hashes={str(p.relative_to(ROOT)):digest(p) for p in (review/'responses').glob('*.json')},
        dev_excludes_participants=sorted(held),reference_kind='local_vlm_silver',human_adjudicated=False,
        scope='Offline centroid filter, fixed recorded associations, no DIS replay; does not calibrate live Pipeline',
        reference_state_comparable=False,reference_definition_warning='VLM anatomical hand-region centroid excludes forearm; proposal centroid can summarize a different support. Silver NEES is a mismatch diagnostic, not true calibration evidence.',
        binding='First high-confidence VLM centroid within 50px OR immutable 2sigma birth-shape ellipse, minimum shape Mahalanobis tie break, fixed ID thereafter',
        ledger_hashes=ledger_hashes,reused_fit=None if args.reuse_fit is None else str(args.reuse_fit),
        code_sha256=digest(Path(__file__))))
    evaluations=0
    def objective(log):
        nonlocal evaluations
        score=replay(dev,*np.exp(log))[0];evaluations+=1
        if evaluations%5==0:print(f'NIS objective evaluation {evaluations}, score={score:.5f}',flush=True)
        return score
    if args.reuse_fit is None:
        fitted=minimize(objective,np.log([.001,.005]),method='Nelder-Mead',
                        bounds=[(np.log(1e-6),np.log(.5)),(np.log(1e-5),np.log(.1))],options={'maxiter':35,'xatol':.05,'fatol':.01})
    else:
        prior=args.reuse_fit if args.reuse_fit.is_absolute() else ROOT/args.reuse_fit
        prior_protocol=json.loads((prior.parent/'dev_fit_protocol.json').read_text())
        if prior_protocol['observation_hashes']!=hashes or prior_protocol['dev_excludes_participants']!=sorted(held):
            raise ValueError('Reused fit has different inputs or participant split')
        fit=json.loads(prior.read_text());parameters=np.asarray(fit['parameters'],float)
        if parameters.shape!=(2,) or not np.isfinite(parameters).all() or np.any(parameters<=0):raise ValueError('Invalid reused fit')
        fitted=SimpleNamespace(x=np.log(parameters),fun=fit['objective'],success=fit['success'],message='Reused identical development-only fit')
        atomic_json(out/'reused_fit_provenance.json',dict(path=str(prior),sha256=digest(prior),
            prior_protocol_sha256=digest(prior.parent/'dev_fit_protocol.json')))
    atomic_json(out/'dev_fit.json',dict(parameters=np.exp(fitted.x).tolist(),objective=float(fitted.fun),
        success=bool(fitted.success),evaluations=evaluations,live_pipeline_calibrated=False))
    while len(list((review/'responses').glob('*.json')))<300:
        print('Dev NIS fit complete; waiting for all 300 locked model reviews',flush=True);time.sleep(30)
    labels=[]
    for path in sorted((review/'responses').glob('*.json')):
        row=json.loads(path.read_text());label=row['label']
        if label is not None and label['visible'] and label['confidence']>=.8:labels.append(row)
    atomic_json(out/'evaluation_protocol.json',dict(dev_fit_sha256=digest(out/'dev_fit.json'),
        review_manifest_sha256=digest(review/'manifest.json'),
        response_hashes={str(p.relative_to(ROOT)):digest(p) for p in (review/'responses').glob('*.json')},
        reference_kind='local_vlm_silver',human_adjudicated=False,reference_state_comparable=False))
    q,floor=np.exp(fitted.x);_,_,states=replay(val,q,floor,True);bound={};scores=[];rows=[];eligible=0
    for item in sorted(labels,key=lambda x:(x['task']['episode_id'],x['task']['timestamp_us'])):
        task=item['task'];episode=task['episode_id'];ts=task['timestamp_us'];z=np.array([item['label']['x'],item['label']['y']]);eligible+=1
        candidates=[]
        for (ep,t,tid),(mean,cov) in states.items():
            if ep!=episode or t!=ts:continue
            delta=z-mean[:2];pixel_distance=np.linalg.norm(delta*[1920,1080]);shape=shapes.get((episode,tid))
            shape_distance=float(delta@np.linalg.solve(shape,delta)) if shape is not None else float('inf')
            if pixel_distance<=50 or shape_distance<=4:candidates.append((shape_distance,tid))
        if episode not in bound:bound[episode]=min(candidates)[1] if candidates else None
        state=states.get((episode,ts,bound[episode]))
        if state is None:continue
        mean,cov=state;delta=z-mean[:2];nees=float(delta@np.linalg.solve(cov[:2,:2],delta));scores.append(nees)
        rows.append(dict(episode_id=episode,timestamp_us=ts,track_id=bound[episode],nees=nees,
                         error_px=float(np.linalg.norm(delta*[1920,1080]))))
    atomic_json(out/'result.json',dict(process_noise=float(q),measurement_floor=float(floor),optimizer_success=bool(fitted.success),
        optimizer_message=str(fitted.message),bound_ids=bound,visible_high_confidence_labels=eligible,evaluable_fixed_id_frames=len(scores),
        silver_mean_2d_nees=float(np.mean(scores)) if scores else None,
        silver_joint_95_coverage=float(np.mean(np.array(scores)<=5.991464547)) if scores else None,
        independent_human_nees=None,independent_identity_idf1=None,human_adjudicated=False,
        stage2_promoted=False,live_pipeline_calibrated=False,reference_state_comparable=False,per_frame=rows,
        limitation='Model centroid errors and incomplete target identity truth prevent human-reference qualification or scene IDF1'))


if __name__=='__main__':main()
