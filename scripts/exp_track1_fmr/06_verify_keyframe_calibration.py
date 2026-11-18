"""Audit silver calibration inputs, scalar replay and fixed-ID binding without retuning."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from pipeline.common import ROOT,atomic_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',default='stage2_silver_v4_keyframe');args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    out=ROOT/'reports/track1_fmr'/args.tag
    protocol=json.loads((out/'dev_fit_protocol.json').read_text());evaluation=json.loads((out/'evaluation_protocol.json').read_text());result=json.loads((out/'result.json').read_text());fit=json.loads((out/'dev_fit.json').read_text())
    assert digest(out/'dev_fit.json')==evaluation['dev_fit_sha256']
    code=ROOT/'scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py';assert digest(code)==protocol['code_sha256']
    snapshot=out/'source'/code.relative_to(ROOT);snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():assert digest(snapshot)==protocol['code_sha256']
    else:snapshot.write_bytes(code.read_bytes())
    for mapping in (protocol['observation_hashes'],evaluation['response_hashes']):
        for rel,sha in mapping.items():assert digest(ROOT/rel)==sha,rel
    assert evaluation['human_adjudicated'] is False and result['human_adjudicated'] is False and result['stage2_promoted'] is False
    reused=json.loads((out/'reused_fit_provenance.json').read_text());assert reused['initializer_only_refit_new_dev'] is True
    assert digest(reused['path'])==reused['sha256'];assert fit['evaluations']>0
    if protocol.get('cohort_protocol_sha256'):
        cohort=ROOT/next(iter(protocol['observation_hashes']));assert digest(cohort.parent/'protocol.json')==protocol['cohort_protocol_sha256']
    metadata=json.loads((ROOT/'reports/track1_belief/anchor_data.json').read_text())['episodes'];meta={r['episode_id']:r for r in metadata};held=set(protocol['dev_excludes_participants']);dev=[];val=[];supports={}
    for rel in protocol['observation_hashes']:
        ep=Path(rel).name.removesuffix('.observations.npz');validation=meta[ep]['video_id'].split('_')[0] in held
        with np.load(ROOT/rel) as z:
            values=z['values'].copy()
            if protocol.get('propagated_lk'):values=np.c_[values,np.where(z['evidence']=='detector',1,2),z['motion_delta'],z['motion_dt']]
            if validation:
                for row,shape,box,hits in zip(values,z['support_shapes'],z['support_boxes'],z['observed_counts'],strict=True):supports[(ep,round(row[0]*1e6),int(row[1]))]=(shape,box,int(hits))
        (val if validation else dev).append((ep,values))
    spec=importlib.util.spec_from_file_location('calibration_replay',snapshot);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    q,floor=fit['parameters']
    if protocol.get('propagated_lk'):
        from track1_fmr.calibration_replay import replay_v3_scalar,replay_v4_scalar
        scalar=replay_v4_scalar if protocol.get('carrier_revision')=='keyframe_v4' else replay_v3_scalar
        assert digest(ROOT/'src/track1_fmr/calibration_replay.py')==protocol['replay_source_sha256']
        replay_snapshot=out/'source/src/track1_fmr/calibration_replay.py';replay_snapshot.parent.mkdir(parents=True,exist_ok=True);replay_snapshot.write_bytes((ROOT/'src/track1_fmr/calibration_replay.py').read_bytes())
    else:scalar=module.replay
    score,n,_=scalar(dev,q,floor);scalar_nis=scalar.last_nis
    assert np.isclose(score,fit['objective'],rtol=1e-10) and np.isclose(scalar_nis,result['dev_mean_nis'],rtol=1e-10)
    _,_,states=scalar(val,q,floor,True)
    labels=[json.loads((ROOT/rel).read_text()) for rel in evaluation['response_hashes']];labels.sort(key=lambda r:(r['task']['episode_id'],r['task']['timestamp_us']))
    bound={};ordinals={};scores=[];rows=[];subset=[];eligible=0
    for item in labels:
        label=item['label'];task=item['task'];ep=task['episode_id'];ts=task['timestamp_us']
        if label is None or not label['visible']:continue
        ordinals[ep]=ordinals.get(ep,0)+1
        if label['confidence']<.8:continue
        eligible+=1
        centre=np.array([label['x'],label['y']]);bbox=np.array(label['bbox']);extent=bbox[2:]-bbox[:2]
        if ep not in bound and ordinals[ep]<=5:
            candidates=[]
            for (episode,t,tid),(mean,cov) in states.items():
                if episode!=ep or t!=ts:continue
                shape,box,hits=supports[(episode,t,tid)]
                area=float(np.prod(4.5*np.sqrt(np.diag(shape))))
                if hits<3 or area>3.5*extent.prod():continue
                if np.any(mean[:2]<bbox[:2]-.15*extent) or np.any(mean[:2]>bbox[2:]+.15*extent):continue
                candidates.append((float(np.sum(((mean[:2]-centre)/extent)**2)),tid))
            if candidates:bound[ep]=min(candidates)[1]
        state=states.get((ep,ts,bound.get(ep)))
        if state is None:continue
        mean,cov=state;delta=centre-mean[:2];nees=float(delta@np.linalg.solve(cov[:2,:2],delta));scores.append(nees);rows.append((ep,ts,bound[ep],nees))
        shape=supports[(ep,ts,bound[ep])][0];inside=np.linalg.norm(delta*[1920,1080])<=50 or float(delta@np.linalg.solve(shape,delta))<=6.25
        subset.append(bool(inside))
    assert bound=={k:v for k,v in result['bound_ids'].items() if v is not None}
    assert len(scores)==result['evaluable_fixed_id_frames']
    if scores:
        assert np.isclose(np.mean(scores),result['silver_mean_2d_nees']) and np.isclose(np.mean(np.array(scores)<=5.991464547),result['silver_joint_95_coverage'])
    else:assert result['silver_mean_2d_nees'] is None and result['silver_joint_95_coverage'] is None
    assert len(rows)==len(result['per_frame'])
    for actual,record in zip(rows,result['per_frame'],strict=True):
        assert actual[:3]==(record['episode_id'],record['timestamp_us'],record['track_id']) and np.isclose(actual[3],record['nees'])
    if 'in_support_binding_rate' in result:
        assert sum(subset)==result['in_support_frames'] and np.isclose(sum(subset)/eligible,result['in_support_binding_rate'])
        if any(subset):assert np.isclose(np.mean(np.asarray(scores)[subset]),result['in_track_mean_2d_nees'])
        else:assert result['in_track_mean_2d_nees'] is None
        assert [r['in_support'] for r in result['per_frame']]==subset
    verification=dict(artifact_integrity_passed=True,development_excludes_participants=sorted(held),development_measurement_updates=n,
        scalar_replay_matches_batched_objective=True,independent_fixed_binding_recomputed=True,bound_ids=bound,evaluable_frames=len(scores),silver_mean_2d_nees=float(np.mean(scores)) if scores else None,
        silver_joint_95_coverage=float(np.mean(np.array(scores)<=5.991464547)) if scores else None,human_adjudicated=False,scientific_acceptance=False,qualification_checks=result['qualification_checks'],
        result_sha256=digest(out/'result.json'),verifier_source_sha256=digest(__file__))
    atomic_json(out/'verification.json',verification);print(json.dumps(verification))


if __name__=='__main__':main()
