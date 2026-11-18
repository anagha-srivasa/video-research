"""Recompute object-time denominators and point evidence from immutable experiment arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
from pipeline.common import ROOT,atomic_json,digest
from track1_neural_observation.benchmark import ArrayUnpickler


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',default='observation_identity_v1');args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    out=ROOT/'reports/track1_fmr'/args.tag
    protocol=json.loads((out/'protocol.json').read_text());result=json.loads((out/'result.json').read_text())
    if protocol.get('kind') in ('keyframe_v2','keyframe_v3','keyframe_v4'):
        from keyframe_benchmark import verify
        return verify(out)
    assert digest(out/'protocol.json')==result['protocol_sha256']
    for relative,sha in protocol['code_hashes'].items():assert digest(out/'source'/relative)==sha,relative
    for relative,sha in protocol['checkpoint_hashes'].items():assert digest(ROOT/relative)==sha,relative
    for relative,sha in protocol['reference_input_hashes'].items():assert digest(ROOT/relative)==sha,relative
    dataset=ROOT/'work/track1_observation_revision/tapvid_davis/tapvid_davis.pkl'
    assert digest(dataset)==protocol['dataset_sha256']
    with dataset.open('rb') as stream:cases=ArrayUnpickler(stream).load()
    totals={};frames_total=0;queries_total=0;visible_total=0;common_total=0;point_hashes={}
    result_rows={(row['video'],row['method']):row for row in result['rows']}
    for name in protocol['case_names']:
        case=cases[name];target=np.asarray(case['points'],float);hidden=np.asarray(case['occluded'],bool)
        keep=(~hidden).any(1);target=target[keep];hidden=hidden[keep];first=np.argmax(~hidden,1)
        count,frames=hidden.shape;frames_total+=frames;queries_total+=count
        eligible=(np.arange(frames)[None,:]>first[:,None]) & ~hidden
        all_time=np.arange(frames)[None,:]>first[:,None];visible_total+=int(eligible.sum())
        marker=json.loads((out/(name+'.json')).read_text());path=out/(name+'.points.npz')
        assert marker['frames']==frames and marker['queries']==count and digest(path)==marker['points_sha256']
        point_hashes[str(path.relative_to(ROOT))]=digest(path)
        common=eligible.copy()
        with np.load(path,allow_pickle=False) as saved:
            methods=[key.removesuffix('_position') for key in saved.files if key.endswith('_position')]
            for method in methods:
                positions=saved[method+'_position'];observed=saved[method+'_observed'];carrier=saved[method+'_carrier'];ids=saved[method+'_fixed_ids']
                assert positions.shape==(count,frames,2) and observed.shape==carrier.shape==(count,frames)
                assert ids.shape==(count,) and np.isfinite(positions).all() and not np.any(observed & ~carrier)
                assert np.all(positions[~observed]==-1e4)
                assert not np.any(carrier & (ids<0)[:,None])
                errors=np.linalg.norm((positions-target)*256,axis=-1)
                matched=eligible & observed;good=matched & (errors<=8)
                actual=dict(visible_point_times=int(eligible.sum()),total_point_times=int(all_time.sum()),
                    initialized_queries=int((ids>=0).sum()),observed_visible_point_times=int(matched.sum()),
                    observed_correct_8px_times=int(good.sum()),carrier_point_times=int((carrier & all_time).sum()))
                row=result_rows[(name,method)]
                assert all(row[key]==value for key,value in actual.items()),(name,method)
                assert np.isclose(row['conditional_observed_squared_error'],np.square(errors[matched]).sum())
                total=totals.setdefault(method,dict(visible=0,observed=0,correct=0,carrier=0,all_time=0,squared_error=0.,initialized=0,latency_sum=0.,latency_frames=0,returns=0,recovered=0))
                for key,field in [('visible','visible_point_times'),('observed','observed_visible_point_times'),('correct','observed_correct_8px_times'),('carrier','carrier_point_times'),('all_time','total_point_times'),('initialized','initialized_queries')]:total[key]+=actual[field]
                total['squared_error']+=float(np.square(errors[matched]).sum())
                total['returns']+=row['reference_occlusion_returns'];total['recovered']+=row['recovered_returns_5frame']
                if row['mean_component_ms'] is not None:total['latency_sum']+=row['mean_component_ms']*frames;total['latency_frames']+=frames
                if method!='LK_query_seeded':common &= observed
        assert marker['common_support']['point_times']==int(common.sum());common_total+=int(common.sum())
    summaries=[]
    for method,total in totals.items():
        summary=dict(method=method,visible_point_times=total['visible'],observed_visible_coverage=total['observed']/total['visible'],
            observed_point_agreement_8px=total['correct']/total['visible'],carrier_coverage=total['carrier']/total['all_time'],
            conditional_observed_rmse_256px=float(np.sqrt(total['squared_error']/total['observed'])) if total['observed'] else None,
            initialized_queries=total['initialized'],frame_weighted_component_ms=total['latency_sum']/total['latency_frames'] if total['latency_frames'] else None,
            reference_occlusion_returns=total['returns'],recovered_returns=total['recovered'],
            reacquisition_point_agreement=total['recovered']/total['returns'] if total['returns'] else None)
        summaries.append(summary)
    detector_summaries=[]
    for method in ['detector_bbox','detector_mask']:
        selected=[r for r in result['rows'] if r['method']==method]
        denominator=sum(r['visible_point_times'] for r in selected)
        assert denominator==visible_total
        detector_summaries.append(dict(method=method,visible_point_times=denominator,
            point_support_recall=sum(r['detector_supported_times'] for r in selected)/denominator))
    visor=[]
    for method in protocol['methods']:
        rows=[r for r in result['visor'] if r['method']==method]
        if not rows:continue
        post=[r for r in rows if not r['initialization_frame']]
        supported=[r for r in rows if r['conditional_centroid_error_norm'] is not None]
        visor.append(dict(method=method,annotated_masks=len(rows),later_masks=len(post),
            direct_detector_mask_recall_iou50=sum(r['detector_best_mask_iou']>=.5 for r in rows)/len(rows),
            subsequent_fixed_id_mask_support=[dict(video=r['video_id'],recall=r['observed_fixed_id_mask_recall']) for r in post],
            conditional_centroid_error_norm=float(np.mean([r['conditional_centroid_error_norm'] for r in supported])) if supported else None,
            conditional_scatter_error=float(np.mean([r['conditional_scatter_error'] for r in supported])) if supported else None,
            observed_reference_masks=len(supported)))
    verification=dict(passed=True,videos=len(protocol['case_names']),frames=frames_total,queries=queries_total,
        visible_point_times=visible_total,common_observed_point_times=common_total,
        denominators_recomputed_from_independent_reference=True,unmatched_queries_retained=True,coasting_never_credited_as_observation=True,
        source_checkpoint_reference_and_output_hashes_verified=True,point_array_hashes=point_hashes,
        result_sha256=digest(out/'result.json'),scientific_acceptance=False)
    atomic_json(out/'verification.json',verification)
    atomic_json(out/'assessment.json',dict(point_metrics=summaries,detector_support=detector_summaries,visor=visor,
        timing_scope='Frame-weighted component timing; shared detection pass charged to every ablation. Combined RSS includes all ablations and loaded dataset.',
        point_tracking_scope='Fixed region ID plus initial point offset; independent point-position agreement is not complete object-identity accuracy.',
        baseline_scope='Query-seeded LK has privileged point initialization. Frozen spline has no unprivileged outputs on this dataset.',
        object_idf1=None,verified_object_identity_coverage=None,contact_f1=None,directional_lobe_acceptance=False,scientific_acceptance=False))
    print(json.dumps({key:value for key,value in verification.items() if key!='point_array_hashes'}))


if __name__=='__main__':main()
