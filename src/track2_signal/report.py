import collections,json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json
from track1_retrain.common import save
from track2_feasibility.data import check_frozen
from .common import C,D,R,W,validate,hashes


def verify():
    check_frozen();validate(json.loads((R/'preserved_v6.json').read_text()))
    pairs=pq.read_table(R/'selected_pairs.parquet').to_pylist()
    assert all(r['hand_latest_us']<=r['cutoff_us'] and r['object_latest_us']<=r['cutoff_us'] for r in pairs)
    from .data import category
    assert all(category(r['hand_class_id']) in ('hand','glove') for r in pairs)
    controls=json.loads((R/'optimization_controls.json').read_text())
    for path in controls['raw_records']:
        record=json.loads((ROOT/path).read_text());held=record['held'];assert held not in record['record']['normalization']['participants']
        assert record['record']['history'][0]['epoch']==0
    memory=json.loads((R/'memory_anchors.json').read_text());assert max(r['correlated_covariance_identity_error'] for r in memory['synthetic_covariance_checks'])<1e-12
    optimization=json.loads((R/'probability_protocol_results.json').read_text())['optimization'];assert all(set(r['training_participants'])<=set(C['training_participants']) for r in optimization)
    resource=[json.loads(line) for line in (W/'resources.jsonl').read_text().splitlines()]
    atomic_json(R/'verification.json',dict(preserved_v6_files=len(json.loads((R/'preserved_v6.json').read_text())),frozen_track1_unchanged=True,selected_hand_classes_verified=True,past_timestamps_verified=True,
        independent_reference_count=0,synthetic_fits=controls['synthetic_control_fits'],real_gradient_fits=controls['real_data_new_fits'],
        covariance_cross_terms_verified=True,thresholds_and_transforms_train_only=True,sampled_peak_tree_rss_gib=max(r['tree_rss_gib'] for r in resource),minimum_sampled_available_gib=min(r['available_gib'] for r in resource)))


def summarize():
    rows=pq.read_table(R/'probability_predictions.parquet').to_pylist();groups=collections.defaultdict(list);episodes=collections.defaultdict(list)
    for r in rows:
        key=(r['cohort'],r['arm'],r['model'],r['horizon_seconds'],r['participant']);groups[key].append(r);episodes[key+(r['episode_id'],)].append(r)
    metrics=['brier','log_loss','scale_nll'];per=[];calibration=[];ep=[]
    for key,rs in groups.items():
        cohort,arm,model,h,p=key;record=dict(cohort=cohort,arm=arm,model=model,horizon_seconds=h,participant=p,queries=len(rs),event_rate=float(np.mean([r['source_displacement_event'] for r in rs])),radial_95_coverage=float(np.mean([r['radial_95_covered'] for r in rs])))
        record.update({metric:float(np.mean([r[metric] for r in rs])) for metric in metrics});record['squared_magnitude_mse']=float(np.mean([(r['target_squared_displacement']-r['predicted_squared_displacement'])**2 for r in rs]));per.append(record)
        for bin_id in range(C['probability_bins']):
            selected=[r for r in rs if min(int(r['probability']*C['probability_bins']),C['probability_bins']-1)==bin_id]
            if selected:calibration.append(dict(cohort=cohort,arm=arm,model=model,horizon_seconds=h,participant=p,bin=bin_id,queries=len(selected),predicted=float(np.mean([r['probability'] for r in selected])),observed=float(np.mean([r['source_displacement_event'] for r in selected]))))
    for key,rs in episodes.items():
        cohort,arm,model,h,p,e=key;record=dict(cohort=cohort,arm=arm,model=model,horizon_seconds=h,participant=p,episode_id=e,queries=len(rs));record.update({metric:float(np.mean([r[metric] for r in rs])) for metric in metrics});ep.append(record)
    save(R/'probability_per_participant.parquet',per);save(R/'probability_per_episode.parquet',ep);save(R/'calibration.parquet',calibration)
    summary=[]
    for cohort,arm,model,h in sorted({(r['cohort'],r['arm'],r['model'],r['horizon_seconds']) for r in per}):
        rs=[r for r in per if (r['cohort'],r['arm'],r['model'],r['horizon_seconds'])==(cohort,arm,model,h)]
        record=dict(cohort=cohort,arm=arm,model=model,horizon_seconds=h,participants=[r['participant'] for r in rs],queries=sum(r['queries'] for r in rs));record.update({m:float(np.mean([r[m] for r in rs])) for m in metrics+['squared_magnitude_mse','radial_95_coverage']});summary.append(record)
    drift=pq.read_table(R/'drift_predictions.parquet').to_pylist();drift_summary=[]
    for cohort,variant,h in sorted({(r['cohort'],r['variant'],r['horizon_seconds']) for r in drift}):
        rs=[r for r in drift if (r['cohort'],r['variant'],r['horizon_seconds'])==(cohort,variant,h)];score=[]
        for p in sorted({r['participant'] for r in rs}):score.append(float(np.sqrt(np.mean([r['position_squared_error'] for r in rs if r['participant']==p]))))
        drift_summary.append(dict(cohort=cohort,variant=variant,horizon_seconds=h,mean_participant_rmse=float(np.mean(score))))
    atomic_json(R/'metrics.json',dict(probability=summary,drift=drift_summary,averaging='Participant means; episode evidence retained; no overlapping-window independence assumption.',physical_event_claim=False,point_forecast_gate_unchanged=True))


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=json.loads((R/'metrics.json').read_text());control=json.loads((R/'optimization_controls.json').read_text());memory=json.loads((R/'memory_anchors.json').read_text());support=json.loads((R/'feature_support.json').read_text());verification=json.loads((R/'verification.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for task in C['control_tasks']+['real']:
        r=next(r for r in control['summary'] if r['task']==task);axes[0].bar(task,r['median_ratio'])
    axes[0].axhline(1,color='black',ls='--');axes[0].tick_params(axis='x',rotation=30);axes[0].set(title='Actual-protocol controls',ylabel='Selected loss / persistence')
    for name in ['functional_endpoint','latest_observation','causal_position_estimate']:
        axes[1].plot(C['horizons_seconds'],[np.mean([r['functional_endpoint_rmse'] for r in memory['records'] if r['anchor']==name and r['horizon_seconds']==h]) for h in C['horizons_seconds']],marker='o',label=name)
    axes[1].set(title='Incoming position anchors',xlabel='Horizon (s)',ylabel='Functional endpoint RMSE');axes[1].legend(fontsize=8)
    for arm,model in [('object','context_free'),('object','context'),('relation','context'),('time_shuffle','context'),('pair_shuffle','context')]:
        rs=[r for r in metrics['probability'] if r['cohort']=='object' and r['arm']==arm and r['model']==model];axes[2].plot([r['horizon_seconds'] for r in rs],[r['brier'] for r in rs],marker='o',label=arm+'/'+model)
    axes[2].set(title='Object source-displacement probability',xlabel='Horizon (s)',ylabel='Mean-participant Brier score');axes[2].legend(fontsize=7)
    fig.savefig(R/'quality.png',dpi=150);plt.close(fig)
    table='| Task | Folds | Median loss / persistence | Selected epochs |\n|---|---:|---:|---|\n'+'\n'.join(f"| {r['task']} | {r['folds']} | {r['median_ratio']:.4f} | {r['selected_epochs']} |" for r in control['summary'])
    lookup={(r['cohort'],r['arm'],r['model'],r['horizon_seconds']):r for r in metrics['probability']}
    probability_table='| Cohort | Horizon | Frequency Brier | History Brier | Corrected relation Brier | History improvement |\n|---|---:|---:|---:|---:|---:|\n'
    for cohort in ['object','hand']:
        for h in C['horizons_seconds']:
            baseline=lookup[cohort,'object','context_free',h];history=lookup[cohort,'object','context',h];relation=lookup[cohort,'relation','context',h]
            probability_table+=f"| {cohort} | {h:g} s | {baseline['brier']:.5f} | {history['brier']:.5f} | {relation['brier']:.5f} | {100*(1-history['brier']/baseline['brier']):.2f}% |\n"
    findings='Hand-history Brier improvement is positive for both exposed validation participants at all three horizons. Object-history improvement is positive on both at 250 ms; at 500 ms one participant regresses slightly and at one second aggregate improvement disappears. Corrected hand relations do not consistently improve on history/reliability, and the alternate-pair control can score better, so no specific interaction-association benefit is established. Scale-NLL gains also vary by cohort and horizon; stronger probability scores do not certify full uncertainty calibration.'
    memory_table='| Incoming anchor | 250 ms | 500 ms | 1 s |\n|---|---:|---:|---:|\n'
    for anchor in ['functional_endpoint','latest_observation','causal_position_estimate']:
        values=[float(np.mean([r['functional_endpoint_rmse'] for r in memory['records'] if r['anchor']==anchor and r['horizon_seconds']==h])) for h in C['horizons_seconds']]
        memory_table+='| '+anchor+' | '+' | '.join(f'{v:.5f}' for v in values)+' |\n'
    text=f'''# Track 2 signal diagnostics

This fresh experiment implements the v6 assessment's diagnostic priorities. Full Track 2 promotion and the existing mean-position gates remain unchanged. All evaluations use source-derived targets on exposed participants; independent physical labels remain absent.

## Measured findings

The original training pipeline learns the predictable controls without an initialization change. Selected loss is about 2–3% of persistence loss on the three positive controls. The null control remains near persistence and real-source selection mostly returns epoch zero. Generator-reset effects remain explicit: controlled fit success is not proof of latent system identification.

{probability_table}

{findings}

{memory_table}

Latest-observation incoming anchors improve the functional endpoint diagnostic, most at 250 ms. These targets differ from the cached kinematic point-forecast benchmark. The signed-drift comparison still selects persistence for objects and intercept-free ridge for hands on training-participant folds; the new probability task does not retroactively change those forecast decisions.

## Hand classification and measurement timing

Hand selection uses the EPIC noun mapping: hand/left hand/right hand classes, plus explicitly sided glove classes under a declared source-label policy. Generic gloves are excluded because wearing is ambiguous. Names containing “handle” cannot establish a hand. hand_policy.json records names/classes and actual counts. The existing selected source cohort contains left/right hand classes; a predicate defect does not imply handles occurred in it.

Object and hand slopes use actual source timestamps within a fixed 200 ms past window, at least three distinct observations, at least 40 ms span and maximum 100 ms age. Repeated timestamps are collapsed. No fixed 250 ms denominator is used for source slopes. Two-point and fitted slopes, support, age, spans and selected pairs remain in selected_pairs.parquet. Fixed interaction invariants include distance, approach/separation rate and relative speed, with explicit validity. Own-history distance/speed summaries are supplied fairly to applicable controls.

Object and hand target cohorts are evaluated separately. Time shuffling tests temporal order within support-matched past hops. Pair shuffling assigns an alternate observed hand with matching four-hop presence/velocity support and age within 50 ms. Where no alternate is available, the features remain unchanged and effective-shuffle counts are explicit; it does not certify physical association.

{json.dumps(support['cohorts'],indent=2)}

## Actual-protocol optimization controls

{table}

Twenty synthetic fold fits and five instrumented real-source fold fits use the original v6 model, Data rows, masks, normalization, optimizer, minibatches, weights and epoch-zero stopping. The only synthetic intervention is replacement of position/velocity targets by known functions of input history or context-independent noise. This is a conditional learnability check on the actual input distribution, not a complete physical simulator. Control thresholds were configured before execution. Positive-control follow-up flags: {control['positive_controls_requiring_followup']}.

Per-step/epoch logs record un-clipped generator, damping, encoder, acceleration and velocity gradient norms, plus generator change from initialization. A selected-state ablation restores the initial generator and records forecast change. Passing a predictable target does not establish that the true latent generator was recovered; readouts can approximate dynamics. No initialization change, warm-start, weak-form loss or real-data architecture search was introduced in this round.

## Intercepts, probabilities and scale

Four signed-displacement probes compare persistence, mean drift alone, ridge without an intercept, and ridge with an unpenalized intercept. The zero-correction candidate remains in train-participant selection. Fixed regularization is {C['ridge_strength']}; there is no grid search.

A separate question predicts whether source displacement exceeds a training-only {C['displacement_threshold_quantile']:.0%} quantile. Context-free frequency, object history, reliability, missingness, corrected relations and both shuffle controls are scored with Brier and Bernoulli log loss. This target is source displacement, not physical contact, onset or intent. No probability result replaces the failed point-forecast gate.

The conditional isotropic residual-scale probe uses Gaussian source-displacement NLL around persistence, compared with a horizon-specific constant scale. Its bounded log-variance correction avoids uncontrolled extrapolation. Radial coverage and probability bins are empirical source-target diagnostics, not independently calibrated physical uncertainty. Models use fixed regularization and training-only thresholds/transforms; numerical convergence is recorded in probability_protocol_results.json. Per-participant/episode scores and calibration bins are separate artifacts. No confidence interval treats overlapping queries as independent.

## Complete memory operator and anchor comparison

{json.dumps(memory['operators'],indent=2)}

The complete persistence-filled operator can amplify errors even when retained history is bounded. All three position anchors use intrinsic-normalized coordinates. Source observation age is retained. Geometry channels and old stores are unchanged. A correlated synthetic covariance identity includes coefficient-anchor cross terms; real joint cross covariance is unavailable, so independent covariance propagation is not asserted.

## Verification and remaining limitations

{json.dumps(verification,indent=2)}

The original v6 source/artifact hashes are checked before and after this run. Stage receipts cover configuration, code dependencies, input artifacts, runtime versions and outputs; partial stages retain their dependency journal. Scratch remains under work/track2_signal and no RGB/source observation payload is duplicated.

The pending blinded physical-motion/contact/identity queue still requires independent reviewers. Automated source diagnostics cannot manufacture those labels. No fresh participant evaluation or scientific finalization has occurred. See metrics.json and per-participant evidence before drawing conclusions from aggregate scores.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=work/track2_signal/matplotlib \\
 /opt/anaconda3/bin/python3.13 -m track2_signal.run --stage all
```

Versioned stages: prepare, controls, memory, probes, summarize, verify, report. Resume rejects changed inputs, code or output hashes. Resource guards retain the 20 GiB tree-RSS ceiling and 5 GiB minimum available memory.
'''
    (R/'ASSESSMENT.md').write_text(text)
    content='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Track 2 signal diagnostics</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;background:#111820;color:#e3eaf0;line-height:1.6}a{color:#9bd6ff}img{max-width:100%}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #435160}</style><h1>Track 2 signal diagnostics</h1><p>Retrospective source targets. Existing point-forecast gate unchanged. No promotion or independent physical-event claim.</p><img src="quality.png" alt="Actual-protocol optimization controls, incoming memory anchors and displacement probability scores"><p>The actual protocol learns controlled motion. Hand-history source-displacement probabilities improve Brier score on both validation participants; object results are mixed at longer horizons. Corrected relations do not establish an added association benefit. Latest-observation incoming anchors improve functional endpoint diagnostics. These are separate from the unchanged mean-position benchmark.</p><h2>Evidence</h2>'
    for name in ['ASSESSMENT.md','metrics.json','optimization_controls.json','hand_policy.json','feature_support.json','memory_anchors.json','drift_selection.json','probability_protocol_results.json','probability_per_participant.parquet','probability_per_episode.parquet','calibration.parquet','selected_pairs.parquet','verification.json']:
        content+=f'<p><a href="{name}">{name}</a></p>'
    (R/'dashboard.html').write_text(content+'</html>');atomic_json(R/'status.json',dict(diagnostic_cycle_completed=True,independent_reference_count=0,promotion=False,point_forecast_gate_changed=False,physical_review_pending=True))
