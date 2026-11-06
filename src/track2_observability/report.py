"""Versioned verification and final evidence generation CLI stage."""
import collections,json,html
import numpy as np
import pyarrow.parquet as pq
import torch
from pipeline.common import ROOT,atomic_json,digest
from track2_feasibility.data import check_frozen
from track2_feasibility.train import weight_hash
from .common import C,D,R,W,validate,hashes
from .train import fit


def verify():
    check_frozen();prior=json.loads((ROOT/'reports/track2_measurement/manifest.json').read_text())['files'];validate(prior)
    records=[json.loads(p.read_text()) for p in W.glob('*.json') if not p.name.endswith('.receipt.json')]
    folds=[r for r in records if r.get('held')];assert len(folds)==len(C['families'])*3*5
    assert all(r['held'] not in r['normalization']['participants'] for r in folds)
    assert all(r['history'][0]['epoch']==0 for r in folds)
    checkpoints=list(D.glob('*.pt'));assert len(checkpoints)==len(C['families'])*3
    for path in checkpoints:
        cp=torch.load(path,map_location='cpu',weights_only=True);record=cp['record']
        if record['selected_epochs']==0:assert record['initial_hash']==record['final_hash']
    original=torch.load(D/'anchored_stable.17.pt',weights_only=True,map_location='cpu');_,rerun=fit('anchored_stable',17,epochs=original['record']['selected_epochs']);assert rerun['final_hash']==original['record']['final_hash']
    relations=pq.read_table(R/'relation_ledger.parquet').to_pylist();assert all(r['latest_hand_source_us']<=r['cutoff_us'] for r in relations if r['available'])
    resources=[json.loads(line) for line in (W/'resources.jsonl').read_text().splitlines()]
    atomic_json(R/'verification.json',dict(fold_fits=len(folds),final_fits=len(checkpoints),held_participant_excluded_from_normalization=True,epoch_zero_checkpoint_present=True,
        deterministic_stable_refit=True,preserved_v5_manifest_files=len(prior),frozen_track1_unchanged=True,hand_inputs_past_query_cutoff=True,
        sampled_peak_tree_rss_gib=max(r['tree_rss_gib'] for r in resources),minimum_sampled_available_gib=min(r['available_gib'] for r in resources),promotion=False))


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=json.loads((R/'results.json').read_text());verification=json.loads((R/'verification.json').read_text());audit=json.loads((R/'candidate_audit.json').read_text());memory=json.loads((R/'overlap_memory.json').read_text());selection=json.loads((R/'epoch_selection.json').read_text())
    lookup={(r['family'],r['horizon_us']):r['median_seed_mean_participant_rmse'] for r in results['summary']};horizons=[250000,500000,1000000];families=['nearest','shrinkage']+C['families']
    table='| Method | 250 ms | 500 ms | 1 s |\n|---|---:|---:|---:|\n'+'\n'.join('| '+f+' | '+' | '.join(f'{lookup[f,h]:.5f}' for h in horizons)+' |' for f in families)
    rel=pq.read_table(R/'relation_predictions.parquet').to_pylist();rel_summary=[]
    for arm in C['relation_arms']:
        for h in C['horizons_seconds']:
            groups=[]
            for participant in C['validation_participants']:
                values=[r['position_squared_error'] for r in rel if r['family']==arm and r['horizon_seconds']==h and r['participant']==participant and r['target_supported']]
                groups.append(float(np.sqrt(np.mean(values))))
            rel_summary.append(dict(arm=arm,horizon_seconds=h,mean_participant_rmse=float(np.mean(groups))))
    atomic_json(R/'relation_metrics.json',dict(records=rel_summary,all_entity_cohort=True,independent_interaction_claim=False))
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for family in ['nearest','anchored_linear','anchored_stable','anchored_polynomial']:
        axes[0].plot(C['horizons_seconds'],[lookup[family,h] for h in horizons],marker='o',label=family)
    axes[0].set(title='Source-position forecast',xlabel='Horizon (s)',ylabel='Mean-participant RMSE');axes[0].legend(fontsize=8)
    for method in ['copy','extension','overlap_persistence','overlap_incoming_ridge']:
        vals=[np.mean([r['endpoint_position_rmse'] for r in memory['records'] if r['method']==method and r['horizon_seconds']==h]) for h in C['horizons_seconds']]
        axes[1].plot(C['horizons_seconds'],vals,marker='o',label=method)
    axes[1].set(title='Future functional endpoint (distinct target)',xlabel='Horizon (s)',ylabel='Functional-position RMSE',yscale='log');axes[1].legend(fontsize=8)
    for arm in C['relation_arms']:axes[2].plot(C['horizons_seconds'],[r['mean_participant_rmse'] for r in rel_summary if r['arm']==arm],marker='o',label=arm)
    axes[2].set(title='Input-information linear probes',xlabel='Horizon (s)',ylabel='Mean-participant RMSE');axes[2].legend(fontsize=8)
    fig.savefig(R/'quality.png',dpi=160);plt.close(fig)
    text=f'''# Exactly anchored dynamics and observability

Bounded development only. Original v5 artifacts are preserved. No promotion, scientific freeze, full-scale Track 2 launch or Track 3 work.

## Completed stages

A: Epoch zero is evaluated before training, is eligible for stopping selection, and an explicit zero-epoch refit performs no updates. The four original anchor/loss controls are rerun in this new release. All complete finite observation candidates inside the 10 ms tolerance are audited separately from the nearest candidate; original target labels are unchanged. Every CLI stage checks code, imported local numerical helpers, input artifacts, configuration, runtime versions and output hashes before reuse. Verification and report generation are versioned stages.

B: A source-displacement predictability audit reports direction, magnitude, participant/episode variation and training-derived low/middle/high displacement strata. Physical quiet, steady-motion, onset, reversal, occlusion and interaction strata remain unknown. No blinded reviews were manufactured; the pending v5 queue remains pending. Source strata cannot substitute for independent event labels.

C: Exactly anchored continuous models evolve position, velocity and a small latent acceleration state through an augmented matrix exponential without inverting its generator. Linear and dissipative residual generators are compared with a small polynomial-acceleration decoder, nearest, shrinkage and the original four controls. Known source-anchor position bypasses the latent decoder. Zero-horizon anchoring and position-velocity derivative consistency are tested before training. These properties do not certify physical correctness of the source anchor. Models are small but parameter counts differ; no exact capacity-matching claim is made.

D: A separate uniform-Legendre overlap operator forgets departing history, retains only overlap, and fills the incoming interval with persistence. It is compared with copy, full polynomial extension and a train-only ridge correction to the incoming value. The four-coefficient interface is preserved. This is a diagnostic approximation for the actual irregular weighted store, not its exact update. Future endpoint errors are reported separately from cached kinematic targets.

E: Same-width linear probes compare object history, measurement age/count/quality, hand missingness, nearest-hand relative position/velocity and a deterministic past-time shuffled hand control. No graph network is introduced. All probes include a zero-correction selection candidate. The cohort includes hands as well as objects, so this is an all-entity source-information test. Relations use provisional annotation identities, explicit availability and only timestamps preceding each query. Retrospective annotation availability remains unverified. Shuffling stays within each query's four past samples; it does not borrow future observations.

## Matched source-position results

{table}

Median seed of mean-participant RMSE, intrinsic-normalized coordinates. Same {results['validation_histories']} validation histories; supported targets {results['supported_targets']}; common all-horizon histories {results['common_histories']}. All selection uses five training participants. The two validation participants are repeatedly exposed and remain exploratory. Check individual seed and participant scores in results.json and per_participant.parquet; overlapping windows are not independent replicates.

Selected final epochs: {selection['tasks']}.

## Findings from this run

Epoch zero wins for two of three seeds in every exactly anchored family. Their median forecasts therefore equal nearest observation; this is a valid no-update outcome, not evidence of learned dynamics. The nonzero seed remains separately visible in results.json. Relational probes select zero correction in every arm, including shuffled and missingness controls, so this experiment does not demonstrate predictive relational information.

Overlap persistence slightly improves short-horizon future functional endpoints compared with copying, and avoids the large polynomial-extension failure. The learned incoming correction still regresses at one second. Neither gain establishes physical motion or source-position forecasting improvement. The expanded any-usable target search produces the same counts on this dataset; its regression fixture now covers incomplete-nearest alternatives.

## Target candidates

{json.dumps(audit['counts'],indent=2)}

Nearest usability and any-usable-within-tolerance are distinct. Alternative positions and complete-candidate disagreement remain diagnostic fields. The original source-state target policy and v5 findings are not overwritten.

## Verification and interpretation

{json.dumps(verification,indent=2)}

The completed test log is work/track2_observability/tests_final.log. Tests and integrity receipts provide engineering evidence. They cannot establish independent event preservation, online annotation availability or fresh-participant generalization. Training uses source-derived targets. No full joint uncertainty or calibrated physical velocity is claimed. CPU inference measurements exclude upstream RGB/Track 1 and are not deployment latency.

The retained-history operator is bounded in the ideal uniform coefficient L2 metric; the full polynomial-extension operator can amplify coefficient error. Neither mathematical statement guarantees better video forecasting. See overlap_memory.json for synthetic checks, operator norms, training selection and source endpoint scores.

Weak-form or latent-consistency training has not been added in this cycle. The current experiment first isolates initialization, memory semantics and input information as requested. Independent reviewers are still needed to complete the blinded physical-event review and eventual fresh evaluation.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=work/track2_observability/matplotlib \\
 /opt/anaconda3/bin/python3.13 -m track2_observability.run --stage all
```

Stages: prepare, train, evaluate, memory, relations, verify, report. Resume refuses changed dependencies or changed outputs. An interrupted stage validates completed job receipts before reusing them. No source observations or RGB files are duplicated; scratch and resource logs remain under work/track2_observability.
'''
    (R/'ASSESSMENT.md').write_text(text)
    content='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Anchored dynamics and observability</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;background:#111820;color:#e3eaf0;line-height:1.6}a{color:#9bd6ff}img{max-width:100%}td,th{padding:9px;border-bottom:1px solid #435160}table{border-collapse:collapse}.status{padding:18px;background:#293b48}</style><h1>Exactly anchored dynamics and observability</h1><p class="status">Exploratory source-state research. No promotion or scientific acceptance. Frozen v5 remains preserved.</p><img src="quality.png" alt="Source forecasts, functional endpoint prediction and relational information comparisons"><table><tr><th>Method</th><th>250 ms</th><th>500 ms</th><th>1 second</th></tr>'
    for f in families:content+='<tr><td>'+html.escape(f)+'</td>'+''.join(f'<td>{lookup[f,h]:.5f}</td>' for h in horizons)+'</tr>'
    content+='</table><p>221 matched histories; 201/183/146 supported targets. Functional endpoint and source-position metrics use different targets. Physical-event labels and independent references remain unavailable.</p><p>Two seeds select epoch zero in each anchored family: median forecasts equal nearest. All relational probes select zero correction. Overlap retention improves shorter-horizon functional endpoints modestly, while learned incoming correction still regresses at one second.</p><h2>Evidence</h2>'
    for name in ['ASSESSMENT.md','results.json','candidate_audit.json','epoch_selection.json','overlap_memory.json','relation_metrics.json','relation_selection.json','verification.json','episode_displacement.parquet','observation_candidates.parquet']:
        content+=f'<p><a href="{name}">{name}</a></p>'
    (R/'dashboard.html').write_text(content+'</html>')
    atomic_json(R/'status.json',dict(engineering_cycle_completed=True,promotion=False,scientific_acceptance=False,full_track2_go=False,track3_started=False,independent_reference_count=0,blinded_review_pending=True))
