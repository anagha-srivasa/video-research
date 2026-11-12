"""python -m track1_belief.run --stage all --resume"""
import argparse
import fcntl
import json
import logging
import time
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','anchors','synthetic','real','geometry','verify','report'],default='all');p.add_argument('--resume',action='store_true');args=p.parse_args()
    initialize();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True)
    with (W/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_preserved()
        protocol=dict(config=C,source_hashes=source_hashes(),validation_selection=False)
        path=R/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol changed; archive prior experiment before rerun')
        atomic_json(path,protocol)
        stages=['anchors','synthetic','real','geometry','verify','report'] if args.stage=='all' else [args.stage]
        with Monitor() as monitor:
            for stage in stages:
                marker=R/f'{stage}.complete.json'
                if args.resume and marker.exists():
                    receipt=json.loads(marker.read_text())
                    for name,expected in receipt.get('outputs',{}).items():
                        if digest(ROOT/name)!=expected:raise ValueError('Stage output changed: '+name)
                    continue
                tick=time.monotonic();LOG.info('Starting %s',stage)
                try:
                    if stage=='anchors':
                        from .data import build_dataset
                        build_dataset(monitor=monitor)
                    elif stage=='synthetic':
                        from .evaluate import select_synthetic,evaluate_synthetic
                        from track1_retrain.common import pool
                        params=select_synthetic(monitor,pool);evaluate_synthetic(params,monitor,pool)
                    elif stage=='real':
                        from .experiment import real
                        real(json.loads((R/'selection.json').read_text())['parameters'],monitor)
                    elif stage=='geometry':
                        from .experiment import geometry
                        geometry(monitor)
                    elif stage=='verify':verify()
                    elif stage=='report':report()
                    monitor.check()
                    names={'anchors':['anchor_data.json','anchor_episodes.parquet'],
                           'synthetic':['selection.json','synthetic.json','synthetic_training.parquet','synthetic_predictions.parquet','synthetic_events.parquet'],
                           'real':['real.json','real_predictions.parquet','memory.json','memory_features.parquet'],
                           'geometry':['geometry.json','geometry.parquet'],
                           'verify':['verification.json'],'report':['ASSESSMENT.md','dashboard.html']}[stage]
                    outputs={str((R/name).relative_to(ROOT)):digest(R/name) for name in names}
                    atomic_json(marker,dict(seconds=time.monotonic()-tick,completed=time.time(),outputs=outputs))
                except BaseException as exc:
                    atomic_json(R/'interruption.json',dict(stage=stage,error=str(exc),time=time.time()));raise
        LOG.info('Completed; preserved %d original inputs',check_preserved())


def verify():
    count=0
    for path in D.glob('*.state.npz'):
        with np.load(path,allow_pickle=False) as a:
            for key in ('mean','covariance','memory_mean','joint_covariance'):
                if not np.isfinite(a[key]).all():raise ValueError('Nonfinite state '+str(path))
            p=a['covariance'][a['valid']]
            if len(p) and np.linalg.eigvalsh(p).min() < -1e-6:raise ValueError('NonPSD state')
            count+=len(p)
    size=output_bytes()
    if size>C['maximum_new_bytes']:raise ValueError('New artifact budget exceeded')
    atomic_json(R/'verification.json',dict(finite_psd_supported_states=count,preserved_files=check_preserved(),new_bytes=size,
        independent_physical_references=0,scientific_acceptance=False,default_release_replaced=False))


def report():
    import html
    source=json.loads((R/'real.json').read_text());synthetic=json.loads((R/'synthetic.json').read_text())
    lines=['# Track 1 dependence-aware state experiment','',
        'Open development candidate. Frozen baseline inputs are preserved. This run tests the prepared native contexts and analytic motion; independent physical references remain absent.','',
        'Native RGB uses a forward/backward checked LK tracker initialized by a manual mask. Point IDs are permanent and lost points stay unsupported. This is a measured classical baseline; no neural point-tracker training or multi-neural-teacher distillation was performed.','',
        '| Reference | Source | Method | Horizon | Proposed | Supported | Participant RMSE px |','|---|---|---|---:|---:|---:|---:|']
    for name,value in [('Exact synthetic',synthetic),('RGB consistency proxy',source)]:
        for r in value['summary']:
            if r['split']=='validation':lines.append(f"| {name} | {r['source']} | {r['method']} | {r['horizon']} | {r['proposed']} | {r['supported']} | {r['mean_participant_rmse']} |")
    lines += ['', 'Both unconditioned support and common-support tables are stored in JSON. Disjoint RGB points still share the same seed, image sequence and algorithm. They are not human references. Retrospective centroid observations are explicitly separated from past-only RGB recursion.', '',
        'Implemented: correlated observation filtering, robust measurements/coherent jumps, joint exponential memories, endpoint-constrained history fitting, full gauge/mixture covariance, bounded static all-ring geometry allocation, task-weighted projection and compact linear probes.', '',
        'Remaining scope: neural point-tracker adaptation/re-detection, multi-neural-teacher verification/distillation, local adaptive spline knots, temporal geometry continuity and independently labeled motion/contact/identity evaluation. These are not completed by the numerical or proxy checks.', '',
        'See synthetic.json, real.json, memory.json, geometry.json and verification.json for measured evidence. No candidate is promoted automatically.']
    text='\n'.join(lines)+'\n';(R/'ASSESSMENT.md').write_text(text)
    (R/'dashboard.html').write_text('<!doctype html><meta charset="utf-8"><title>Track 1 belief experiment</title><style>body{max-width:1200px;margin:32px auto;padding:20px;font:16px system-ui;line-height:1.5}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>')
    atomic_json(D/'status.json',dict(implementation='bounded_experimental_components',finalized=False,scientific_acceptance=False,independent_references=0,
        neural_tracker=False,neural_teacher_distillation=False,local_knot_refinement=False,full_proposed_research_program_completed=False))


if __name__=='__main__':main()
