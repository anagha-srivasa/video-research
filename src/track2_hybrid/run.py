"""Checkpointed bounded IMM experiment; immutable inputs and memory admission."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='1'
import argparse
import fcntl
import json
import logging
import threading
import time
import numpy as np
from pipeline.common import ROOT,atomic_json,digest
from track1_retrain.common import guard,pool
from track2_feasibility.data import check_frozen
from . import experiment as exp
from . import geometry,synthetic

class Monitor:
    def __enter__(self):
        self.stop=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def loop(self):
        while not self.stop.is_set():
            try:
                rss,available=guard()
                with (exp.REPORT/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),tree_rss_gib=rss,available_gib=available))+'\n')
            except BaseException as exc:self.error=exc;return
            self.stop.wait(2)
    def check(self):
        if self.error:raise self.error
        guard()
    def __exit__(self,*args):self.stop.set();self.thread.join(3)


def identity():
    paths=list((ROOT/'src/track2_hybrid').glob('*.py'))+[ROOT/'configs/track2_hybrid.json',ROOT/'tests/test_track2_hybrid.py']
    # Include all helper code and previous experiment evidence; no broad dataset copying.
    paths+=list((ROOT/'src').rglob('*.py'))
    paths+=[p for folder in ('reports/track2_feasibility','reports/track2_kinematics','derived/track2_feasibility','derived/track2_kinematics')
            for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix not in ('.pyc',)]
    paths+=[p for folder in ('derived/track1_context_v3/spatial','derived/track1_context_v3/functional') for p in (ROOT/folder).glob('*') if p.is_file()]
    paths+=[ROOT/'dataset/context_v3/episodes.parquet',ROOT/'reports/track1_encoder_revision/event_parameters.json']
    return dict(files={str(p.relative_to(ROOT)):digest(p) for p in set(paths)},
        report_sha256=digest(ROOT.parent.parent/'.codex/attachments/dcab991c-d605-4bfe-bc81-2a4558707fbb/Pasted text.txt'),
        config=exp.CFG,prior_validation_exposed=True,scientific_acceptance=False)


def benchmark():
    from .imm import update,forecast
    histories=json.loads((exp.DATA/'histories.json').read_text());h=next(iter(histories.values()))
    parameters=json.loads((exp.REPORT/'selection.json').read_text())['parameters'];s=exp.filtered(h,parameters)
    timings={k:[] for k in ('one_measurement_update','three_horizon_forecast','full_one_second_refit')}
    for _ in range(60):
        for label,fn in [('one_measurement_update',lambda:update(s,.02,h['positions'][-1],h['variances'][-1])),
                         ('three_horizon_forecast',lambda:[forecast(s,x) for x in (.25,.5,1.)]),
                         ('full_one_second_refit',lambda:exp.filtered(h,parameters))]:
            start=time.perf_counter();fn();timings[label].append((time.perf_counter()-start)*1000)
    atomic_json(exp.REPORT/'latency.json',dict(records={k:dict(median_ms=float(np.median(v[10:])),p95_ms=float(np.quantile(v[10:],.95))) for k,v in timings.items()},
        device='CPU',precision='float64',upstream_excluded=True,incremental_update_not_equivalent_to_reset_window_refit=True))


def report():
    metrics=json.loads((exp.REPORT/'metrics.json').read_text())['records'];decision=json.loads((exp.REPORT/'decision.json').read_text())
    geo=json.loads((exp.REPORT/'geometry.json').read_text());support=json.loads((exp.REPORT/'support.json').read_text())
    table='| Method | Horizon ms | Position RMSE | Velocity RMSE | Targets |\n|---|---:|---:|---:|---:|\n'
    for r in metrics:table+=f"| {r['method']} | {r['horizon_us']/1000:g} | {r['position']:.6f} | {r['velocity']:.6f} | {r['targets']} |\n"
    text=f'''# Bounded Track 2 hybrid prototype

Exploratory source-state evaluation. Proxy feasibility passed: **{decision['proxy_feasibility_passed']}**. Scientific acceptance and full Track 2 GO remain false. No baseline promotion.

Implemented an actual stationary/CV interacting multiple-model filter: Markov interaction, between-mode covariance, Joseph measurement updates and likelihood-based mode weights. Twelve settings were selected on five training participants only. Single CV and stationary Kalman filters share the selected IMM parameters; they are ablations, not separately optimized competitors. Method reference: [Dingler 2022](https://arxiv.org/abs/2207.04875).

This prototype reads retrospective centroid observations within one second. Earlier Koopman/GRU/MLP models read four cached kinematic states. Their input information differs, so this is not an isolated architecture ranking. Frozen six-second contexts and participant assignments are unchanged. Two validation participants have been used repeatedly; they are not a fresh test.

{table}

Units are intrinsic-normalized position and position per second. RMSE is calculated within participant, then averaged. Unsupported future targets remain null in prediction ledgers; all original opportunities remain in target_ledger.parquet.

Input support: {json.dumps(support['records'])}.

Optional certified M32 transport scored {geo['scored']} of {geo['attempts']} proposals. Missing geometry: {geo['missing_geometry']}. This freezes the historical canonical shape and positive scale and translates it with predicted position. It does not learn shape dynamics, certify future physical boundaries or supply contact ground truth. Target topology changes remain in evaluation. Boundary scores use annotation pixels.

Conditional uncertainty includes within-mode and between-mode covariance. It is not calibrated physical uncertainty; NEES/coverage are descriptive source-target diagnostics. Five calibration participant units cannot supply a finite 95% participant-level split-conformal bound under the previous construction.

Exact synthetic stress tests report event recall, delay, false alarms per quiet minute, velocity and continuation. The independently implemented innovation-evidence channel remains separate from position estimation. These are previously exposed synthetic episodes, not independent human motion/contact labels.

Prototype 1 and 3 results are preserved in ../track2_feasibility and ../track2_kinematics. The stable Koopman candidate failed its original short-horizon gate; the newer kinematic neural models did not beat constant position across horizons. This run fills Prototype 2 without repeating those searches. No scientific finalization or expanded training is authorized by negative proxy results.

Reproduce: `PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track2_hybrid.run --resume`. Source/input/output hashes are checked; scratch checkpoints are under work/track2_hybrid. No RGB data is duplicated. The guard enforces tree RSS below 20 GiB, available memory above 5 GiB and the existing disk reserve.
'''
    (exp.REPORT/'RESULTS.md').write_text(text)
    import html
    links=['metrics.json','selection.json','support.json','geometry.json','synthetic.json','uncertainty.json','latency.json','decision.json','verification.json']
    body='<h1>Track 2 hybrid prototype</h1><p>Exploratory; no promotion or scientific freeze.</p><pre style="white-space:pre-wrap">'+html.escape(text)+'</pre>'
    body+='<p>'+' · '.join(f'<a href="{p}">{p}</a>' for p in links)+'</p>'
    (exp.REPORT/'dashboard.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Track 2 hybrid</title><style>body{max-width:1100px;margin:30px auto;padding:20px;font:16px system-ui;background:#111820;color:#e3eaf0}a{color:#8ed6ff}</style>'+body)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    for p in (exp.DATA,exp.REPORT,exp.WORK):p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True)
    with (exp.WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_frozen();fingerprint=identity();path=exp.REPORT/'protocol_lock.json'
        if path.exists():
            if not args.resume or json.loads(path.read_text())!=fingerprint:raise ValueError('Existing locked experiment or input/source drift')
        else:atomic_json(path,fingerprint)
        with Monitor() as monitor:
            stages=[('prepare',exp.prepare),('select',lambda:exp.select(pool(exp.grid_job,
                list(__import__('itertools').product(range(len(exp.GRID)),exp.CFG['training_participants'])),monitor,workers=exp.CFG['workers']))),
                ('evaluate',exp.evaluate),('summarize',exp.summarize),('geometry',geometry.evaluate),
                ('synthetic',lambda:synthetic.summarize(pool(synthetic.episode,list(range(60,90)),monitor,workers=exp.CFG['workers']))),
                ('benchmark',benchmark),('report',report)]
            for name,fn in stages:
                marker=exp.WORK/f'{name}.done.json'
                if marker.exists():
                    for p,h in json.loads(marker.read_text())['outputs'].items():
                        if digest(ROOT/p)!=h:raise ValueError('Changed stage output '+p)
                    logging.info('REUSE %s',name);continue
                monitor.check();logging.info('START %s',name);start=time.time()
                before={p:digest(p) for folder in (exp.REPORT,exp.DATA) for p in folder.glob('*') if p.is_file() and p.name!='resources.jsonl'}
                fn();monitor.check()
                outputs={str(p.relative_to(ROOT)):digest(p) for folder in (exp.REPORT,exp.DATA) for p in folder.glob('*')
                         if p.is_file() and p.name!='resources.jsonl' and (p not in before or digest(p)!=before[p])}
                atomic_json(marker,dict(elapsed_seconds=time.time()-start,outputs=outputs));logging.info('DONE %s %.1fs',name,time.time()-start)
            check_frozen()
            if identity()!=fingerprint:raise ValueError('Source or preserved artifact drift')
            atomic_json(exp.REPORT/'verification.json',dict(locked_source_and_prior_artifacts_unchanged=True,
                numerical_and_causality_tests='tests/test_track2_hybrid.py',paired_prediction_ledger_validated=True,scientific_acceptance=False))
            atomic_json(exp.DATA/'manifest.json',dict(protocol_sha256=digest(path),outputs={str(p.relative_to(ROOT)):digest(p)
                for folder in (exp.DATA,exp.REPORT) for p in folder.glob('*') if p.is_file() and p.name not in ('manifest.json','resources.jsonl')}))
if __name__=='__main__':main()
