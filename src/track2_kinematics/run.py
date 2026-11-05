"""Hash-locked six-run extension. No automatic architecture expansion or promotion."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '1'
import argparse
import fcntl
import json
import logging
import platform
import sys
import threading
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from pipeline.common import ROOT, atomic_json, digest
from track1_retrain.common import guard, pool
from track2_feasibility.data import Data, check_frozen
from track2_feasibility.train import weight_hash
from track1_encoder_revision.references import audit as reference_audit
from .model import CFG, normalization, tensors, predict
from .fit import DATA, REPORT, WORK, fold_job, final_job, fit, load
from .evaluate import evaluate, summarize, report

LOG = logging.getLogger(__name__)
SUPPLIED = Path('/Users/anagha/.codex/attachments/742b1763-ad12-437c-b283-2b38b2881b49/Pasted text.txt')


class Monitor:
    def __enter__(self):
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def loop(self):
        while not self.stop.is_set():
            try:
                rss, available = guard()
                with (REPORT/'resources.jsonl').open('a') as f:
                    f.write(json.dumps(dict(unix=time.time(), tree_rss_gib=rss, available_gib=available,
                                            system_cpu_percent=psutil.cpu_percent()))+'\n')
            except BaseException as exc:
                self.error = exc
                return
            self.stop.wait(2)

    def check(self):
        if self.error:
            raise self.error
        guard()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(3)


def protocol():
    check_frozen()
    previous = ROOT/'derived/track2_feasibility/manifest.json'
    manifest = json.loads(previous.read_text())
    for name, value in manifest['source'].items():
        if digest(ROOT/name) != value:
            raise ValueError('Prior experiment source changed: '+name)
    for name, value in manifest['outputs'].items():
        if digest(previous.parent/name) != value:
            raise ValueError('Prior experiment model/adapter changed: '+name)
    files = list((ROOT/'src/track2_kinematics').glob('*.py')) + [ROOT/'configs/track2_kinematics.json',
        ROOT/'src/track2_assessment.py', ROOT/'src/pipeline/common.py', ROOT/'src/track1_retrain/common.py',
        ROOT/'src/track1_encoder_revision/references.py']
    original = list((ROOT/'reports/track2_feasibility').glob('*'))
    return dict(config=CFG, source_hashes={str(p.relative_to(ROOT)): digest(p) for p in files},
        prior_experiment_files={str(p.relative_to(ROOT)): digest(p) for p in original if p.is_file()},
        prior_artifact_manifest_sha256=digest(previous), supplied_report_sha256=digest(SUPPLIED),
        preregistered_before_this_training=True, prior_validation_exposure=True,
        new_final_runs=6, stopping_fits=30, deterministic_repeats=1,
        reported_participant_reassignment_rejected='P11 is an exposed training participant, not a new final test',
        scientific_acceptance=False, full_track2_go=False)


def prepare(monitor):
    data = Data()
    if sorted(set(data.participants[data.split=='train'])) != CFG['training_participants'] or sorted(set(data.participants[data.split=='validation'])) != CFG['validation_participants']:
        raise ValueError('Participant split differs from locked protocol')
    for sample in data.samples:
        times = sample['input_times_us']
        if len(times) != 4 or any(b-a != 250000 for a,b in zip(times,times[1:])):
            raise ValueError('Nonconsecutive input history')
    for split in ('train', 'validation'):
        if not np.all(np.sum(data.mask[data.indices(split, usable=False)], axis=0) > 0):
            raise ValueError('No future support')
    atomic_json(REPORT/'support.json', dict(cohort='existing frozen context-v3 M16 adapter',
        participants={s: sorted(set(data.participants[data.split==s])) for s in ('train','validation')},
        input_histories={s: len(data.indices(s, usable=False)) for s in ('train','validation')},
        supported_targets={s: data.mask[data.indices(s, usable=False)].sum(axis=0).tolist() for s in ('train','validation')},
        original_opportunities='reports/track2_feasibility/target_ledger.parquet',
        copied_rgb_bytes=0, source_arrays_reused=True, optional_geometry_required=False,
        overlapping_histories_not_independent_episodes=True))
    atomic_json(REPORT/'independent_references.json', reference_audit())
    atomic_json(REPORT/'environment.json', dict(python=sys.version, platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, device='cpu', precision='float32', workers=CFG['workers'],
        process_tree_rss_limit_gib=20, minimum_available_gib=5))
    monitor.check()


def stopping(monitor):
    jobs = [(f,s,p) for f in CFG['models'] for s in CFG['seeds'] for p in CFG['training_participants']]
    records = pool(fold_job, jobs, monitor, workers=CFG['workers'])
    selected = []
    for f in CFG['models']:
        for s in CFG['seeds']:
            values = [r for r in records if r['family']==f and r['seed']==s]
            epochs = int(np.median([r['best_epoch'] for r in values]))
            selected.append(dict(family=f, seed=s, epochs=epochs, fold_best_epochs={r['held_participant']: r['best_epoch'] for r in values}))
            LOG.info('LOCK EPOCHS %s/%s=%d', f, s, epochs)
    atomic_json(REPORT/'selected_epochs.json', dict(records=selected, stopping_fits=len(records), validation_used=False))


def training(monitor):
    selected = json.loads((REPORT/'selected_epochs.json').read_text())['records']
    records = pool(final_job, [(r['family'], r['seed'], r['epochs']) for r in selected], monitor, workers=min(CFG['workers'],6))
    atomic_json(REPORT/'training.json', dict(records=records, final_runs=6))


def verification(monitor):
    data = Data()
    for f in CFG['models']:
        for s in CFG['seeds']:
            model, norm, value = load(DATA/'models'/f'{f}.{s}.pt')
            if norm['training_participants'] != CFG['training_participants']:
                raise ValueError('Incorrect training normalization')
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError('Nonfinite checkpoint')
    # Fixed in advance, independent of validation performance.
    original, _, value = load(DATA/'models'/'kinematic_mlp.17.pt')
    repeated, _, record = fit('kinematic_mlp', 17, epochs=value['epochs'])
    if weight_hash(original) != weight_hash(repeated):
        raise ValueError('Deterministic refit mismatch')
    atomic_json(WORK/'repeat.json', record)
    atomic_json(REPORT/'verification.json', dict(final_models=6, training_only_normalization=True,
        all_reload_checks_bitwise_equal=True, fixed_mlp17_repeat_bitwise_equal=True,
        repeated_weight_hash=weight_hash(repeated), frozen_track1_hashes_unchanged=True,
        prior_experiment_preserved=True, validation_inference_not_repeated=True,
        invalid_inputs_remain_in_source_ledger=True, independent_physical_references=False))
    monitor.check()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for directory in (DATA, REPORT, WORK, WORK/'folds', DATA/'models'):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = protocol()
        lock_path = REPORT/'protocol_lock.json'
        if lock_path.exists():
            if not args.resume or json.loads(lock_path.read_text()) != identity:
                raise ValueError('Existing protocol differs or --resume missing')
        else:
            atomic_json(lock_path, identity)
        stages = [('prepare', prepare), ('folds', stopping), ('training', training),
                  ('evaluation', lambda m: evaluate()), ('summary', lambda m: summarize()),
                  ('verification', verification), ('report', lambda m: report())]
        with Monitor() as monitor:
            try:
                for name, function in stages:
                    monitor.check()
                    marker = WORK/(name+'.complete.json')
                    if marker.exists():
                        previous = json.loads(marker.read_text())
                        if previous['protocol_sha256'] != digest(lock_path):
                            raise ValueError('Stage belongs to a different protocol')
                        if any(not (ROOT/p).exists() or digest(ROOT/p) != h for p,h in previous['outputs'].items()):
                            raise ValueError('Completed stage output changed: '+name)
                        LOG.info('RESUME %s', name)
                        continue
                    atomic_json(REPORT/'status.json', dict(state='running', stage=name, scientific_acceptance=False))
                    LOG.info('BEGIN %s', name)
                    start = time.monotonic()
                    function(monitor)
                    monitor.check()
                    artifacts = list(REPORT.glob('*')) + list((DATA/'models').glob('*.pt')) + list((WORK/'folds').glob('*.json'))
                    outputs = {str(p.relative_to(ROOT)): digest(p) for p in artifacts
                               if p.is_file() and p.name not in ('status.json','resources.jsonl','resources.json')}
                    atomic_json(marker, dict(protocol_sha256=digest(lock_path), seconds=time.monotonic()-start, outputs=outputs))
                    LOG.info('COMPLETE %s %.1fs', name, time.monotonic()-start)
                if protocol() != identity:
                    raise ValueError('Inputs or source changed during run')
                samples = [json.loads(line) for line in (REPORT/'resources.jsonl').read_text().splitlines()]
                atomic_json(REPORT/'resources.json', dict(peak_sampled_tree_rss_gib=max(r['tree_rss_gib'] for r in samples),
                    minimum_sampled_available_gib=min(r['available_gib'] for r in samples), rss_includes_shared_pages=True))
                atomic_json(DATA/'manifest.json', dict(protocol_sha256=digest(lock_path),
                    models={str(p.relative_to(DATA)): digest(p) for p in (DATA/'models').glob('*.pt')},
                    prediction_sha256=digest(REPORT/'predictions.parquet')))
                atomic_json(REPORT/'status.json', dict(state='complete', final_runs=6, stopping_fits=30,
                    full_track2_go=False, scientific_acceptance=False, automatic_followup_training=False))
            except BaseException as exc:
                atomic_json(REPORT/'status.json', dict(state='failed', error=str(exc), full_track2_go=False))
                raise


if __name__ == '__main__':
    main()
