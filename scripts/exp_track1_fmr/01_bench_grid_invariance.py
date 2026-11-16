"""Stage 1 paired validation. Predictions are never scored as observed coverage."""
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json
from track1_belief import data
from track1_fmr.pipeline import Pipeline
from track1_fmr.dorsal_flux import PromptFreeHeartbeat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev_pool():
    """Merge consumed runs and exact manual-reference clips, never candidate holdouts."""
    path = ROOT / 'reports/track1_fmr/stage1_dev_pool_v3.json'
    if path.exists():
        return json.loads(path.read_text())
    from pipeline.common import digest
    old = json.loads((ROOT / 'reports/track1_belief/anchor_data.json').read_text())['episodes']
    source_frames = pq.read_table(ROOT / 'dataset/context_v3/frames.parquet').to_pylist()
    episodes = []
    for meta in old:
        if meta['video_id'] not in {'P02_122', 'P08_09', 'P18_03'}:
            continue
        # One consumed P02 smoke episode, all five consumed exploratory validation episodes.
        if meta['video_id'] == 'P02_122' and any(e['video_id'] == 'P02_122' for e in episodes):
            continue
        with np.load(ROOT / meta['path'], allow_pickle=False) as z:
            times = z['timestamp_us']
        times = times[times >= meta['manual_seed_us']][:30].tolist()
        frames = sorted([f for f in source_frames if f['video_id'] == meta['video_id'] and f['timestamp_us'] in times],
                        key=lambda f: f['timestamp_us'])
        episodes.append(dict(episode_id=meta['episode_id'], video_id=meta['video_id'], frames=frames, role='consumed_dev'))
    old_holdout = ROOT / 'reports/track1_fmr/stage1_holdout_v2_split.json'
    episodes.extend(dict(e, role='consumed_dev') for e in json.loads(old_holdout.read_text())['episodes'])
    # Exact manual annotations are reference-only, never supplied to proposal extraction.
    references = {}
    for video in ['P02_122', 'P06_108', 'P08_09', 'P18_03', 'P21_01']:
        mask_path = ROOT / f'canonical/v1/masks.{video}.parquet'
        frame_path = ROOT / f'canonical/v1/frames.{video}.parquet'
        annotations = pq.read_table(mask_path, columns=['timestamp_us', 'source_kind']).to_pylist()
        timestamp = min(r['timestamp_us'] for r in annotations if r['source_kind'] == 'manual_sparse')
        frames = pq.read_table(frame_path).to_pylist()
        index = next(i for i, f in enumerate(frames) if f['timestamp_us'] == timestamp)
        clip = frames[index:index + 30]
        episodes.append(dict(episode_id=f'reference_{video}_{timestamp}', video_id=video, frames=clip,
                             role='dev_manual_reference', reference_timestamp_us=timestamp))
        references[str(mask_path.relative_to(ROOT))] = digest(mask_path)
        references[str(frame_path.relative_to(ROOT))] = digest(frame_path)
    value = dict(name='stage1_dev_pool_v3', participants=['P02', 'P06', 'P08', 'P18', 'P21'],
                 episodes=episodes, input_hashes=references, prior_holdout_split_sha256=sha(old_holdout),
                 role='Consumed development only; no fresh holdout or promotion evidence')
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
    return value


def manual_recall(rows, proposal_union):
    """Exact-timestamp reference pixel coverage, not object detection recall or identity."""
    results = []
    for row in rows:
        mask = np.zeros((160, 160), np.uint8)
        width, height = row['annotation_width'], row['annotation_height']
        if not width or not height:
            continue
        for polygon in json.loads(row['segments_raw']):
            points = np.asarray(polygon, float) / [width, height] * 160
            if len(points) >= 3 and np.isfinite(points).all():
                cv2.fillPoly(mask, [np.clip(np.round(points), 0, 159).astype(np.int32)], 1)
        total = int(mask.sum())
        if total:
            results.append(dict(mask_uid=row['mask_uid'], label=row['entity_name_open'],
                                reference_pixels=total, covered_pixels=int((mask.astype(bool) & proposal_union).sum())))
    return results


def native_frames(meta, report):
    if 'source_path' not in meta:
        yield from data.native_frames(meta['video_id'], meta['frames'], meta['episode_id'] + '.invariance')
        return
    import struct
    import subprocess
    import sys
    from track1_reference.frame_checks import read_exact
    request = report / (meta['episode_id'] + '.decode.json')
    atomic_json(request, dict(source_path=meta['source_path'], frames=meta['frames']))
    process = subprocess.Popen([sys.executable, '-m', 'track1_fmr.source_stream', str(request)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    try:
        while True:
            header = process.stdout.read(8)
            if not header:
                break
            if len(header) != 8:
                raise EOFError('Truncated native frame header')
            index, size = struct.unpack('<II', header)
            if index != count or size != 1920 * 1080 * 3 or index >= len(meta['frames']):
                raise ValueError('Native PTS pipe schema/order violation')
            rgb = np.frombuffer(read_exact(process.stdout, size), np.uint8).reshape(1080, 1920, 3)
            yield index, rgb, None
            count += 1
        error = process.stderr.read().decode(errors='replace')
        if process.wait() or count != len(meta['frames']):
            raise RuntimeError('Exact-PTS decode incomplete: ' + error[-2000:])
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'dev', 'holdout'], default='train')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--v-min', type=float, default=.0035)
    parser.add_argument('--grid', type=int, choices=[128, 160], default=160)
    parser.add_argument('--sigma-flow', type=float, default=.005)
    parser.add_argument('--tag', default='stage1_v3_dev')
    parser.add_argument('--episodes', type=int, default=0, help='Dev-only first N episodes; 0 runs all')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Tag must be alphanumeric with underscores')
    cv2.setNumThreads(4)
    import torch
    torch.set_num_threads(4)
    if args.split == 'holdout':
        dev_report = ROOT / 'reports/track1_fmr/stage1_dropin_dev_v5_cached'
        checks_path = dev_report / 'preholdout_checks.json'
        if not checks_path.exists() or not json.loads(checks_path.read_text()).get('all_four_passed'):
            raise ValueError('Fresh holdout remains closed until all dev_v5 checks pass')
        current_hashes = {str(p.relative_to(ROOT)): sha(p) for p in
                          [Path(__file__).resolve(), *sorted((ROOT / 'src/track1_fmr').glob('*.py'))]}
        if json.loads((dev_report / 'protocol.json').read_text())['source_hashes'] != current_hashes:
            raise ValueError('Code changed after dev_v5 evaluation')
        split_path = ROOT / 'reports/track1_fmr/stage1_holdout_v4_split.json'
        if not split_path.exists():
            raise FileNotFoundError('A fresh v4 split must be locked after candidate freeze')
        split_manifest = json.loads(split_path.read_text())
        if split_manifest.get('code_hashes') != current_hashes:
            raise ValueError('Code differs from the fresh holdout freeze')
        selected = split_manifest['episodes']
        history = json.loads((ROOT / 'reports/track1_fmr/stage1_holdout_v2_split.json').read_text())
        used = set(history['excluded_development_participants']) | {e['video_id'].split('_')[0] for e in history['episodes']} | {'P14','P26'}
        if used & {e['video_id'].split('_')[0] for e in selected}:
            raise ValueError('Holdout participants overlap recorded prior training/evaluation')
        if len(selected) != 8 or args.episodes or (args.v_min, args.grid, args.sigma_flow) != (.0035,160,.005):
            raise ValueError('Holdout requires the full locked eight-episode default protocol')
    else:
        split_manifest = dev_pool()
        split_path = ROOT / 'reports/track1_fmr/stage1_dev_pool_v3.json'
        selected = split_manifest['episodes']
        if args.split == 'train':
            selected = selected[:1]
        elif args.episodes:
            selected = selected[:args.episodes]
    from pipeline.common import digest
    for relative, expected in split_manifest['input_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Split/reference input changed: ' + relative)
    report = ROOT / 'reports/track1_fmr' / args.tag
    report.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'work/yoloe-26s-seg-pf.pt'
    protocol = dict(split=args.split, episode_ids=[m['episode_id'] for m in selected], frames_per_episode=30,
                    device=args.device, min_area_frac=.00045, max_area_frac=.4, radius_frac=.012,
                    max_flux_proposals=10, ego_speed_scale=.12, merge_pad_norm=.02, border_margin_norm=.025, v_min=args.v_min, sigma_flow=args.sigma_flow, dorsal_grid=args.grid, comparison_grid=320, heartbeat_seconds=2., rss_cap_gib=18,
                    anti_alias="pyrDown while both axes allow 2x decimation, then linear; no RGB grid",
                    extractor_variant="stateful, top15 recent cached-inverse ROI 1.8-sigma super-Gaussian S priors, lazy <=1024 masked Lab samples, closing, .85 novelty threshold, saliency>=2.5e-5, gate0.65; ego-scaled floor, border-strip guard, greedy comoving merge; exact per-frame prediction reuse",
                    lattice_steps=32, core_density_thresh=.92, min_saliency=2.5e-5, ransac_reproj_norm=.004, prior_budget=15, max_provisional_tracks=40, single_hit_miss_limit=3, crop_quality_gate=True,
                    birth_velocity="mean image displacement / exact dt; frame variance sigma_flow^2 / dt^2",
                    flow_units="normalized displacement/frame; state velocity/sec multiplied by exact dt; camera residual applied consistently",
                    holdout_split_sha256=sha(split_path),
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                                   [Path(__file__).resolve(), *sorted((ROOT / 'src/track1_fmr').glob('*.py'))]},
                    checkpoint_sha256=sha(checkpoint),
                    source_ledger_sha256=sha(ROOT / 'reports/track1_belief/anchor_data.json'),
                    split_limitation=split_manifest.get('freshness', 'Consumed development; never promotion evidence'),
                    holdout_gate='If dev passes: pooled warm ingestion <=15ms, >=85% nonempty-flux action episodes, sampled RSS<=18GiB; scale IoU diagnostic only',
                    gate='Dev precheck: pooled median preprocessing <=1ms, pooled mean full ingestion <=12ms, three recovered focus clips with cap fraction<=.10 and median novel<=5, current unit/local-prior suite passes; RSS<=18GiB. Scale IoU reported separately for future holdout.')
    protocol['software'] = dict(opencv=cv2.__version__, torch=torch.__version__, opencv_threads=cv2.getNumThreads(), torch_threads=torch.get_num_threads())
    atomic_json(report / 'protocol.json', protocol)
    for relative in protocol['source_hashes']:
        destination = report / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    stop = threading.Event()
    resources = dict(peak_rss=0, failure=None)

    root = psutil.Process()
    monitored = [root]
    monitor_lock = threading.Lock()

    def refresh_processes():
        # Process discovery scans all system PIDs on macOS; keep it out of timed ingestion.
        children = root.children(recursive=True)
        with monitor_lock:
            monitored[:] = [root] + children

    def monitor():
        while not stop.is_set():
            try:
                with monitor_lock:
                    processes = list(monitored)
                rss = 0
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                resources['peak_rss'] = max(resources['peak_rss'], rss)
                if rss > 18 * 2**30:
                    resources['failure'] = '18 GiB process-tree cap exceeded'
            except Exception as exc:
                resources['failure'] = repr(exc)
            stop.wait(.05)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    heartbeat = PromptFreeHeartbeat(device=args.device)
    data.W = ROOT / 'work/track1_fmr'
    results = []
    try:
        # Cold loading/warmup is separately recorded, never silently included in warm throughput.
        start = time.perf_counter()
        heartbeat(np.zeros((640, 640, 3), np.uint8))
        warmup_seconds = time.perf_counter() - start
        for meta in selected:
            if resources['failure']:
                raise MemoryError(resources['failure'])
            frames = meta['frames']
            times = [f['timestamp_us'] for f in frames]
            if len(times) != 30 or sorted(set(times)) != times:
                raise ValueError('Native PTS schema mismatch')
            # Load source masks into evaluator only. No labels/polygons enter Pipeline.
            mask_path = ROOT / f"canonical/v1/masks.{meta['video_id']}.parquet"
            manual = [] if not mask_path.exists() else pq.read_table(mask_path).to_pylist()
            by_time = {}
            for row in manual:
                if row['source_kind'] == 'manual_sparse' and row['timestamp_us'] in times:
                    by_time.setdefault(row['timestamp_us'], []).append(row)
            pipelines = {'grid320': Pipeline(dorsal_grid=320, heartbeat=heartbeat, v_min=args.v_min, sigma_flow=args.sigma_flow),
                         'grid': Pipeline(dorsal_grid=args.grid, heartbeat=heartbeat, v_min=args.v_min, sigma_flow=args.sigma_flow)}
            rows, rgb_hash, pixels = [], hashlib.sha256(), 0
            intersections = unions = 0
            mask_records = []
            for n, rgb, _ in native_frames(meta, report):
                refresh_processes()
                if resources['failure']:
                    raise MemoryError(resources['failure'])
                rgb_hash.update(rgb.tobytes())
                pixels += rgb.shape[0] * rgb.shape[1]
                row = {'timestamp_us': times[n]}
                masks = {}
                for name in (list(pipelines) if n % 2 == 0 else list(pipelines)[::-1]):
                    start = time.perf_counter()
                    value = pipelines[name].step(rgb, times[n] / 1e6)
                    row[name] = dict(value, ingestion_ms=(time.perf_counter() - start) * 1000)
                    result = pipelines[name].last_flux
                    mask = np.zeros((160, 160), np.uint8) if result is None else cv2.resize(
                        result['proposal_union'].astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA)
                    masks[name] = mask >= .5
                    motion = None if result is None else result['motion_mask']
                    row[name]['residual_motion_pixels'] = None if motion is None else int(motion.sum())
                    row[name]['novel_flux_pixels_eval_grid'] = int(masks[name].sum())
                    row[name]['manual_reference'] = manual_recall(by_time.get(times[n], []), pipelines[name].last_proposal_union)
                intersection = int((masks['grid320'] & masks['grid']).sum())
                union = int((masks['grid320'] | masks['grid']).sum())
                intersections += intersection
                unions += union
                row.update(flux_intersection=intersection, flux_union=union,
                           iou_flux=None if union == 0 else intersection / union)
                mask_records.append(np.stack([masks['grid320'], masks['grid']]))
                rows.append(row)
            summary = {}
            for name in pipelines:
                latency = [r[name]['ingestion_ms'] for r in rows]
                summary[name] = dict(proposals=sum(r[name]['proposals'] for r in rows),
                                     flux_proposals=sum(r[name]['flux_proposals'] for r in rows),
                                     heartbeat_proposals=sum(r[name]['heartbeat_proposals'] for r in rows),
                                     median_preprocessing_ms=float(np.median([r[name]['preprocessing_ms'] for r in rows])),
                                     components_before_merge=sum(r[name]['eligible_flux_proposals'] for r in rows),
                                     components_after_merge=sum(r[name]['merged_flux_proposals'] for r in rows),
                                     border_components_rejected=sum(r[name]['border_rejected'] for r in rows),
                                     configured_flux_cap=rows[-1][name]['flux_proposal_limit'],
                                     working_set_evictions=sum(r[name]['evicted_working_set'] for r in rows),
                                     patch_crops=sum(r[name]['patch_crops'] for r in rows),
                                     flux_cap_frames=sum(r[name]['flux_cap_reached'] for r in rows[1:]),
                                     flux_cap_rate=sum(r[name]['flux_cap_reached'] for r in rows[1:]) / max(len(rows) - 1, 1),
                                     median_flux_proposals=float(np.median([r[name]['flux_proposals'] for r in rows[1:]])),
                                     max_rendered_priors=max(r[name]['rendered_priors'] for r in rows),
                                     min_background_eligible=min(r[name]['background_lattice_eligible'] for r in rows[1:]),
                                     mean_prior_render_ms=float(np.mean([r[name]['prior_render_ms'] for r in rows])),
                                     prior_patch_pixels=sum(r[name]['prior_patch_pixels'] for r in rows),
                                     prior_full_grid_pixels=sum(r[name]['prior_full_grid_pixels'] for r in rows),
                                     background_fallback_frames=sum(r[name]['background_mode'] not in ('affine','bg_lattice','bg_sparse_pixels','initialization') for r in rows),
                                     mean_preprocessing_ms=float(np.mean([r[name]['preprocessing_ms'] for r in rows])),
                                     mean_discovery_ms=float(np.mean([r[name]['flux_ms'] for r in rows])),
                                     mean_heartbeat_ms=float(np.mean([r[name]['heartbeat_ms'] for r in rows])),
                                     mean_ingestion_ms=float(np.mean(latency)), median_ingestion_ms=float(np.median(latency)),
                                     p95_ingestion_ms=float(np.percentile(latency, 95)),
                                     final_tracks=rows[-1][name]['tracks'], tentative=None, confirmed=None,
                                     nonempty_flux_frames=sum(r[name]['novel_flux_pixels_eval_grid'] > 0 for r in rows),
                                     nonempty_motion_frames=sum((r[name]['residual_motion_pixels'] or 0) > 0 for r in rows),
                                     flux_activity_rate=sum(r[name]['novel_flux_pixels_eval_grid'] > 0 for r in rows[1:]) / max(len(rows) - 1, 1))
            native, grid = summary['grid320'], summary['grid']
            iou = intersections / unions if unions else None
            passed = (iou is not None and iou >= .65 and native['nonempty_flux_frames'] > 0
                      and grid['nonempty_flux_frames'] > 0 and grid['mean_ingestion_ms'] <= 15)
            for name in pipelines:
                refs = [v for row in rows for v in row[name]['manual_reference']]
                total = sum(v['reference_pixels'] for v in refs)
                summary[name]['manual_reference_instances'] = len(refs)
                summary[name]['reference_pixel_recall'] = sum(v['covered_pixels'] for v in refs) / total if total else None
            mask_path = report / (meta['episode_id'] + '.flux_masks.npz')
            np.savez_compressed(mask_path, masks=np.asarray(mask_records, dtype=bool), timestamp_us=times)
            results.append(dict(episode_id=meta['episode_id'], video_id=meta['video_id'],
                                rgb_sha256=rgb_hash.hexdigest(), decoded_pixels=pixels, summary=summary,
                                iou_flux=iou, flux_intersection=intersections, flux_union=unions,
                                mask_artifact=str(mask_path.relative_to(ROOT)), mask_sha256=sha(mask_path),
                                local_gate_passed=passed, records=rows))
            atomic_json(report / 'progress.json', dict(completed=len(results), total=len(selected), episodes=results))
            print(f"{len(results)}/{len(selected)} {meta['video_id']}: IoU={iou}, grid mean={grid['mean_ingestion_ms']:.2f}ms, pass={passed}", flush=True)
        if resources['failure']:
            raise MemoryError(resources['failure'])
        # The supplied pre-holdout checks are stricter than the historical 15ms gate.
        preprocess = [row['grid']['preprocessing_ms'] for episode in results for row in episode['records']]
        warm = [row['grid']['ingestion_ms'] for episode in results for row in episode['records']]
        target_ids = {'dis_e56c2c85b7a2500492f8d63f7c916816', 'dis_e5ffcb291dc874d2a6220cc7cc48ad72', 'dis_2d54d66bfd929751b1376e8e26f28578'}
        focused = [e for e in results if e['episode_id'] in target_ids]
        test_report_path = ROOT / 'reports/track1_fmr/stage1_dropin_dev_v5_cached_unit_checks.json'
        test_report = json.loads(test_report_path.read_text()) if test_report_path.exists() else {}
        source_matches = test_report.get('source_hashes') == {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src/track1_fmr').glob('*.py'))}
        precheck = dict(
            median_preprocessing_ms=float(np.median(preprocess)), preprocessing_pass=float(np.median(preprocess)) <= 1.,
            mean_ingestion_ms=float(np.mean(warm)), ingestion_pass=float(np.mean(warm)) <= 12.,
            focus_episodes={e['episode_id']: dict(nonempty_flux_frames=e['summary']['grid']['nonempty_flux_frames'], cap_frames=e['summary']['grid']['flux_cap_frames'], cap_rate=e['summary']['grid']['flux_cap_rate'], median_novel_proposals=e['summary']['grid']['median_flux_proposals']) for e in focused},
            focus_pass=len(focused) == 3 and all(e['summary']['grid']['nonempty_flux_frames'] >= 5 and e['summary']['grid']['flux_cap_rate'] <= .10 and e['summary']['grid']['median_flux_proposals'] <= 5. for e in focused),
            cap_rule='Each focus clip: >=5/29 recovered pairs, <=10% of 29 frame pairs at configured 10-proposal cap and median <=5 novel proposals; all-episode stats also reported',
            unit_checks_pass=test_report.get('passed') is True and test_report.get('merge_border_ego_regressions') is True and source_matches,
            pool_complete=len(results) == len(dev_pool()['episodes']),
            rss_pass=resources['peak_rss'] <= 18 * 2**30)
        precheck['all_four_passed'] = all(precheck[key] for key in ['preprocessing_pass','ingestion_pass','focus_pass','unit_checks_pass']) and precheck['pool_complete'] and precheck['rss_pass']
        atomic_json(report / 'preholdout_checks.json', precheck)
        atomic_json(report / 'result.json', dict(protocol_sha256=sha(report / 'protocol.json'),
                    warmup_seconds=warmup_seconds, heartbeat_calls=heartbeat.calls,
                    sampled_process_tree_peak_rss_gib=resources['peak_rss'] / 2**30,
                    memory_sampling_interval_ms=50, process_discovery='Before timing each decoded frame, cached PIDs sampled during ingestion', episodes=results,
                    novel_flux_episode_rate=sum(e['summary']['grid']['nonempty_flux_frames'] > 0 for e in results) / len(results),
                    local_gate_passed=all(r['local_gate_passed'] for r in results),
                    preholdout_checks_passed=precheck['all_four_passed'],
                    qualification_passed=(args.split == 'holdout' and float(np.mean(warm)) <= 15.
                        and sum(e['summary']['grid']['nonempty_flux_frames'] > 0 for e in results)/len(results) >= .85
                        and resources['peak_rss'] <= 18*2**30),
                    promotion_qualified=False, lifecycle_implemented=False,
                    qualification_limitation='Stage 2 not started; dev results cannot qualify promotion; unexplained silence needs independent adjudication'))
    except Exception as exc:
        atomic_json(report / 'failure.json', dict(error=repr(exc), episodes=results, resources=resources))
        raise
    finally:
        stop.set()
        thread.join()


if __name__ == '__main__':
    main()
