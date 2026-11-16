"""Recompute Stage 2 observed-only counts, verify frozen inputs and report gate failures."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest


def verify(tag):
    report=ROOT/'reports/track1_fmr'/tag
    result=json.loads((report/'result.json').read_text());protocol=json.loads((report/'protocol.json').read_text())
    assert digest(report/'protocol.json')==result['protocol_sha256']
    for rel,sha in protocol['source_hashes'].items():assert digest(report/'source'/rel)==sha,rel
    for rel,sha in {**protocol['source_ledgers'],**protocol['reference_readiness_hashes']}.items():assert digest(ROOT/rel)==sha,rel
    dorsal='src/track1_fmr/dorsal_flux.py'
    freeze=json.loads((ROOT/'reports/track1_fmr/stage1_holdout_v4/protocol.json').read_text())
    assert protocol['source_hashes'][dorsal]==freeze['source_hashes'][dorsal]
    counts=Counter();sources=Counter();segments=Counter();fixed_segments=0;unique_ids=0;observation_rows=0;initial_max_hits=[]
    for episode in result['episodes']:
        stem=episode['episode_id']
        for suffix,key in [('frames.json','frames_sha256'),('observations.npz','observations_sha256'),('tubes.parquet','ledger_sha256')]:
            assert digest(report/(stem+'.'+suffix))==episode[key]
        rows=json.loads((report/(stem+'.frames.json')).read_text());counts['decoded_frames']+=len(rows)
        counts['post_seed_frames']+=sum(r['post_seed'] for r in rows)
        eligible=[r for r in rows if r['post_seed'] and r['source_centroid_valid']];counts['source_reference_eligible']+=len(eligible)
        good=[r for r in eligible if r['fixed_id_observed'] and r['source_centroid_error_px'] is not None and r['source_centroid_error_px']<=50]
        assert len(good)==episode['observed_only_proxy_successes']
        counts['observed_only_proxy_successes']+=len(good);sources.update(r.get('measurement_source','proposal') for r in good)
        with np.load(report/(stem+'.observations.npz')) as z:
            assert z['values'].shape[1]==28 and np.isfinite(z['values']).all();observation_rows+=len(z['values'])
            if 'observed_counts' in z:
                hit_counts=z['observed_counts'][z['values'][:,1]==episode['initialized_fixed_id']]
                initial_max_hits.append(int(hit_counts.max()) if len(hit_counts) else 0)
        table=pq.read_table(report/(stem+'.tubes.parquet'));tags=table.column('epistemic_tag').to_pylist();ids=table.column('track_id').to_pylist()
        assert tags.count('OBSERVED')==episode['observed_segments'] and tags.count('PREDICTED')==episode['predicted_segments']
        segments.update(tags);fixed_segments+=ids.count(episode['initialized_fixed_id']);unique_ids+=len(set(ids))
    assert all(result[k]==v for k,v in counts.items())
    budget=[]
    for episode in result['episodes']:
        rows=json.loads((report/(episode['episode_id']+'.frames.json')).read_text())
        if rows and 'savpe_forward_on_frame' in rows[0]:
            frames=[i for i,row in enumerate(rows) if row['savpe_forward_on_frame']]
            assert all(row['savpe_forward_on_frame'] in (0,1) for row in rows)
            assert all(b-a>=10 for a,b in zip(frames,frames[1:]))
            assert len(frames)==rows[-1]['savpe_calls'];budget.append(len(frames))
    n=len(result['episodes']);mean_segments=segments['OBSERVED']/n
    checks=dict(initialization=result['fixed_ids_initialized']==n,observed_only_proxy_coverage=result['observed_only_proxy_coverage']>=.65,
                mean_ingestion_ms=result.get('mean_ingestion_ms',float('inf'))<=13.5,
                sampled_rss=result['sampled_process_tree_peak_rss_gib']<=18)
    value=dict(artifact_integrity_passed=True,stage1_extractor_hash_unchanged=True,counts=dict(counts),
        successful_measurement_sources=dict(sources),finite_observation_rows=observation_rows,initial_ids_never_confirmed=sum(n<3 for n in initial_max_hits),initial_ids_never_archive_eligible=sum(n<5 for n in initial_max_hits),segments=dict(segments),
        unique_ids_in_emitted_segments=unique_ids,mean_all_observed_segments_per_episode=mean_segments,
        fixed_id_segments=fixed_segments,mean_fixed_id_segments_per_episode=fixed_segments/n,
        fragmentation_caveat='Tube boundaries include event/fit/status boundaries across all entities; not physical-ID fragmentation truth',
        engineering_checks=checks,cohort_engineering_passed=all(checks.values()),independent_identity_verified=False,
        external_lmstudio_memory_included=False,savpe_forward_budget_verified=bool(budget),savpe_forward_calls=sum(budget),stage2_promoted=False)
    atomic_json(report/'verification.json',value);print(tag,json.dumps(value),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('tags',nargs='+');args=ap.parse_args()
    for tag in args.tags:verify(tag)
