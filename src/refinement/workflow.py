"""CPU fitting, bounded Metal extraction, and source-compatible sequence assembly."""
from contextlib import contextmanager
from collections import defaultdict
import json,sys,os,multiprocessing,logging
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import save_index
from .protocol import config,baseline_config,folder,Monitor,LOG,split
from .build import directory,manifest,functional_video,spatial_video,assemble_spatial

_MODEL=None

def audit_baseline():
    records=json.loads((ROOT/'reports/track1_freeze_manifest.json').read_text())['files']
    for name,record in records.items():
        if digest(ROOT/name)!=record['sha256']:raise ValueError('Frozen baseline changed: '+name)
    return len(records)

def build_spatial_all(monitor):
    import psutil
    cfg=config();selection=folder('reports')/'spatial_selection.json'
    if not selection.exists():atomic_json(selection,json.loads((folder('reports')/'selection_locked.json').read_text())['spatial'])
    videos=cfg['train']+cfg['validation']+cfg['test'];videos.sort(key=lambda v:-pq.read_metadata(ROOT/f'derived/track1_spatial_v1/{v}.parquet').num_rows)
    for video in videos:
        n=pq.read_metadata(ROOT/f'derived/track1_spatial_v1/{video}.parquet').num_rows;out=folder('cache')/'spatial_chunks';out.mkdir(exist_ok=True);jobs=[]
        for begin in range(0,n,2048):
            end=min(begin+2048,n);marker=out/f'{video}.{begin:08d}.json';reuse=False
            if marker.exists():
                r=json.loads(marker.read_text());reuse=r.get('completed') and r['stop_index']==end and r['selection_sha256']==digest(selection) and r['kernel_sha256']==digest(ROOT/'src/refinement/kernels.py') and r['builder_sha256']==digest(ROOT/'src/refinement/build.py') and r['tensor_sha256']==digest(marker.with_suffix('.pt')) and r['index_sha256']==digest(marker.with_suffix('.parquet'))
            if not reuse:jobs.append((begin,end))
        if jobs:
            workers=max(1,min(cfg['workers'],len(jobs),int((psutil.virtual_memory().available/2**30-cfg['minimum_available_gib'])/1.2),int((cfg['memory_budget_gib']-2)/1.2)))
            pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'));pending={pool.submit(spatial_video,video,a,b,True):a for a,b in jobs}
            try:
                while pending:
                    done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
                    for future in done:LOG.info('SPATIAL CHUNK %s %d %s',video,pending.pop(future),future.result())
            except BaseException:
                for process in pool._processes.values():process.terminate()
                pool.shutdown(wait=True,cancel_futures=True);raise
            pool.shutdown()
        LOG.info('SPATIAL ASSEMBLED %s %s',video,assemble_spatial(video));monitor.check()

def activate_refined():
    cfg=config();files={}
    for v in cfg['train']+cfg['validation']+cfg['test']:
        for extension in ['pt','parquet']:
            p=directory('spatial')/f'{v}.{extension}'
            if not p.exists():raise FileNotFoundError(p)
            files[str(p.relative_to(ROOT))]=digest(p)
    atomic_json(folder('reports')/'representation.json',dict(temporal_source='refined',spatial_shards=files))

def run_pool(jobs,function,workers,cfg,monitor):
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    pending={pool.submit(function,job):job for job in jobs};results=[]
    try:
        while pending:
            done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
            for future in done:
                key=pending.pop(future);result=future.result();results.append(result);LOG.info('COMPLETE %s %s',key,result)
    except BaseException:
        for process in pool._processes.values():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True);raise
    pool.shutdown();return results

@contextmanager
def feature_cache_scope():
    # Legacy extraction accepts a fixed scratch path. Swap only rebuildable caches,
    # keep the original directory untouched under a backup, and always restore it.
    root=ROOT/'work/track1_features';backup=folder('cache')/'baseline_feature_cache';new=folder('cache')/'features'
    if backup.exists():raise RuntimeError('An interrupted feature-cache scope needs recovery; baseline cache is retained at '+str(backup))
    marker=folder('cache')/'feature_cache_scope.json';atomic_json(marker,dict(active=True,baseline_backup=str(backup.relative_to(ROOT)),refined_cache=str(new.relative_to(ROOT))))
    had_baseline=root.exists()
    if had_baseline:root.rename(backup)
    if new.exists():new.rename(root)
    else:root.mkdir()
    try:yield
    finally:
        if root.exists():root.rename(new)
        if had_baseline:backup.rename(root)
        atomic_json(marker,dict(active=False,restored_baseline=had_baseline))

def adapter_config():
    adapter=folder('cache')/'adapter';adapter.mkdir(exist_ok=True)
    for legacy,stage in [('track1_spatial_v1','spatial'),('track1_calibration_v1','calibration'),('track1_appearance_v1','appearance')]:
        p=adapter/legacy;target=directory(stage)
        if not p.exists():p.symlink_to(target,target_is_directory=True)
        if p.resolve()!=target.resolve():raise ValueError('Unexpected adapter target: '+str(p))
    return dict(baseline_config(),output=str(adapter.relative_to(ROOT)))

def appearance_worker(choice):
    global _MODEL
    import torch
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True)
    from track1.appearance import backbone,extract_video
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('Metal is required for this configured extraction run')
    torch.mps.set_per_process_memory_fraction(.2)
    if _MODEL is None:_MODEL=backbone().to('mps')
    cfg=config()
    with Monitor(cfg) as monitor:extract_video(adapter_config(),choice,_MODEL,torch.device('mps'),monitor)
    return choice['video_id']

def appearance_all(monitor):
    import fcntl
    from track1.appearance import fit_projection
    cfg=config();choices=json.loads((ROOT/'dataset/metadata/pilot_selection.json').read_text());choices.sort(key=lambda c:-pq.read_metadata(directory('spatial')/f"{c['video_id']}.parquet").num_rows)
    baseline_lock=(ROOT/'work/track1.lock').open('w');fcntl.flock(baseline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with feature_cache_scope():
            run_pool(choices,appearance_worker,3,cfg,monitor)
            report=fit_projection(adapter_config(),json.loads((ROOT/'dataset/metadata/pilot_selection.json').read_text()))
    finally:baseline_lock.close()
    out=directory('appearance');model=directory('calibration')/'model.json'
    manifest('appearance','projection',[out/'projection.pt',out/'projection.json'],[model,ROOT/'work/track1_weights/mobilenet_v3_small-047dcff4.pth']+[directory('spatial')/f'{v}.pt' for v in cfg['train']],report)
    for v in cfg['train']+cfg['validation']+cfg['test']:
        manifest('appearance',v,[out/f'{v}.pt'],[directory('spatial')/f'{v}.pt',model,out/'projection.pt'],report['evaluations'][v])
    return report

def functional_all(monitor):
    cfg=config();return run_pool(cfg['train']+cfg['validation']+cfg['test'],functional_video,8,cfg,monitor)

def indices():
    cfg=config();grouped=defaultdict(list);all_windows=[]
    for v in cfg['train']+cfg['validation']+cfg['test']:
        tuning=split(v);LOG.info('INDEX %s',v)
        path=directory('functional')/f'{v}.parquet';rows=pq.read_table(path).to_pylist();tracks=defaultdict(list)
        for r in rows:r['tuning_split']=tuning;tracks[r['entity_track_id']].append(r)
        save_index(path,rows);all_windows+=rows
        original=json.loads(path.with_name(v+'.manifest.json').read_text())
        manifest('functional',v,[path,path.with_suffix('.pt')],[ROOT/p for p in original['inputs']],original['metrics'])
        for track,rs in tracks.items():
            for i in range(len(rs)-15):
                seq=rs[i:i+16]
                if not all(r['valid'] for r in seq) or not all(b['window_end_us']-a['window_end_us']==250000 for a,b in zip(seq,seq[1:])):continue
                grouped[tuning].append(dict(video_id=v,entity_track_id=track,row_indices=[r['row_index'] for r in seq],sequence_length=16,start_us=seq[0]['window_start_us'],end_us=seq[-1]['window_end_us'],action_split=seq[0]['action_split'],visor_split=seq[0]['visor_split'],tuning_split=tuning))
    out=directory('indices')
    for name in ['train','validation','test']:save_index(out/f'{name}_index.parquet',grouped[name])
    for stage in ['spatial','functional']:
        tables=[pq.read_table(directory(stage)/f'{v}.parquet') for v in cfg['train']+cfg['validation']+cfg['test']]
        pq.write_table(pa.concat_tables(tables),directory(stage)/'index.parquet',compression='zstd')
    result=dict(sequence_counts={k:len(v) for k,v in grouped.items()},windows=len(all_windows),valid_windows=sum(r['valid'] for r in all_windows),split_policy='Source-compatible 5/2/4 tuning split, whole participants/videos. Original source and engineering splits remain metadata.')
    atomic_json(out/'manifest.json',result);return result
