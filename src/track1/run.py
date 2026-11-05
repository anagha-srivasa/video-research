"""Resumable Track 1 controller with CPU workers, Metal appearance and resource guards."""
import os
# Tiny solves use one BLAS thread per CPU worker; video-level parallelism owns the cores.
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
import argparse,json,logging,time,multiprocessing,fcntl
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import psutil
from pipeline.common import ROOT,atomic_json,digest
from .artifacts import load_config,Monitor,reusable,output_dir,identity
from .offline import spatial_video,functional_video
from .calibration import calibration_video,fit_spatial_calibration,temporal_calibration_cases,fit_temporal_calibration

LOG=logging.getLogger('track1')

def parallel(cfg,choices,stage,function,monitor,resume=False):
    jobs=[c for c in choices if not(resume and reusable(cfg,stage,c['video_id']))]
    workers=max(1,min(cfg['workers'],len(jobs) or 1,int((psutil.virtual_memory().available/2**30-cfg['minimum_available_gib'])/2.0),int(cfg['memory_budget_gib']/2.0)))
    LOG.info('%s: %d videos, %d CPU workers (memory-admitted)',stage,len(jobs),workers)
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    pending={pool.submit(function,cfg,c):c['video_id'] for c in jobs}
    try:
        while pending:
            done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
            for future in done:
                video=pending.pop(future);result=future.result();LOG.info('%s %s PASS %s',stage,video,result)
    except BaseException:
        for process in pool._processes.values():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True);raise
    pool.shutdown()

def audit():
    freeze=json.loads((ROOT/'reports/freeze_manifest.json').read_text())
    for relative,record in freeze['files'].items():
        if digest(ROOT/relative)!=record['sha256']:raise ValueError('Frozen canonical/source content changed: '+relative)
    return len(freeze['files'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/track1.yaml');parser.add_argument('--stage',default='all',choices=['all','audit','spatial','calibrate','appearance','functional','evaluate']);parser.add_argument('--video');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    cfg=load_config(args.config)
    if not (ROOT/cfg['output']).resolve().is_relative_to(ROOT):raise ValueError('Output must stay inside video_research')
    lock=(ROOT/'work/track1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    choices=json.loads((ROOT/'dataset/metadata/pilot_selection.json').read_text());choices=[c for c in choices if not args.video or c['video_id']==args.video]
    status=ROOT/cfg['output']/'track1_status.json';status.parent.mkdir(exist_ok=True);atomic_json(status,dict(ready=False,phase='processing',**identity(cfg)))
    start=time.monotonic()
    with Monitor(cfg) as monitor:
        stages=['audit','spatial','calibrate','appearance','functional','evaluate'] if args.stage=='all' else [args.stage]
        for stage in stages:
            LOG.info('STAGE %s START',stage)
            if stage=='audit':LOG.info('Verified %d frozen source/canonical files',audit())
            elif stage=='spatial':parallel(cfg,choices,'spatial',spatial_video,monitor,args.resume)
            elif stage=='calibrate':
                parallel(cfg,choices,'calibration',calibration_video,monitor,args.resume);fit_spatial_calibration(cfg,choices)
                parallel(cfg,choices,'temporal_calibration',temporal_calibration_cases,monitor);fit_temporal_calibration(cfg,choices)
            elif stage=='appearance':
                import torch
                from .appearance import backbone,extract_video,fit_projection
                torch.set_num_threads(16)
                if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable: explicitly select and benchmark a CPU run rather than silently switching')
                torch.mps.set_per_process_memory_fraction(.35);device=torch.device('mps');model=backbone().to(device)
                for c in choices:monitor.check();extract_video(cfg,c,model,device,monitor)
                del model;torch.mps.empty_cache();fit_projection(cfg,choices)
            elif stage=='functional':parallel(cfg,choices,'functional',functional_video,monitor,args.resume)
            elif stage=='evaluate':
                from .evaluation import evaluate_video,sequence_indices
                for c in choices:monitor.check();LOG.info('Evaluation %s %s',c['video_id'],evaluate_video(cfg,c)['manual_median_dominant_iou'])
                LOG.info('Sequence indices: %s',sequence_indices(cfg,choices))
            monitor.check();LOG.info('STAGE %s COMPLETE elapsed=%.1fs',stage,time.monotonic()-start)
    LOG.info('Requested stages complete. Sampled visual review and final gate audit are required before readiness.')

if __name__=='__main__':main()
