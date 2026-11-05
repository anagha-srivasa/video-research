import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import argparse,json,logging,multiprocessing,fcntl,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from .protocol import config,folder,benchmark_contract,Monitor,LOG
from .data import build_video

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','cache','search','spatial','calibrate-spatial','temporal-search','calibrate-temporal','appearance','functional','evaluate','diagnostics','freeze','all'],default='all');p.add_argument('--resume',action='store_true');args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    cfg=config();benchmark_contract();lock=(folder('cache')/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from .workflow import audit_baseline,build_spatial_all,activate_refined,appearance_all,functional_all,indices
    from .calibration import spatial_calibration,temporal_calibration
    stages=['audit','cache','search','spatial','calibrate-spatial','temporal-search','calibrate-temporal','appearance','functional','evaluate','diagnostics','freeze'] if args.stage=='all' else [args.stage]
    with Monitor(cfg) as monitor:
        for stage in stages:
            start=time.monotonic();LOG.info('STAGE %s START',stage)
            if stage=='audit':LOG.info('Verified baseline files %d',audit_baseline())
            elif stage=='cache':
                with ProcessPoolExecutor(max_workers=min(cfg['workers'],8),mp_context=multiprocessing.get_context('spawn')) as pool:
                    jobs=[pool.submit(build_video,v) for v in cfg['train']+cfg['validation']+cfg['test']]
                    for job in as_completed(jobs):monitor.check();LOG.info('CACHE PASS %s',job.result())
            elif stage in ['search','temporal-search']:
                sealed=folder('reports')/'test_evaluation.json'
                if sealed.exists():
                    if not args.resume:raise RuntimeError('Test evaluation is already sealed; do not retune against exposed test results')
                    LOG.info('Reusing sealed selection without retuning')
                elif args.resume and (folder('reports')/('temporal_selection.json' if stage=='temporal-search' else 'spatial_selection.json')).exists():LOG.info('Reusing selection checkpoint')
                else:
                    from .search import search_all
                    if stage=='temporal-search':activate_refined()
                    search_all(cfg,monitor,temporal_only=stage=='temporal-search')
            elif stage=='spatial':build_spatial_all(monitor)
            elif stage=='calibrate-spatial':spatial_calibration()
            elif stage=='calibrate-temporal':activate_refined();temporal_calibration()
            elif stage=='appearance':appearance_all(monitor)
            elif stage=='functional':functional_all(monitor);LOG.info('INDICES %s',indices())
            elif stage=='evaluate':
                from .evaluation import evaluate_all
                evaluate_all(monitor)
            elif stage=='diagnostics':
                from .diagnostics import scorecard
                scorecard(monitor)
            elif stage=='freeze':
                from .evaluation import freeze
                freeze()
            monitor.check();LOG.info('STAGE %s COMPLETE elapsed=%.1fs',stage,time.monotonic()-start)

if __name__=='__main__':main()
