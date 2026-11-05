import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import argparse,logging,time,fcntl
from . import *
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['correspondence','camera','motion','tiered','evaluation','verify','all'],default='all');args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
 with (W/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic_json(D/'status.json',dict(ready=False,stage=args.stage,scientific_acceptance=False,finalized=False,track2_started=False))
  with Monitor() as monitor:
   for stage in ['correspondence','camera','motion','tiered','evaluation','verify'] if args.stage=='all' else [args.stage]:
    LOG.info('BEGIN %s',stage);start=time.monotonic()
    if stage=='correspondence':from .correspondence import run
    elif stage=='camera':from .motion import camera as run
    elif stage=='motion':from .motion import run
    elif stage=='tiered':from .store import build as run
    elif stage=='evaluation':from .evaluation import run
    else:from .evaluation import verify as run
    run(monitor)
    if stage=='evaluation':
     from .evaluation import geometry_quality
     geometry_quality(monitor)
    if stage=='motion':
     from .motion import synthetic
     synthetic(monitor)
    monitor.check();LOG.info('COMPLETE %s %.1fs',stage,time.monotonic()-start)
if __name__=='__main__':main()
