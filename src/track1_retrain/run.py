import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
import argparse,logging,time,fcntl
from .common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['spatial','appearance','search','refine-selection','functional','evaluate','verify','readiness-audit','transport','context','all'],default='all');a=p.parse_args()
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True);lock=(WORK/(a.stage+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.monotonic()
 if a.stage not in ['readiness-audit','transport','context']:atomic_json(OUT/'status.json',dict(ready=False,stage=a.stage,finalized=False,track2_started=False))
 with Monitor() as monitor:
  stages=['spatial','appearance','search','refine-selection','functional','evaluate','verify'] if a.stage=='all' else [a.stage]
  for stage in stages:
   LOG.info('BEGIN %s',stage);start=time.monotonic()
   if stage=='spatial':
    from .spatial import run
   elif stage=='appearance':
    from .appearance import run
   elif stage=='search':
    from .temporal import search as run
   elif stage=='refine-selection':
    from .selection import run
   elif stage=='functional':
    from .temporal import build as run
   elif stage=='evaluate':
    from .evaluation import run
   elif stage=='readiness-audit':
    from .readiness import run
   elif stage=='context':
    from .context import run
   elif stage=='transport':
    from .transport import run
   else:
    from .store import verify as run
   run(monitor);LOG.info('COMPLETE %s %.1fs',stage,time.monotonic()-start);monitor.check()
 LOG.info('Workflow complete %.1fs',time.monotonic()-started)
if __name__=='__main__':main()
