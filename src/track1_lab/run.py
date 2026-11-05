import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import argparse,logging,fcntl,time
from .common import path,Monitor,LOG,audit

def main():
 import sys
 from .common import config,seed_runtime
 seed=str(config()['seed'])
 if os.environ.get('PYTHONHASHSEED')!=seed:
  os.environ['PYTHONHASHSEED']=seed;os.execv(sys.executable,[sys.executable,'-m','track1_lab.run',*sys.argv[1:]])
 seed_runtime()
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['audit','temporal','spatial','contacts','appearance','motion','uncertainty','build-spatial','derivatives','motion-dense','topology','architecture','build-functional','variants','pilot32','contact-architecture','stream-eval','compound','kinematics','residual-stress','review-fixes','scientific-audit','hybrid','hybrid-evaluate','remediation','remediation-audit','kalman-stress','report']);a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');lock=(path('work')/(a.stage+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
 with Monitor() as monitor:
  if a.stage=='audit':audit()
  elif a.stage=='temporal':
   from .temporal import breakdown,search
   LOG.info('Channel breakdown %s',breakdown(monitor));search(monitor)
  elif a.stage=='spatial':
   from .spatial import search
   search(monitor)
  elif a.stage=='contacts':
   from .contacts import evaluate
   evaluate(monitor)
  elif a.stage=='appearance':
   from .appearance import run
   run(monitor)
  elif a.stage=='motion':
   from .motion import run
   run(monitor)
  elif a.stage=='uncertainty':
   from .uncertainty import run
   run(monitor)
  elif a.stage=='build-spatial':
   from .build import run
   run(monitor)
  elif a.stage=='derivatives':
   from .derivatives import run
   run(monitor)
  elif a.stage=='motion-dense':
   from .motion import run
   run(monitor,dense=True)
  elif a.stage=='topology':
   from .topology import run
   run(monitor)
  elif a.stage=='architecture':
   from .architecture import run
   run(monitor)
  elif a.stage=='build-functional':
   from .working import run
   run(monitor)
  elif a.stage=='variants':
   from .variants import run
   run(monitor)
  elif a.stage=='pilot32':
   from .pilot32 import run
   run(monitor)
  elif a.stage=='contact-architecture':
   from .contact_architecture import run
   run(monitor)
  elif a.stage=='stream-eval':
   from .stream_eval import run
   run(monitor)
  elif a.stage=='compound':
   from .compound import run
   run(monitor)
  elif a.stage=='kinematics':
   from .kinematics import run
   run(monitor)
  elif a.stage=='review-fixes':
   from .review_fixes import run
   run(monitor)
  elif a.stage=='scientific-audit':
   from .scientific import run
   run(monitor)
  elif a.stage=='hybrid':
   from .hybrid import run
   run(monitor)
  elif a.stage=='hybrid-evaluate':
   from .hybrid import evaluate_stream
   evaluate_stream(monitor)
  elif a.stage=='remediation':
   from .remediation import run
   run(monitor)
  elif a.stage=='remediation-audit':
   from .remediation_audit import run
   run(monitor)
  elif a.stage=='kalman-stress':
   from .kinematics import run
   run(monitor,include_residual=True,include_kalman=True)
  elif a.stage=='residual-stress':
   from .kinematics import run
   run(monitor,include_residual=True)
  elif a.stage=='report':
   from .report import build
   build(monitor)
  monitor.check()
 LOG.info('STAGE %s COMPLETE %.1fs; experiments remain open',a.stage,time.monotonic()-start)
if __name__=='__main__':main()
