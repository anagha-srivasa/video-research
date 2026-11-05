import json,logging,hashlib,time,threading,os
from pathlib import Path
import numpy as np,yaml,psutil
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import memory_guard
LOG=logging.getLogger('track1_lab')
CONFIG=ROOT/'configs/track1_lab.yaml'
def seed_runtime():
 import random,torch
 seed=config()['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True,warn_only=False)
def config():return yaml.safe_load(CONFIG.read_text())
def path(kind):
 p=ROOT/({'reports':'reports/track1_lab','work':'work/track1_lab','derived':'derived/track1_lab'}[kind]);p.mkdir(parents=True,exist_ok=True);return p

def allowed(videos):
 cfg=config();assert set(videos)<=set(cfg['train']+cfg['validation']),'Exposed test videos are excluded from new lab experiments'
 return videos

def split(v):
 cfg=config();allowed([v]);return 'train' if v in cfg['train'] else 'validation'
def identity():return dict(geometry_contract_sha256=digest(ROOT/'src/data_v2/geometry.py'),config_sha256=digest(CONFIG),v2_freeze_sha256=digest(ROOT/'reports/track1_refinement/freeze_manifest.json'),code_sha256=hashlib.sha256(''.join(digest(p) for p in sorted((ROOT/'src/track1_lab').glob('*.py'))).encode()).hexdigest())
def report(name,value):atomic_json(path('reports')/name,value)
def audit():
 for name in ['track1_freeze_manifest.json','track1_refinement/freeze_manifest.json']:
  files=json.loads((ROOT/'reports'/name).read_text())['files']
  for p,r in files.items():assert digest(ROOT/p)==r['sha256'],p
 LOG.info('Frozen v1/v2 baseline audit passed')
 report('status.json',dict(state='open',finalized=False,track2_started=False,test_videos_used_for_tuning=False,**identity()))

class Monitor:
 def __init__(self):self.stop=threading.Event();self.error=None;self.last_warning=0.
 def __enter__(self):self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def loop(self):
  while not self.stop.is_set():
   try:
    cfg=config();rss,available=memory_guard(cfg);level='warning' if rss>=cfg.get('memory_warning_gib',18)*2**30 else 'above_target' if rss>=cfg.get('memory_target_gib',16)*2**30 else 'target';r=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),tree_rss_gib=rss/2**30,available_gib=available/2**30,cpu_percent=psutil.cpu_percent(),memory_level=level)
    if level=='warning' and time.monotonic()-self.last_warning>30:LOG.warning('RSS %.2f GiB: above warning threshold; hard limit remains %.1f GiB',rss/2**30,cfg['memory_budget_gib']);self.last_warning=time.monotonic()
    with (path('reports')/'resources.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   except MemoryError as e:self.error=e;return
   except Exception as e:LOG.warning('Monitor: %s',e)
   self.stop.wait(5)
 def check(self):
  if self.error:raise self.error
  memory_guard(config())
 def __exit__(self,*args):self.stop.set();self.thread.join(6)

def pool(function,jobs,monitor,workers=None):
 from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
 import multiprocessing
 ex=ProcessPoolExecutor(max_workers=workers or config()['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=seed_runtime);pending={ex.submit(function,j):j for j in jobs};out=[]
 try:
  while pending:
   done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
   for future in done:
    job=pending.pop(future);result=future.result();out.append(result);LOG.info('Complete %s',str(job)[:150])
 except BaseException:
  for proc in ex._processes.values():proc.terminate()
  ex.shutdown(wait=True,cancel_futures=True);raise
 ex.shutdown();return out
