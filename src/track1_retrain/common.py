import json,logging,time,os,shutil,threading,hashlib
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,psutil
from pipeline.common import ROOT,atomic_json,digest
CFG=json.loads((ROOT/'configs/track1_retrain.json').read_text())
VIDEOS=CFG['train']+CFG['validation'];OUT=ROOT/'derived/track1_dataset_v2';REPORT=ROOT/'reports/track1_retrain';WORK=ROOT/'work/track1_retrain'
for p in [OUT,REPORT,WORK]:p.mkdir(parents=True,exist_ok=True)
LOG=logging.getLogger('track1_retrain')
def split(v):
 if v not in VIDEOS:raise ValueError('Exposed test participant is excluded')
 return 'train' if v in CFG['train'] else 'validation'
def folder(name):
 p=OUT/name;p.mkdir(exist_ok=True);return p
def save(path,rows):
 table=pa.Table.from_pylist(rows);temp=path.with_suffix('.tmp');pq.write_table(table,temp,compression=None);temp.replace(path)
def array(path,value):
 temp=path.with_suffix('.npy.tmp')
 with temp.open('wb') as f:np.save(f,np.asarray(value,np.float32),allow_pickle=False)
 temp.replace(path)
def rank(s):return hashlib.sha256((str(CFG['seed'])+str(s)).encode()).hexdigest()
def code():
 paths=list((ROOT/'src/track1_retrain').glob('*.py'))+[ROOT/'configs/track1_retrain.json',ROOT/'src/data_v2/geometry.py',ROOT/'src/track1_lab/geometry_contract.py',ROOT/'src/track1_lab/architecture.py',ROOT/'src/track1_lab/temporal.py',ROOT/'src/track1_lab/history_detail.py',ROOT/'src/track1_lab/causal_kinematics.py',ROOT/'src/track1_lab/topology_certificate.py']
 return {str(p.relative_to(ROOT)):digest(p) for p in paths}
def spatial_identity():
 # Stage-specific dependencies: adding a report must not invalidate spatial work.
 paths=[ROOT/f'src/track1_retrain/{p}.py' for p in ['spatial','common']]+[ROOT/'configs/track1_retrain.json',ROOT/'src/data_v2/geometry.py',ROOT/'src/track1_lab/architecture.py',ROOT/'src/track1_lab/geometry_contract.py']
 return {str(p.relative_to(ROOT)):digest(p) for p in paths}
def guard():
 rss=0
 for p in [psutil.Process(),*psutil.Process().children(recursive=True)]:
  try:rss+=p.memory_info().rss
  except psutil.Error:pass
 available=psutil.virtual_memory().available/2**30
 if rss>CFG['memory_limit_gib']*2**30 or available<CFG['minimum_available_gib']:raise MemoryError(f'RSS={rss/2**30:.2f} GiB available={available:.2f} GiB')
 if shutil.disk_usage(ROOT).free<CFG['disk_reserve_bytes']:raise OSError('Dataset/training disk reserve reached')
 return rss/2**30,available
class Monitor:
 def __enter__(self):self.stop=threading.Event();self.error=None;self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def loop(self):
  while not self.stop.is_set():
   try:
    rss,available=guard();r=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),tree_rss_gib=rss,available_gib=available,cpu_percent=psutil.cpu_percent(),disk_free_gb=shutil.disk_usage(ROOT).free/1e9)
    with (REPORT/'resources.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   except Exception as e:self.error=e;return
   self.stop.wait(5)
 def check(self):
  if self.error:raise self.error
  guard()
 def __exit__(self,*args):self.stop.set();self.thread.join(6)
def pool(function,jobs,monitor,workers=None):
 from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
 import multiprocessing
 ex=ProcessPoolExecutor(max_workers=workers or CFG['workers'],mp_context=multiprocessing.get_context('spawn'));jobs=iter(jobs);pending={};results=[];done_count=0
 try:
  while True:
   while len(pending)<2*(workers or CFG['workers']):
    try:job=next(jobs)
    except StopIteration:break
    pending[ex.submit(function,job)]=job
   if not pending:break
   done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
   for f in done:
    pending.pop(f);results.append(f.result());done_count+=1
    if done_count%8==0:LOG.info('Completed %d jobs',done_count)
 except BaseException:
  for p in ex._processes.values():p.terminate()
  ex.shutdown(wait=True,cancel_futures=True);raise
 ex.shutdown();return results
