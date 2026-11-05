"""Atomic shards, versioned manifests, bounded resource accounting and progress."""
from pathlib import Path
import os,json,time,threading,logging,hashlib,sys
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from pipeline.common import ROOT,digest,atomic_json

LOG=logging.getLogger('track1')

def load_config(path='configs/track1.yaml'):
    import yaml
    return yaml.safe_load((ROOT/path).read_text())

def code_hash():
    return hashlib.sha256(''.join(digest(p) for p in sorted((ROOT/'src/track1').glob('*.py'))).encode()).hexdigest()

def identity(cfg):
    return dict(schema=cfg['schema'],code_sha256=code_hash(),config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                canonical_freeze_sha256=digest(ROOT/'reports/freeze_manifest.json'),plan_sha256=digest(ROOT/'configs/track1_implementation_plan.md'))

def versions():
    import importlib.metadata as m
    return {p:m.version(p) for p in ['torch','torchvision','numpy','scipy','PyWavelets','psutil','pyarrow']}

def output_dir(cfg,stage):
    p=ROOT/cfg['output']/f'track1_{stage}_v1';p.mkdir(parents=True,exist_ok=True);return p

def save_shard(path,arrays):
    import torch
    path=Path(path);temp=path.with_suffix('.pt.tmp')
    torch.save({k:torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v,np.ndarray) else v for k,v in arrays.items()},temp)
    temp.replace(path)

def load_shard(path):
    import torch
    return {k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in torch.load(path,map_location='cpu',weights_only=True,mmap=True).items()}

def save_index(path,rows):
    path=Path(path);temp=path.with_suffix('.parquet.tmp')
    pq.write_table(pa.Table.from_pylist(rows),temp,compression='zstd');temp.replace(path)

def manifest(cfg,stage,v,files,extra=None):
    target=output_dir(cfg,stage)/f'{v}.manifest.json'
    inputs=[ROOT/'reports/freeze_manifest.json',ROOT/'configs/track1_implementation_plan.md']
    if stage=='spatial':
        inputs += [ROOT/cfg['input']/f'{table}.{v}.parquet' for table in ['masks','annotation_frames','camera_frames','videos']]
        inputs += [ROOT/f'reports/track1_camera.{v}.json']
    if stage in ['calibration','appearance','functional'] and v not in ['model','projection']:
        inputs += [output_dir(cfg,'spatial')/f'{v}.{suffix}' for suffix in ['pt','parquet']]
    if stage in ['appearance','functional']:
        inputs += [output_dir(cfg,'calibration')/'model.json']
    if stage=='appearance':
        inputs += [ROOT/'work/track1_weights/mobilenet_v3_small-047dcff4.pth']
        if v!='projection':inputs += [output_dir(cfg,'appearance')/'projection.pt']
    if stage=='functional':inputs += [output_dir(cfg,'appearance')/f'{v}.pt']
    if stage=='calibration' and v=='model':inputs += list(output_dir(cfg,'calibration').glob('*.parquet'))
    atomic_json(target,dict(**identity(cfg),video_id=v,stage=stage,completed=True,git_commit=None,git_reason='Workspace is not a Git checkout',versions=versions(),
        inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},outputs={str(p.relative_to(ROOT)):digest(p) for p in files},metrics=extra or {}))

def reusable(cfg,stage,v):
    path=output_dir(cfg,stage)/f'{v}.manifest.json'
    if not path.exists():return False
    m=json.loads(path.read_text())
    return m.get('completed') and all(m.get(k)==v for k,v in identity(cfg).items()) and all((ROOT/p).exists() and digest(ROOT/p)==h for p,h in {**m.get('inputs',{}),**m['outputs']}.items())

def memory_guard(cfg):
    process=psutil.Process();rss=process.memory_info().rss
    for child in process.children(recursive=True):
        try:rss+=child.memory_info().rss
        except psutil.Error:pass
    available=psutil.virtual_memory().available
    if available<cfg['minimum_available_gib']*2**30 or rss>cfg['memory_budget_gib']*2**30:
        raise MemoryError(f'Resource guard: RSS={rss/2**30:.2f} GiB, available={available/2**30:.2f} GiB; progress is checkpointed')
    return rss,available

class Monitor:
    def __init__(self,cfg):self.cfg=cfg;self.stop=threading.Event();self.failure=None
    def __enter__(self):
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def loop(self):
        path=ROOT/'reports/track1_resources.jsonl'
        while not self.stop.is_set():
            try:
                rss,available=memory_guard(self.cfg);record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),rss_gib=rss/2**30,available_gib=available/2**30,cpu_percent=psutil.cpu_percent())
                # Do not initialise Metal in CPU worker pools just for monitoring.
                torch=sys.modules.get('torch')
                if torch is not None and hasattr(torch,'mps') and hasattr(torch,'backends') and torch.backends.mps.is_available():
                    record.update(mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
                with path.open('a') as f:f.write(json.dumps(record)+'\n')
            except MemoryError as e:self.failure=e;LOG.error('%s',e);return
            except Exception as e:LOG.warning('Resource monitor: %s',e)
            self.stop.wait(5)
    def check(self):
        if self.failure:raise self.failure
        memory_guard(self.cfg)
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=6)
