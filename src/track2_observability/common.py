import json,threading,time
from pathlib import Path
from pipeline.common import ROOT,atomic_json,digest
from track1_retrain.common import guard
C=json.loads((ROOT/'configs/track2_observability.json').read_text())
D=ROOT/'derived/track2_observability';R=ROOT/'reports/track2_observability';W=ROOT/'work/track2_observability'
OLD=ROOT/'derived/track2_measurement'

def hashes(paths):return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths)) if p.is_file()}

def validate(manifest):
    for path,value in manifest.items():
        if digest(ROOT/path)!=value:raise ValueError('Integrity mismatch: '+path)

def dependencies():
    # Include imported local helpers conservatively, rather than maintain a fragile hand-written import graph.
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.json'))+list((ROOT/'tests').glob('*.py'))
    for folder in ['derived/track2_feasibility','derived/track2_measurement','derived/track1_context_v3/spatial']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for p in C['training_participants']+C['validation_participants']:
        paths+=list((ROOT/'canonical/v2').glob('masks.'+p+'_*.parquet'))+list((ROOT/'derived/track1_refined_v2/spatial').glob(p+'_*.camera.json'))
    paths+=[ROOT/'dataset/context_v3/episodes.parquet',ROOT/'reports/track2_measurement/target_audit.parquet',ROOT/'reports/track2_measurement/baseline_selection.json']
    result=hashes(paths)
    import sys,torch,numpy,scipy,pyarrow
    result["runtime_versions"]=json.dumps([sys.version,torch.__version__,numpy.__version__,scipy.__version__,pyarrow.__version__])
    return result

def stage(name,function):
    identity=dependencies();path=R/f'{name}.receipt.json';started=R/f'{name}.started.json'
    if started.exists():
        if json.loads(started.read_text())['dependencies']!=identity:raise ValueError('Interrupted-stage dependencies changed for '+name)
    else:atomic_json(started,dict(dependencies=identity,started_at=time.time()))
    if path.exists():
        receipt=json.loads(path.read_text())
        if receipt['dependencies']!=identity:raise ValueError('Resume dependencies changed for '+name)
        validate(receipt['outputs']);return
    function()
    after=hashes(list(R.glob('*'))+list(D.glob('*'))+list(W.glob('*.json')))
    # Include preexisting identical outputs after a partial-stage retry as well.
    outputs={p:h for p,h in after.items() if not p.endswith(('.receipt.json','.started.json'))}
    atomic_json(path,dict(dependencies=identity,outputs=outputs,completed_at=time.time()))

class Monitor:
    def __enter__(self):
        self.stop=threading.Event();self.error=None;self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def loop(self):
        while not self.stop.is_set():
            try:
                rss,available=guard()
                with (W/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),tree_rss_gib=rss,available_gib=available))+'\n')
            except Exception as e:self.error=e;return
            self.stop.wait(5)
    def check(self):
        if self.error:raise self.error
        guard()
    def __exit__(self,*args):self.stop.set();self.thread.join(6)
