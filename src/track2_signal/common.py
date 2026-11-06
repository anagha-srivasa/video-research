import json,time
from pipeline.common import ROOT,atomic_json,digest
from track1_retrain.common import guard
from track2_observability.common import Monitor as BaseMonitor,validate,hashes
C=json.loads((ROOT/'configs/track2_signal.json').read_text())
D=ROOT/'derived/track2_signal';R=ROOT/'reports/track2_signal';W=ROOT/'work/track2_signal';OLD=ROOT/'derived/track2_measurement'

class Monitor(BaseMonitor):
    def loop(self):
        while not self.stop.is_set():
            try:
                rss,available=guard()
                with (W/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),tree_rss_gib=rss,available_gib=available))+'\n')
            except Exception as e:self.error=e;return
            self.stop.wait(5)

def identity():
    from track2_observability.common import dependencies
    result=dependencies();result.update(hashes(list((ROOT/'src/track2_signal').glob('*.py'))+[ROOT/'dataset/actions/EPIC_100_noun_classes_v2.csv',ROOT/'configs/track2_signal.json']))
    return result

def stage(name,function):
    dependencies=identity();start=R/f'{name}.started.json';receipt=R/f'{name}.receipt.json'
    if start.exists():
        if json.loads(start.read_text())['dependencies']!=dependencies:raise ValueError('Dependencies changed: '+name)
    else:atomic_json(start,dict(dependencies=dependencies))
    if receipt.exists():
        record=json.loads(receipt.read_text());validate(record['outputs']);return
    function()
    outputs=hashes(list(R.glob('*'))+list(D.glob('*'))+list(W.glob('*.json')))
    outputs={p:h for p,h in outputs.items() if not p.endswith(('.receipt.json','.started.json'))}
    atomic_json(receipt,dict(dependencies=dependencies,outputs=outputs))
