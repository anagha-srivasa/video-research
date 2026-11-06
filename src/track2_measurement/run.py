"""Resumable bounded measurement/anchor experiment; old releases remain immutable."""
import argparse,json,logging,time,threading
import numpy as np
from pipeline.common import ROOT,atomic_json,digest
from track1_retrain.common import guard,pool
from track2_feasibility.data import check_frozen
from .data import C,D,R,W,prepare
from .anchors import baselines,fold,final

class Monitor:
    def __enter__(self):
        self.stop=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
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

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','evaluate','diagnostics','all'],default='all');args=parser.parse_args()
    for p in (D,R,W):p.mkdir(exist_ok=True,parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    paths=list((ROOT/'src/track2_measurement').glob('*.py'))+[ROOT/'configs/track2_measurement.json']
    identity={str(p.relative_to(ROOT)):digest(p) for p in paths}
    lock=R/'protocol_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())['source_hashes']!=identity:raise ValueError('Protocol source changed; review amendment before resuming')
    else:atomic_json(lock,dict(source_hashes=identity,config=C,created=time.time(),validation_selection=False))
    check_frozen()
    with Monitor() as monitor:
        if args.stage in ('train','all'):
            if not (R/'support.json').exists():prepare()
            if not (R/'baseline_selection.json').exists():baselines()
            jobs=[(f,s,p) for f in C['families'] for s in C['seeds'] for p in C['training_participants']]
            logging.info('Training %d participant-held-out jobs',len(jobs));records=pool(fold,jobs,monitor,workers=C['workers'])
            selected=[(f,s,int(np.median([r['epochs'] for r in records if r['family']==f and r['seed']==s]))) for f in C['families'] for s in C['seeds']]
            atomic_json(R/'epoch_selection.json',dict(jobs=selected,validation_used=False))
            logging.info('Training twelve final refits');pool(final,selected,monitor,workers=C['workers'])
        if args.stage in ('evaluate','all'):
            from .evaluate import evaluate
            evaluate()
        if args.stage in ('diagnostics','all'):
            from .diagnostics import diagnostics
            diagnostics()
    check_frozen();logging.info('Completed %s; frozen Track 1 hashes unchanged',args.stage)

if __name__=='__main__':main()
