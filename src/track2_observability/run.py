import argparse,logging,os
from pipeline.common import atomic_json,ROOT
from .common import C,D,R,W,Monitor,stage,dependencies

def main():
    parser=argparse.ArgumentParser();stages=['prepare','train','evaluate','memory','relations','verify','report'];parser.add_argument('--stage',choices=stages+['all'],default='all');args=parser.parse_args()
    for p in (D,R,W):p.mkdir(exist_ok=True,parents=True)
    os.environ.setdefault('MPLCONFIGDIR',str(W/'matplotlib'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    from .data import prepare
    from .train import train
    from .evaluate import evaluate
    from .memory import run as memory
    from .relations import run as relations
    from .report import verify,render
    with Monitor() as monitor:
        functions={'prepare':prepare,'train':lambda:train(monitor),'evaluate':evaluate,'memory':memory,'relations':relations,'verify':verify,'report':render}
        for name in stages if args.stage=='all' else [args.stage]:
            logging.info('Starting %s',name);stage(name,functions[name]);monitor.check();logging.info('Completed %s',name)
    logging.info('Complete; no promotion')

if __name__=='__main__':main()
