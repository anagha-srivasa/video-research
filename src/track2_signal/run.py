import argparse,json,logging,os
from pipeline.common import ROOT,atomic_json
from .common import C,D,R,W,Monitor,stage,validate


def main():
    parser=argparse.ArgumentParser();names=['prepare','controls','memory','probes','summarize','verify','report'];parser.add_argument('--stage',choices=names+['all'],default='all');args=parser.parse_args()
    for p in [D,R,W]:p.mkdir(exist_ok=True,parents=True)
    os.environ.setdefault('MPLCONFIGDIR',str(W/'matplotlib'));logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    preserved=R/'preserved_v6.json'
    if preserved.exists():validate(json.loads(preserved.read_text()))
    else:
        receipt=json.loads((ROOT/'reports/track2_observability/report.receipt.json').read_text());hashes=dict(receipt['outputs']);hashes.update({p:h for p,h in receipt['dependencies'].items() if p!='runtime_versions'});validate(hashes);atomic_json(preserved,hashes)
    from .data import prepare
    from .controls import run as controls
    from .memory import run as memory
    from .probes import run as probes
    from .report import summarize,verify,render
    with Monitor() as monitor:
        functions={'prepare':prepare,'controls':lambda:controls(monitor),'memory':memory,'probes':probes,'summarize':summarize,'verify':verify,'report':render}
        for name in names if args.stage=='all' else [args.stage]:
            logging.info('Starting %s',name);stage(name,functions[name]);monitor.check();logging.info('Completed %s',name)
    validate(json.loads(preserved.read_text()));logging.info('Complete; v6 preserved, no promotion')

if __name__=='__main__':main()
