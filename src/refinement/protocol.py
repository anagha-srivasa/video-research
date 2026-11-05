from pathlib import Path
import json,hashlib,logging,time,threading,sys,os
import yaml
from pipeline.common import ROOT,digest,atomic_json
from track1.artifacts import memory_guard,save_shard,load_shard,save_index
import psutil

LOG=logging.getLogger('refinement')

class Monitor:
    def __init__(self,cfg):self.cfg=cfg;self.stop=threading.Event();self.failure=None
    def __enter__(self):
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def loop(self):
        while not self.stop.is_set():
            try:
                rss,available=memory_guard(self.cfg)
                record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),rss_gib=rss/2**30,available_gib=available/2**30,cpu_percent=psutil.cpu_percent())
                torch=sys.modules.get('torch')
                if torch is not None and hasattr(torch,'mps') and torch.backends.mps.is_available():record.update(mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
                with (folder('reports')/'resources.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            except MemoryError as error:self.failure=error;return
            except Exception as error:LOG.warning('Resource monitor: %s',error)
            self.stop.wait(5)
    def check(self):
        if self.failure:raise self.failure
        memory_guard(self.cfg)
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=6)

def config():return yaml.safe_load((ROOT/'configs/track1_refinement.yaml').read_text())
def baseline_config():return yaml.safe_load((ROOT/config()['baseline_config']).read_text())
def folder(kind):
    cfg=config();p=ROOT/cfg['cache' if kind=='cache' else 'reports'];p.mkdir(parents=True,exist_ok=True);return p
def source_identity():
    cfg=config()
    return dict(baseline_freeze_sha256=digest(ROOT/'reports/track1_freeze_manifest.json'),config_sha256=digest(ROOT/'configs/track1_refinement.yaml'),
        code_sha256=hashlib.sha256(''.join(digest(p) for p in sorted((ROOT/'src/refinement').glob('*.py'))).encode()).hexdigest())
def rank(value):return hashlib.sha256(str(value).encode()).hexdigest()
def write_report(name,value):atomic_json(folder('reports')/name,value)
def split(video):
    cfg=config();return next(k for k in ['train','validation','test'] if video in cfg[k])
def benchmark_contract():
    cfg=config();assert not(set(cfg['train'])&set(cfg['validation']) or set(cfg['train'])&set(cfg['test']) or set(cfg['validation'])&set(cfg['test']))
    selection={r['video_id']:r for r in json.loads((ROOT/'dataset/metadata/pilot_selection.json').read_text())}
    assert all(selection[v]['visor_split']=='train' for v in cfg['train']+cfg['validation'])
    report=dict(**source_identity(),task='Annotated-contour compression and physical-time functional reconstruction',
        sources=[dict(url='https://github.com/epic-kitchens/VISOR-VOS',task='First-frame-mask-conditioned video object segmentation',comparable=False,reason='Predicts unseen masks from RGB; our spatial fitting receives the target contour.'),
                 dict(url='https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html',task='Shape-preserving interpolation',comparable=True,limits='Local interpolation baseline with more stored observations; not equal four-coefficient compression capacity.'),
                 dict(url='https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html',task='C2 interpolation',comparable=True,limits='All supplied knots; report capacity separately.')],
        train_videos=cfg['train'],validation_videos=cfg['validation'],test_videos=cfg['test'],test_policy='Selection is locked before new test evaluation. These pilot videos have prior baseline diagnostics; this is not a never-seen external benchmark.',
        dimensions=dict(controls=16,spatial_degree=3,state_channels=35,temporal_coefficients=4),
        stopping='Train/validation search with plateau checks; no repeated test-set tuning or unsupported public-benchmark parity claim.',
        metrics=['Held-out contour normal error and geometric fidelity','Per-protocol temporal RMSE in native and train-standardised feature units','95/68 percent uncertainty coverage and NLL','Retention, invalid-fit rate and conditioning','Runtime, memory and coefficient budget'])
    write_report('benchmark_contract.json',report);return report
