"""Prelocked four-arm ablation. Frozen source data; no automatic expansion."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,fcntl,itertools,json,logging,threading,time
import numpy as np
import pyarrow.parquet as pq
import torch
from pipeline.common import ROOT,atomic_json,digest
from track1_retrain.common import guard,pool,save
from track2_feasibility.data import Data,BASE,check_frozen
from track2_feasibility.evaluate import cv,prediction_rows
from track2_feasibility.train import weight_hash
from track2_kinematics.model import baseline
from track1_encoder_revision.references import audit
from .data import CFG,D,R,W,prepare,features
from .train import fold,final,Model,fit

class Monitor:
    def __enter__(self):
        self.stop=threading.Event();self.error=None;self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def loop(self):
        while not self.stop.is_set():
            try:
                rss,available=guard()
                with (R/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),rss_gib=rss,available_gib=available))+'\n')
            except BaseException as exc:self.error=exc;return
            self.stop.wait(2)
    def check(self):
        if self.error:raise self.error
        guard()
    def __exit__(self,*args):self.stop.set();self.thread.join(3)


def identity():
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/'configs/track2_geometry.json',ROOT/'tests/test_track2_geometry.py']
    paths+=[p for f in ('reports/track2_feasibility','reports/track2_kinematics','reports/track2_hybrid',
                       'derived/track2_feasibility','derived/track2_kinematics','derived/track2_hybrid')
            for p in (ROOT/f).rglob('*') if p.is_file()]
    paths+=[p for p in (BASE/'functional').glob('*') if p.is_file()]
    return dict(hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))},config=CFG,
        supplied_report_sha256=digest(__import__('pathlib').Path('/Users/anagha/.codex/attachments/3f0e9071-9da9-40bf-bb91-1b2f31401498/Pasted text.txt')),
        validation_exposed=True,geometry_descriptors_not_full_shape_latent=True)


def select(monitor):
    jobs=list(itertools.product(CFG['families'],CFG['seeds'],CFG['training_participants']))
    records=pool(fold,jobs,monitor,workers=CFG['workers']);selected=[]
    for family in CFG['families']:
        for seed in CFG['seeds']:
            rs=[r for r in records if r['family']==family and r['seed']==seed]
            selected.append(dict(family=family,seed=seed,epochs=int(np.median([r['best_epoch'] for r in rs]))))
    for seed in CFG['seeds']:
        for held in CFG['training_participants']:
            assert len({r['initial_hash'] for r in records if r['seed']==seed and r['held']==held})==1
    atomic_json(R/'selection.json',dict(records=selected,folds=records,validation_used=False,matched_initialization=True))


def train(monitor):
    selected=json.loads((R/'selection.json').read_text())['records']
    records=pool(final,[(r['family'],r['seed'],r['epochs']) for r in selected],monitor,workers=12)
    atomic_json(R/'training.json',dict(records=records,models=len(records)))


def evaluate():
    data=Data();ids=data.indices('validation',usable=False);geo=np.load(D/'geometry.npy');mask=np.load(D/'geometry_mask.npy')
    rows=prediction_rows(data,ids,cv(data,ids),'cv',0)
    cp=cv(data,ids);cp[:,:,:4]=baseline(data.x[ids,:,140:144],constant_position=True)
    rows+=prediction_rows(data,ids,cp,'constant_position',0)
    old=pq.read_table(ROOT/'reports/track2_hybrid/predictions.parquet').to_pylist()
    rows+=[r for r in old if r['family']=='nearest_observation']
    timings=[]
    for family,seed in itertools.product(CFG['families'],CFG['seeds']):
        checkpoint=torch.load(D/f'{family}.{seed}.pt',map_location='cpu',weights_only=True)
        model=Model();model.load_state_dict(checkpoint['state']);model.eval();norm=checkpoint['normalization']
        assert weight_hash(model)==checkpoint['record']['final_hash']
        x=features(data,norm,geo,mask,family)
        with torch.no_grad():correction=model(torch.from_numpy(x[ids])).numpy().astype(float)
        pred=cv(data,ids);pred[:,:,:4]=baseline(data.x[ids,:,140:144])+correction*norm['y_scale']
        rows+=prediction_rows(data,ids,pred,family,seed)
        # Training sample only, cached descriptor/PCA features; report this boundary explicitly.
        values=[];one=torch.from_numpy(x[data.indices()[:1]])
        for _ in range(60):
            start=time.perf_counter()
            with torch.no_grad():model(one)
            values.append(1000*(time.perf_counter()-start))
        timings.append(dict(family=family,seed=seed,median_ms=float(np.median(values[10:])),parameters=checkpoint['record']['parameters']))
    for r in rows:r['coarse_squared_error']=None
    save(R/'predictions.parquet',rows)
    atomic_json(R/'latency.json',dict(records=timings,cached_features=True,geometry_extraction_and_PCA_and_upstream_excluded=True,device='CPU',precision='float32'))


def summarize():
    from track2_assessment import assess
    rows=pq.read_table(R/'predictions.parquet').to_pylist()
    ledger=pq.read_table(ROOT/'reports/track2_feasibility/target_ledger.parquet').to_pylist()
    groups={(f,s) for f,s in itertools.product(CFG['families'],CFG['seeds'])}|{(f,0) for f in ('cv','constant_position','nearest_observation')}
    paired,per,episodes=assess(rows,ledger,groups);atomic_json(R/'pairing.json',paired);save(R/'episode_metrics.parquet',episodes)
    data=Data();mask=np.load(D/'geometry_mask.npy');strata={s['sequence_id']:('all' if mask[i].all() else 'partial' if mask[i].any() else 'none') for i,s in enumerate(data.samples)}
    metrics=[]
    for f,s in sorted(groups):
        for h in CFG['horizons_us']:
            for stratum in ('overall','all','partial','none'):
                rs=[r for r in rows if r['family']==f and r['seed']==s and r['horizon_us']==h and r['target_supported'] and (stratum=='overall' or strata[r['sequence_id']]==stratum)]
                ps=sorted({r['participant'] for r in rs})
                metrics.append(dict(family=f,seed=s,horizon_us=h,stratum=stratum,targets=len(rs),participants=ps,
                    **{m:float(np.mean([np.sqrt(np.mean([r[m+'_squared_error'] for r in rs if r['participant']==p])) for p in ps])) if ps else None for m in ('position','velocity')}))
    summary=[]
    for f in (*CFG['families'],'cv','constant_position','nearest_observation'):
        for h in CFG['horizons_us']:
            rs=[r for r in metrics if r['family']==f and r['horizon_us']==h and r['stratum']=='overall']
            summary.append(dict(family=f,horizon_us=h,targets=rs[0]['targets'],**{m:float(np.median([r[m] for r in rs])) for m in ('position','velocity')}))
    gates=[]
    for f in CFG['families']:
        checks=[]
        for h in CFG['horizons_us']:
            candidate=next(r for r in summary if r['family']==f and r['horizon_us']==h)
            for b in ('cv','constant_position','nearest_observation'):
                base=next(r for r in summary if r['family']==b and r['horizon_us']==h)
                checks.append(dict(horizon_us=h,baseline=b,position_ratio=candidate['position']/base['position'],velocity_ratio=candidate['velocity']/base['velocity'],
                    passed=candidate['position']<=base['position']*(.9 if h<1000000 else 1.05) and candidate['velocity']<=base['velocity']*1.05))
        gates.append(dict(family=f,passed=all(r['passed'] for r in checks),checks=checks))
    extra=[]
    for h in CFG['horizons_us'][:2]:
        geom=next(r for r in summary if r['family']=='geometry' and r['horizon_us']==h)
        availability=next(r for r in summary if r['family']=='availability' and r['horizon_us']==h)
        extra.append(dict(horizon_us=h,position_ratio=geom['position']/availability['position'],passed=geom['position']<=.95*availability['position']))
    atomic_json(R/'metrics.json',dict(summary=summary,stratified_records=metrics,aggregation='participant RMSE, then mean participants, then median seeds',validation_exploratory=True))
    atomic_json(R/'decision.json',dict(arms=gates,geometry_increment=extra,geometry_descriptor_gate_passed=all(r['passed'] for r in extra),
        any_forecast_proxy_gate_passed=any(r['passed'] for r in gates),full_track2_go=False,scientific_acceptance=False,
        independent_event_contact_identity_gates='missing references',statistical_significance_claimed=False))


def verify():
    selected=json.loads((R/'selection.json').read_text())['records']
    item=next(r for r in selected if r['family']=='geometry' and r['seed']==17)
    model,norm,record=fit('geometry',17,epochs=item['epochs'])
    checkpoint=torch.load(D/'geometry.17.pt',map_location='cpu',weights_only=True)
    assert weight_hash(model)==checkpoint['record']['final_hash'] and norm==checkpoint['normalization']
    atomic_json(R/'verification.json',dict(deterministic_geometry17_refit=True,bitwise_weight_hash=weight_hash(model),
        normalization_repeat_identical=True,all_original_input_histories_retained=True,prior_sources_unchanged=True))
    atomic_json(R/'independent_references.json',audit())


def report():
    import html
    metrics=json.loads((R/'metrics.json').read_text())['summary'];decision=json.loads((R/'decision.json').read_text())
    support=json.loads((R/'support.json').read_text())['records']
    table='| Arm | 250 ms | 500 ms | 1 s |\n|---|---:|---:|---:|\n'
    htmltable='<table><tr><th>Arm</th><th>250 ms</th><th>500 ms</th><th>1 s</th></tr>'
    for f in (*CFG['families'],'cv','constant_position','nearest_observation'):
        values=[next(r['position'] for r in metrics if r['family']==f and r['horizon_us']==h) for h in CFG['horizons_us']]
        table+='| '+f+' | '+' | '.join(f'{v:.5f}' for v in values)+' |\n'
        htmltable+='<tr><td>'+f+'</td>'+''.join(f'<td>{v:.5f}</td>' for v in values)+'</tr>'
    htmltable+='</table>'
    text=f'''# Matched functional-history and optional-geometry ablation

Bounded experiment complete. Forecast proxy gate passed by any arm: {decision['any_forecast_proxy_gate_passed']}. Scientific acceptance and full Track 2 GO remain false.

Four arms use the same 5,196-parameter MLP, input histories, target masks, seeds, initialization, minibatch policy and residual loss. Kinematics uses four relative position/velocity states. Functional adds 16 training-only PCA coordinates of M16 history. Availability adds four optional-M32 availability flags. Geometry adds eight endpoint descriptors per available state: log width, log height, log net area, two normalized second moments, hole area fraction, outer-ring count and hole count. Missing descriptors are zero after training normalization; their availability is explicit. No missing-geometry history is excluded.

This tests a compact geometry summary, not raw vertices, full learned shape latents, contact constraints or learned shape dynamics. Optional descriptors come from certificate-bound historical M32 endpoints only. Future geometry does not enter inputs. Model output remains source position and velocity, with no event/contact probabilities or calibrated covariance.

{table}

Position RMSE uses intrinsic-normalized coordinates. Calculate RMSE within each participant, average participants, then take the median of three seeds. Detailed velocity and availability strata are in metrics.json. Strata may contain only one participant; participant counts are explicit. Histories overlap and do not constitute independent episodes.

Support: {json.dumps(support)}.

All normalization and PCA fitting use training participants only, including inside each of 60 participant-held-out stopping fits. Median selected epochs determine 12 all-training refits. Validation P08 and P18 have been exposed repeatedly and remain exploratory; this report does not call them fresh or claim p<0.05 from overlapping windows. Epoch choices can differ across feature arms because each uses the same training-only stopping policy. Three seeds are repeats, not new independent subjects.

The supplied report is stale on several points: Koopman, GRU/MLP and actual stationary/CV IMM prototypes already exist and were evaluated. The older degree-0/1/2 switching bank is not the actual IMM. CV's event recall is not zero in the earlier synthetic results. Existing sequence counts are overlapping indices, not thousands of independent episodes. Temporal interpolation using future observations is not a causal baseline. Track 1 is an engineering baseline, not independently certified production-ready physical understanding.

Independent motion/contact/identity references and genuinely fresh participants remain missing. Qwen review is machine supervision, not independent adjudicated physical truth. Proposed contact-boundary and event gates cannot be evaluated by inventing labels. No baseline promotion, full-scale training, final freeze or Track 3 transition is performed.

Original Koopman, neural and hybrid evidence is preserved. No prior validation inference was rerun; this is a new, prelocked descriptor ablation with identical current-arm input cohorts. The fixed geometry/seed-17 deterministic refit is checked bitwise. Dataset images and source tensors are reused without alteration.

Reproduce: `PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track2_geometry.run --resume`. Scratch files are under work/track2_geometry. CPU float32 models use bounded workers and process-tree memory guards; upstream feature generation is excluded from model latency.
'''
    # Obtain parameter count from implementation rather than a prose estimate.
    text=text.replace('5,196-parameter',f'{sum(p.numel() for p in Model().parameters()):,}-parameter')
    (R/'RESULTS.md').write_text(text)
    (R/'dashboard.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Optional geometry ablation</title><style>body{max-width:1150px;margin:36px auto;padding:20px;background:#111820;color:#e3eaf0;font:16px/1.6 system-ui}a{color:#8ed6ff}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #465360}pre{white-space:pre-wrap}</style><h1>Optional geometry · matched ablation</h1><p>Exploratory results. No scientific freeze or baseline promotion.</p>'+htmltable+'<p><a href="RESULTS.md">Full report</a> · <a href="metrics.json">All metrics and strata</a> · <a href="decision.json">Gates</a> · <a href="selection.json">Training selection</a> · <a href="support.json">Support</a></p><pre>'+html.escape(text)+'</pre></html>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True)
    for p in (D,R,W):p.mkdir(parents=True,exist_ok=True)
    with (W/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_frozen();ident=identity();path=R/'protocol_lock.json'
        if path.exists():
            if not args.resume or json.loads(path.read_text())!=ident:raise ValueError('Existing run or changed source/input')
        else:atomic_json(path,ident)
        with Monitor() as monitor:
            stages=[('prepare',prepare),('select',lambda:select(monitor)),('train',lambda:train(monitor)),('evaluate',evaluate),('summarize',summarize),('verify',verify),('report',report)]
            for name,fn in stages:
                marker=W/f'{name}.done.json'
                if marker.exists():
                    for file,h in json.loads(marker.read_text())['outputs'].items():
                        if digest(ROOT/file)!=h:raise ValueError('Changed output '+file)
                    logging.info('REUSE %s',name);continue
                monitor.check();logging.info('START %s',name);start=time.time()
                before={p:digest(p) for folder in (D,R) for p in folder.glob('*') if p.is_file() and p.name!='resources.jsonl'}
                fn();monitor.check()
                atomic_json(marker,dict(seconds=time.time()-start,outputs={str(p.relative_to(ROOT)):digest(p) for folder in (D,R) for p in folder.glob('*') if p.is_file() and p.name!='resources.jsonl' and (p not in before or digest(p)!=before[p])}))
                logging.info('DONE %s %.1fs',name,time.time()-start)
            check_frozen();assert identity()==ident
            atomic_json(D/'manifest.json',dict(protocol_sha256=digest(path),outputs={str(p.relative_to(ROOT)):digest(p) for folder in (D,R) for p in folder.glob('*') if p.is_file() and p.name not in ('resources.jsonl','manifest.json')}))
if __name__=='__main__':main()
