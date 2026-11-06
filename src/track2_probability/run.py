import json,logging,time
import numpy as np
from pipeline.common import ROOT,atomic_json,digest
from track2_feasibility.data import Data
from track2_observability.common import hashes,validate
from track1_retrain.common import save,guard
from .core import arms,fit,scores
from .synthetic import run as synthetic
R=ROOT/'reports/track2_probability';D=ROOT/'derived/track2_probability';W=ROOT/'work/track2_probability'


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    for p in [R,D,W]:p.mkdir(exist_ok=True,parents=True)
    preserved=R/'preserved_v7.json'
    if not preserved.exists():
        old=json.loads((ROOT/'reports/track2_signal/report.receipt.json').read_text());items=dict(old['outputs']);items.update({k:v for k,v in old['dependencies'].items() if k!='runtime_versions'});validate(items);atomic_json(preserved,items)
    validate(json.loads(preserved.read_text()))
    dependencies=hashes(list((ROOT/'src/track2_probability').glob('*.py'))+list((ROOT/'derived/track2_signal').glob('*.npy'))+list((ROOT/'derived/track2_feasibility').glob('*'))+[ROOT/'derived/track2_measurement/recent_observation.npy',ROOT/'derived/track2_measurement/input_mask.npy'])
    protocol=dict(dependencies=dependencies,threshold_quantile=.75,regularization=1.,scale_calibration='LOPO training residual/conditional variance mean, equal participant weight; no validation fitting',arms=['energy','position','position_motion','position_metadata','both','relation','functional','memory'],no_selection=True,physical_reference_count=0,deterministic_models_frozen=True)
    lock=R/'protocol.json'
    if lock.exists():
        if json.loads(lock.read_text())!=protocol:raise ValueError('Protocol/dependencies changed')
    else:atomic_json(lock,protocol)
    data=Data();raw=np.load(ROOT/'derived/track2_signal/features.npy');anchor=np.load(ROOT/'derived/track2_measurement/recent_observation.npy')[:,-1,:2];ok=np.load(ROOT/'derived/track2_measurement/input_mask.npy');cohorts=np.load(ROOT/'derived/track2_signal/cohorts.npy')
    coeff=data.x[:,:,:140].reshape(-1,4,4,35)[:,:,:,:2].astype(float);features=arms(raw,coeff,anchor)
    predictions=[];results=[];calibration=[];resources=[]
    for cohort in ['object','hand']:
        train=data.indices();train=train[ok[train]&(cohorts[train]==cohort)];val=data.indices('validation',usable=False);val=val[ok[val]&(cohorts[val]==cohort)]
        for hi,h in enumerate([.25,.5,1.]):
            ix=train[data.mask[train,hi]];v=val[data.mask[val,hi]];sq=np.sum((data.y[:,hi,:2]-anchor)**2,1)
            for arm,x in features.items():
                logging.info('%s %.2fs %s',cohort,h,arm);rss,available=guard();resources.append(dict(time=time.time(),rss_gib=rss,available_gib=available))
                fold_ratios=[];fold_constant=[];folds=[]
                for held in sorted(set(data.participants[ix])):
                    tr=ix[data.participants[ix]!=held];te=ix[data.participants[ix]==held];f=fit(x,sq,data.participants,tr)
                    fold_ratios.append(float(np.mean(sq[te]/(2*f['variance'][te]))));fold_constant.append(float(np.mean(sq[te]/(2*f['constant_variance']))));folds.append(dict(held=held,training=sorted(set(data.participants[tr])),threshold=f['threshold'],queries=len(te)))
                factor=max(float(np.mean(fold_ratios)),1e-8);constant_factor=max(float(np.mean(fold_constant)),1e-8);f=fit(x,sq,data.participants,ix)
                calibration.append(dict(cohort=cohort,horizon=h,arm=arm,factor=factor,constant_factor=constant_factor,folds=folds))
                np.savez(D/f'{cohort}.{hi}.{arm}.npz',scale=f['scale'],probability_weights=f['bp'],scale_weights=f['bs'],threshold=f['threshold'],calibration=factor,baseline_variance=f['constant_variance'],baseline_frequency=f['frequency'])
                for p in sorted(set(data.participants[v])):
                    ids=v[data.participants[v]==p];y=f['event'][ids]
                    for name,prob in [('frequency',np.full(len(ids),f['frequency'])),('half',np.full(len(ids),.5)),('logistic',f['probability'][ids])]:
                        results.append(dict(cohort=cohort,horizon=h,arm=arm,participant=p,model=name,**scores(prob,y)))
                    for name,var in [('constant',np.full(len(ids),f['constant_variance'])),('constant_calibrated',np.full(len(ids),f['constant_variance']*constant_factor)),('conditional',f['variance'][ids]),('conditional_calibrated',f['variance'][ids]*factor)]:
                        implied=np.exp(-f['threshold']**2/(2*var));nll=np.log(2*np.pi*var)+sq[ids]/(2*var)
                        results.append(dict(cohort=cohort,horizon=h,arm=arm,participant=p,model=name,**scores(implied,y),coverage95=float(np.mean(sq[ids]<=5.991464547*var)),mean_radius95=float(np.mean(np.sqrt(5.991464547*var))),nll=float(nll.mean()),logistic_disagreement=float(np.mean(abs(implied-f['probability'][ids])))))
                    for i in ids:
                        predictions.append(dict(cohort=cohort,horizon=h,arm=arm,participant=p,episode_id=data.samples[i]['episode_id'],sequence_id=data.samples[i]['sequence_id'],event=bool(f['event'][i]),squared_displacement=float(sq[i]),probability=float(f['probability'][i]),variance=float(f['variance'][i]),calibrated_variance=float(f['variance'][i]*factor),threshold=f['threshold']))
    save(R/'predictions.parquet',predictions);atomic_json(R/'scores.json',results);atomic_json(R/'calibration.json',calibration);atomic_json(R/'resources.json',resources)
    logging.info('Synthetic noise/variance controls');atomic_json(R/'synthetic.json',synthetic())
    validate(json.loads(preserved.read_text()));validate(dependencies)
    render(results)
    atomic_json(R/'status.json',dict(completed=True,scientific_acceptance=False,independent_references=0,source_shared_anchor_confound_unresolved=True,preserved_files=len(json.loads(preserved.read_text())),peak_sampled_rss_gib=max(r['rss_gib'] for r in resources)))
    atomic_json(R/'manifest.json',hashes([p for p in R.glob('*') if p.name!='manifest.json']+list(D.glob('*'))))
    logging.info('Complete; no promotion or new deterministic model')


def render(rows):
    import html
    lines=['# Probability signal validation','', 'Exploratory source-displacement diagnostics on the same two exposed participants. No physical-motion acceptance or model promotion.','', 'Scale multipliers use held-training-participant predictions only. AUC and reliability bins are descriptive; overlapping queries are not independent samples. Fixed 0.5 is a diagnostic, not a fitted baseline.','', '| Cohort | Horizon | Arm | Brier (participant mean) | AUC (participant mean) |','|---|---:|---|---:|---:|']
    for cohort in ['object','hand']:
        for h in [.25,.5,1.]:
            for arm in ['energy','position','position_motion','position_metadata','both','relation','functional','memory']:
                rs=[r for r in rows if r['cohort']==cohort and r['horizon']==h and r['arm']==arm and r['model']=='logistic'];auc=[r['auc'] for r in rs if r['auc'] is not None]
                lines.append(f"| {cohort} | {h} | {arm} | {np.mean([r['brier'] for r in rs]):.5f} | {np.mean(auc):.4f} |")
    lines+=['','## Fixed baselines and conditional-scale calibration','', '| Cohort | Horizon | Frequency Brier | Fixed 0.5 Brier | Both coverage before / after | Mean 95% radius before / after |','|---|---:|---:|---:|---:|---:|']
    for cohort in ['object','hand']:
        for h in [.25,.5,1.]:
            group=[r for r in rows if r['cohort']==cohort and r['horizon']==h and r['arm']=='both']
            def mean(model,key):return np.mean([r[key] for r in group if r['model']==model])
            lines.append(f"| {cohort} | {h} | {mean('frequency','brier'):.5f} | 0.25000 | {mean('conditional','coverage95'):.3f} / {mean('conditional_calibrated','coverage95'):.3f} | {mean('conditional','mean_radius95'):.4f} / {mean('conditional_calibrated','mean_radius95'):.4f} |")
    lines+=['','Position arms retain the v7 per-hop displacement norms. Functional augmentation contains position coefficients only. Every arm uses its own training-only RMS scaling and the same fixed regularization; this is a bounded capacity comparison, not causal attribution.','', 'The initial diagnostic run was archived under work/track2_probability/initial_diagnostic before correcting baseline feature parity and using each past hop own observation for its memory update. No parameters were selected from either validation run.','', '## Limits','Synthetic controls reproduce feature-level noise mechanisms, not the complete irregular annotation pipeline. Independent-anchor synthetic controls do not remove the shared-anchor confound from the real dataset. Source ancestry and independent physical references remain unresolved. No claim that observed gains survive independent measurement is supported.','', 'Memory is a fixed past-only position-coefficient projection; this tests additional position history, not certified shape or a Koopman generator. Relation gains remain confounded by common scene motion; no matched background-motion feature is added in this round.','', 'Detailed per-participant discrimination, reliability bins, 95% radii/coverage, NLL and Gaussian-implied versus logistic probabilities: scores.json. Per-query evidence: predictions.parquet. Training-only calibration provenance: calibration.json.']
    text='\n'.join(lines)+'\n';(R/'ASSESSMENT.md').write_text(text)
    (R/'dashboard.html').write_text('<!doctype html><meta charset="utf-8"><title>Probability validation</title><style>body{max-width:1200px;margin:40px auto;font:16px system-ui}pre{white-space:pre-wrap}</style><h1>Probability signal validation</h1><p><a href="scores.json">Participant scores</a> · <a href="synthetic.json">Noise controls</a> · <a href="calibration.json">Calibration</a></p><pre>'+html.escape(text)+'</pre>')

if __name__=='__main__':main()
