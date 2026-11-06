"""Fixed-capacity source probes: signed drift, proper probability scores, residual scale."""
import collections
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from pipeline.common import atomic_json
from track1_retrain.common import save
from track2_feasibility.data import Data
from track2_feasibility.train import participant_weights
from .common import C,D,R,OLD


def features(raw,paired,arm):
    x=raw.copy()
    if arm=='object':x[:,:,2:]=0
    elif arm=='reliability':x[:,:,8:20]=0
    elif arm=='missingness':x[:,:,8:15]=0
    elif arm=='time_shuffle':
        rng=np.random.default_rng(C['seed'])
        for i in range(len(x)):
            # Preserve presence/velocity masks; shuffle relational order within support-matched past hops.
            for presence,velocity in [(0,0),(1,0),(1,1)]:
                slots=np.flatnonzero((x[i,:,15]==presence)&(x[i,:,16]==velocity));order=rng.permutation(slots);x[i,slots,8:15]=raw[i,order,8:15]
    elif arm=='pair_shuffle':x=paired.copy()
    elif arm!='relation':raise ValueError('Unknown input arm')
    x=np.concatenate([x,np.linalg.norm(x[:,:,:2],axis=2,keepdims=True),np.linalg.norm(x[:,:,2:4],axis=2,keepdims=True)],axis=2)
    return x.reshape(len(x),-1)


def design(x,ids):
    scale=np.maximum(np.sqrt(np.mean(x[ids]**2,0)),.001);return np.c_[x/scale,np.ones(len(x))],scale


def weights(data,ids):
    w=participant_weights(data,ids).astype(float);return w/w.sum()


def drift_fit(data,x,raw,ids,variant):
    a,scale=design(x,ids);models=[]
    for hi in range(3):
        ix=ids[data.mask[ids,hi]];w=weights(data,ix);target=data.y[ix,hi,:2]-raw[ix,-1,:2];coef=np.zeros((a.shape[1],2))
        if variant=='intercept':coef[-1]=np.sum(w[:,None]*target,0)
        elif variant in ('ridge_no_intercept','ridge_intercept'):
            cols=a.shape[1] if variant=='ridge_intercept' else a.shape[1]-1;d=a[ix,:cols];penalty=np.eye(cols)*C['ridge_strength']
            if variant=='ridge_intercept':penalty[-1,-1]=0
            coef[:cols]=np.linalg.solve(d.T@(w[:,None]*d)+penalty,d.T@(w[:,None]*target))
        elif variant!='persistence':raise ValueError('Unknown drift variant')
        models.append(coef)
    return scale,np.array(models)


def drift_predict(x,raw,scale,coef):return raw[:,-1:,:2]+np.einsum('nd,hdk->nhk',np.c_[x/scale,np.ones(len(x))],coef)


def probabilistic_fit(a,target,w,kind,regularization):
    if kind=='probability':
        rate=float(np.sum(w*target));baseline=float(np.log(np.clip(rate,1e-6,1-1e-6)/(1-np.clip(rate,1e-6,1-1e-6))));start=np.zeros(a.shape[1]);start[-1]=baseline
        def objective(beta):
            z=a@beta;prediction=expit(z);pen=beta.copy();pen[-1]=0
            return float(np.sum(w*(np.logaddexp(0,z)-target*z))+.5*regularization*np.sum(pen*pen)),a.T@(w*(prediction-target))+regularization*pen
    else:
        baseline=float(np.log(max(np.sum(w*target)/2,1e-10)));start=np.zeros(a.shape[1])
        def objective(beta):
            linear=a@beta;unit=np.tanh(linear/3);logvar=baseline+3*unit;ratio=target*np.exp(-logvar)/2;pen=beta.copy();pen[-1]=0
            grad=(1-ratio)*(1-unit*unit)
            return float(np.sum(w*(logvar+ratio))+.5*regularization*np.sum(pen*pen)),a.T@(w*grad)+regularization*pen
    fit=minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12,'gtol':1e-8})
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():raise FloatingPointError('Nonfinite probability/scale optimization')
    return fit.x,baseline,dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),gradient_norm=float(np.linalg.norm(fit.jac)))


def probability_scores(probability,event):
    p=np.clip(probability,1e-8,1-1e-8);return (probability-event)**2,-event*np.log(p)-(1-event)*np.log1p(-p)


def run():
    data=Data();raw=np.load(OLD/'recent_observation.npy');ok=np.load(OLD/'input_mask.npy');information=np.load(D/'features.npy');paired=np.load(D/'paired_features.npy');cohorts=np.load(D/'cohorts.npy');rows=[];drift_rows=[];selections=[];optimization=[]
    for cohort in ['object','hand']:
        train=data.indices();train=train[ok[train]&(cohorts[train]==cohort)];val=data.indices('validation',usable=False);val=val[ok[val]&(cohorts[val]==cohort)]
        if not len(train) or not len(val):continue
        # Drift comparison uses the corrected relation arm for all four intercept policies.
        x=features(information,paired,'relation');folds=[]
        for held in C['training_participants']:
            fit_ids=train[data.participants[train]!=held];stop=train[data.participants[train]==held]
            if not len(stop):continue
            for variant in ['persistence','intercept','ridge_no_intercept','ridge_intercept']:
                norm,coef=drift_fit(data,x,raw,fit_ids,variant);pred=drift_predict(x,raw,norm,coef);scores=[]
                for hi in range(3):
                    ix=stop[data.mask[stop,hi]]
                    if len(ix):scores.append(float(np.sqrt(np.mean(np.sum((pred[ix,hi]-data.y[ix,hi,:2])**2,1)))))
                folds.append(dict(held=held,variant=variant,score=float(np.mean(scores))))
        selected=min(['persistence','intercept','ridge_no_intercept','ridge_intercept'],key=lambda variant:np.mean([r['score'] for r in folds if r['variant']==variant]));selections.append(dict(cohort=cohort,selected=selected,folds=folds))
        for variant in ['persistence','intercept','ridge_no_intercept','ridge_intercept']:
            scale,coef=drift_fit(data,x,raw,train,variant);np.save(D/f'drift.{cohort}.{variant}.npy',coef,allow_pickle=False);pred=drift_predict(x,raw,scale,coef)
            for i in val:
                for hi,h in enumerate(C['horizons_seconds']):
                    if data.mask[i,hi]:drift_rows.append(dict(cohort=cohort,variant=variant,participant=data.samples[i]['participant'],episode_id=data.samples[i]['episode_id'],horizon_seconds=h,position_squared_error=float(np.sum((pred[i,hi]-data.y[i,hi,:2])**2))))
        for arm in C['arms']:
            x=features(information,paired,arm);a,scale=design(x,train);np.save(D/f'feature_scale.{cohort}.{arm}.npy',scale,allow_pickle=False)
            for hi,h in enumerate(C['horizons_seconds']):
                ix=train[data.mask[train,hi]];v=val[data.mask[val,hi]]
                if not len(ix) or not len(v):continue
                squared=np.sum((data.y[: ,hi,:2]-raw[:,-1,:2])**2,1);threshold=float(np.quantile(np.sqrt(squared[ix]),C['displacement_threshold_quantile']));event=(np.sqrt(squared)>threshold).astype(float);w=weights(data,ix)
                bp,logit,recordp=probabilistic_fit(a[ix],event[ix],w,'probability',C['probability_regularization']);bs,logvar,records=probabilistic_fit(a[ix],squared[ix],w,'scale',C['scale_regularization'])
                np.savez(D/f'probability.{cohort}.{arm}.{hi}.npz',probability_weights=bp,scale_weights=bs,threshold=threshold,baseline_logit=logit,baseline_logvariance=logvar)
                optimization.append(dict(cohort=cohort,arm=arm,horizon_seconds=h,probability=recordp,scale=records,threshold=threshold,training_participants=sorted(set(data.participants[ix].tolist()))))
                for model in ['context_free','context']:
                    probability=np.full(len(v),expit(logit)) if model=='context_free' else expit(a[v]@bp)
                    lv=np.full(len(v),logvar) if model=='context_free' else logvar+3*np.tanh(a[v]@bs/3);variance=np.exp(lv);brier,ll=probability_scores(probability,event[v]);nll=np.log(2*np.pi)+lv+squared[v]/(2*variance)
                    for j,i in enumerate(v):
                        sample=data.samples[i];rows.append(dict(cohort=cohort,arm=arm,model=model,participant=sample['participant'],episode_id=sample['episode_id'],sequence_id=sample['sequence_id'],horizon_seconds=h,
                            event_threshold=threshold,source_displacement_event=bool(event[i]),probability=float(probability[j]),brier=float(brier[j]),log_loss=float(ll[j]),
                            target_squared_displacement=float(squared[i]),predicted_squared_displacement=float(2*variance[j]),scale_nll=float(nll[j]),radial_95_covered=bool(squared[i]/variance[j]<=5.991464547107979),physical_event=False))
    save(R/'drift_predictions.parquet',drift_rows);save(R/'probability_predictions.parquet',rows);atomic_json(R/'drift_selection.json',dict(records=selections,regularization=C['ridge_strength'],zero_candidate_retained=True))
    atomic_json(R/'probability_protocol_results.json',dict(optimization=optimization,threshold_quantile=C['displacement_threshold_quantile'],probability_regularization=C['probability_regularization'],scale_regularization=C['scale_regularization'],
        no_validation_selection=True,point_forecast_gate_unchanged=True,proper_scores=['Brier','Bernoulli log loss','isotropic Gaussian source-displacement NLL'],calibration_not_independent_physical_coverage=True))
