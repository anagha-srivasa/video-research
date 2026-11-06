"""Identical-width linear probes, including missingness and past-time shuffle controls."""
import numpy as np
from pipeline.common import atomic_json
from track1_retrain.common import save
from track2_feasibility.data import Data
from track2_feasibility.train import participant_weights
from .common import C,D,R,OLD


def features(raw,arm):
    x=raw.copy()
    if arm=='object':x[:,:,2:]=0
    elif arm=='reliability':x[:,:,5:]=0
    elif arm=='missingness':x[:,:,5:9]=0
    elif arm=='shuffled':
        rng=np.random.default_rng(C['relation_shuffle_seed'])
        for i in range(len(x)):
            # Negative control permutes only the four hand samples already available at the query cutoff.
            order=rng.permutation(4);x[i,:,5:9]=raw[i,order,5:9];x[i,:,9:]=raw[i,order,9:]
    elif arm!='relation':raise ValueError('Unknown information arm')
    return x.reshape(len(x),-1)


def fit(data,x,raw,ids,ridge):
    scale=np.maximum(np.sqrt(np.mean(x[ids]**2,0)),.001);a=np.c_[x/scale,np.ones(len(x))];weights=[]
    for hi in range(3):
        ix=ids[data.mask[ids,hi]];w=participant_weights(data,ix).astype(float);w/=w.sum();design=a[ix];target=data.y[ix,hi,:2]-raw[ix,-1,:2]
        penalty=np.eye(a.shape[1])*ridge;penalty[-1,-1]=0
        weights.append(np.linalg.solve(design.T@(w[:,None]*design)+penalty,design.T@(w[:,None]*target)))
    return scale,np.array(weights)


def predict(x,raw,scale,weights):return raw[:,-1:, :2]+np.einsum('nd,hdk->nhk',np.c_[x/scale,np.ones(len(x))],weights)


def run():
    data=Data();information=np.load(D/'information.npy');raw=np.load(OLD/'recent_observation.npy');ok=np.load(OLD/'input_mask.npy');ids=data.indices();ids=ids[ok[ids]];rows=[];selection=[]
    for arm in C['relation_arms']:
        x=features(information,arm);folds=[]
        for held in C['training_participants']:
            train=ids[data.participants[ids]!=held];stop=ids[data.participants[ids]==held]
            for ridge in [None]+C['ridge_grid']:
                pred=np.repeat(raw[:,-1:,:2],3,1) if ridge is None else predict(x,raw,*fit(data,x,raw,train,ridge));scores=[]
                for hi in range(3):
                    ix=stop[data.mask[stop,hi]];scores.append(float(np.sqrt(np.mean(np.sum((pred[ix,hi]-data.y[ix,hi,:2])**2,1)))))
                folds.append(dict(held=held,ridge=ridge,score=float(np.mean(scores))))
        selected=min([None]+C['ridge_grid'],key=lambda v:np.mean([r['score'] for r in folds if r['ridge']==v]));selection.append(dict(arm=arm,folds=folds,selected_ridge=selected))
        scale,weights=fit(data,x,raw,ids,selected) if selected is not None else (np.ones(x.shape[1]),np.zeros((3,x.shape[1]+1,2)))
        np.save(D/f'relation.{arm}.scale.npy',scale,allow_pickle=False);np.save(D/f'relation.{arm}.weights.npy',weights,allow_pickle=False);pred=predict(x,raw,scale,weights)
        val=data.indices('validation',usable=False);val=val[ok[val]]
        for i in val:
            for hi,h in enumerate(C['horizons_seconds']):
                s=data.samples[i];valid=bool(data.mask[i,hi]);rows.append(dict(family=arm,sequence_id=s['sequence_id'],participant=s['participant'],episode_id=s['episode_id'],horizon_seconds=h,target_supported=valid,
                    hand_history_complete=bool((information[i,:,9]>0).all()),hand_history_any=bool((information[i,:,9]>0).any()),
                    predicted_position=pred[i,hi].tolist(),position_squared_error=float(np.sum((pred[i,hi]-data.y[i,hi,:2])**2)) if valid else None))
    save(R/'relation_predictions.parquet',rows);atomic_json(R/'relation_selection.json',dict(records=selection,train_only=True,zero_correction_eligible=True,
        controls='Same 44 input slots and 270 regression coefficients in all arms. Unused slots zero. Shuffle is within-query past history; presence flags shuffle with hand values.',
        caveat='Source annotation positions/identities, not independently verified interaction or physical contact. Object may itself be a hand; report all-entity cohort explicitly.'))
