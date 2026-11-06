"""Frozen-model interpretation checks; no validation-driven model adjustment."""
import json
import numpy as np
import torch
from pipeline.common import ROOT,atomic_json
from track2_feasibility.data import Data,DATA,CFG
from track2_feasibility.train import load_model,tensors,seed_all
from .data import R,C,D
from .math_checks import transport_matrix,polygon_moments


def diagnostics():
    data=Data();ids=data.indices('validation',usable=False);ids=ids[np.load(D/'input_mask.npy')[ids]]
    current=data.x[:,-1,:140].reshape(-1,4,35).astype(float);target=data.y[:,:,4:].reshape(-1,3,4,35).astype(float)
    train=data.indices();scales=np.maximum(current[train].std(axis=(0,1)),.001);memory=[]
    for hi,h in enumerate(np.array(C['horizons_us'])/1e6):
        reindexed=np.einsum('ij,njk->nik',transport_matrix(2*h),current)
        for name,pred in [('copy',current),('polynomial_reindex',reindexed)]:
            scores=[]
            for participant in sorted(set(data.participants[ids])):
                ix=ids[(data.participants[ids]==participant)&data.mask[ids,hi]]
                scores.append(dict(participant=participant,queries=len(ix),coefficient_nrmse=float(np.sqrt(np.mean(((pred[ix]-target[ix,hi])/scales)**2)))))
            memory.append(dict(method=name,horizon_seconds=h,per_participant=scores))
    atomic_json(R/'memory_diagnostic.json',dict(records=memory,
        semantics='Exact extension/reindexing of the old polynomial. Not the new irregular weighted fit, an innovation model, or certified future geometry.',learned_innovation_trained=False))
    checks=[];lookup={(s['video_id'],s['entity_track_id'],s['episode_id'],s['epoch'],s['input_times_us'][-1]):i for i,s in enumerate(data.samples)}
    for family in ('koopman','stable_koopman'):
        for seed in C['seeds']:
            seed_all(seed);model,norm,_=load_model(DATA/'models'/f'{family}.{seed}.pt');x,_,_=tensors(data,norm);scale=np.array(norm['y_scale']);mean=np.array(norm['y_mean'])
            with torch.no_grad():
                z=model.encoder(x.flatten(1));decoded=model.decoder(z[ids]).numpy()*scale+mean
                reference=np.c_[data.x[ids,-1,140:144],data.x[ids,-1,:140]]
                a=model.generator();latent=[];consistency=[]
                for hi,h in enumerate(np.array(C['horizons_us'])/1e6):
                    ix=ids[data.mask[ids,hi]];e=.001
                    def decode(t):return model.decoder(z[ix]@torch.matrix_exp(a*float(t)).T).numpy()*scale+mean
                    mid=decode(h);derivative=(decode(h+e)[:,:2]-decode(h-e)[:,:2])/(2*e)
                    consistency.append(dict(horizon_seconds=h,queries=len(ix),rms_derivative_minus_velocity=float(np.sqrt(np.mean((derivative-mid[:,2:4])**2)))))
                    before=[];after=[]
                    for i in ix:
                        s=data.samples[i];key=(s['video_id'],s['entity_track_id'],s['episode_id'],s['epoch'],s['input_times_us'][-1]+int(round(h*1e6)))
                        if key in lookup:before.append(i);after.append(lookup[key])
                    if before:
                        predicted=z[before]@torch.matrix_exp(a*float(h)).T;actual=z[after]
                        latent.append(dict(horizon_seconds=h,queries=len(before),latent_mse=float((predicted-actual).square().mean()),reference_latent_variance=float(actual.var(dim=0,unbiased=False).mean())))
                record=dict(family=family,seed=seed,zero_horizon_position_rmse=float(np.sqrt(np.mean(np.sum((decoded[:,:2]-reference[:,:2])**2,axis=1)))),
                    zero_horizon_standardized_state_rmse=float(np.sqrt(np.mean(((decoded-reference)/scale)**2))),latent_consistency=latent,derivative_consistency=consistency)
                if family=='stable_koopman':
                    b=model.B.detach().numpy().astype(float);h=b@b.T+CFG['generator_epsilon']*np.eye(len(b));aa=a.numpy().astype(float)
                    record.update(metric_condition_number=float(np.linalg.cond(h)),dissipation_largest_eigenvalue=float(np.linalg.eigvalsh(aa.T@h+h@aa).max()),
                        propagator_norms=[float(np.linalg.norm(torch.matrix_exp(a*float(t)).numpy(),2)) for t in [.25,.5,1.]])
                checks.append(record)
    atomic_json(R/'koopman_diagnostic.json',dict(records=checks,frozen_models_unchanged=True,latent_future_encoding_is_diagnostic_target_only=True,
        caveat='Source-derived states, not independent physical dynamics. These checks do not retrain the encoder or establish Koopman invariance.'))
    ring=np.array([[0.,0.],[4.,0.],[4.,2.],[0.,2.]])
    resampled=np.vstack([ring[0]+u*(ring[1]-ring[0]) for u in np.linspace(0,1,100,endpoint=False)]+[ring[1],ring[2],ring[3]])
    old=[np.cov(p.T,bias=True).tolist() for p in [ring,resampled]];new=[polygon_moments([p],['outer'])[2].tolist() for p in [ring,resampled]]
    atomic_json(R/'region_moments.json',dict(boundary_point_covariances=old,region_covariances=new,
        invariant_max_error=float(np.max(np.abs(np.array(new[0])-new[1]))),new_feature_version='polygon_region_moments_v1',frozen_geometry_features_unchanged=True,
        caveat='Integrates the supplied polygon region with hole subtraction; does not validate topology or create independent boundaries.'))
