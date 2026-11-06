"""Probability-specific controls using the real ablation and probe functions."""
import numpy as np
from .core import fit,scores,arms


def run():
    rows=[]
    for noise in ['iid','correlated','smoothed']:
        for signal in ['stationary','conditional_variance']:
            rng=np.random.default_rng(1701);n=12000;e=rng.normal(size=(n,20,2))
            if noise=='correlated':
                for t in range(1,20):e[:,t]=.7*e[:,t-1]+np.sqrt(1-.7**2)*e[:,t]
            if noise=='smoothed':e=(e[:,1:]+e[:,:-1])/2
            regime=rng.integers(0,2,n);sigma=.2+3*regime
            latent=np.cumsum(rng.normal(size=e.shape)*sigma[:,None,None],axis=1) if signal!='stationary' else np.zeros_like(e)
            observed=latent+e;cutoffs=[2,5,8,11];anchor=observed[:,11];raw=np.zeros((n,4,21))
            # Regular 50ms observation times; each short OLS slope sees only its three retained measurements.
            for j,t in enumerate(cutoffs):
                raw[:,j,:2]=observed[:,t]-anchor
                raw[:,j,2:4]=(observed[:,t]-observed[:,t-2])/.1
                raw[:,j,5:8]=[3,1,1]
                window=observed[:,t-2:t+1];line=(window[:,0]+window[:,2])/2
                raw[:,j,20]=np.sqrt(np.mean((window[:,1]-line)**2,axis=1))
            x=arms(raw,np.zeros((n,4,4,2)),anchor)['both'];participants=np.array(['train']*8000+['test']*4000)
            recent=observed[:,11]-observed[:,10];energy=np.sum(recent**2,1)
            for anchor_kind in ['shared','independent']:
                # Independent localization error at the SAME latent current position.
                origin=anchor if anchor_kind=='shared' else latent[:,11]+np.random.default_rng(701).permutation(e[:,11])
                delta=observed[:,17]-origin;sq=np.sum(delta**2,1);f=fit(x,sq,participants,np.arange(8000));ix=np.arange(8000,n)
                rows.append(dict(noise=noise,signal=signal,anchor=anchor_kind,squared_correlation=float(np.corrcoef(energy,sq)[0,1]),baseline=scores(np.full(len(ix),f['frequency']),f['event'][ix]),context=scores(f['probability'][ix],f['event'][ix])))
    return dict(scope='Synthetic regular-time noisy positions passed through shared position/motion/metadata feature and probability fitting functions. Not the full irregular repository mask/kinematic target pipeline or physical dataset evidence.',records=rows)
