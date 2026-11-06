"""Same coefficients, three position anchors; shared-measurement covariance stays explicit."""
import numpy as np
from pipeline.common import atomic_json
from track2_feasibility.data import Data
from track2_observability.memory import overlap
from .common import C,OLD,R


def update(c,anchor,h):
    f,b=overlap(h);return np.einsum('ij,njk->nik',f,c)+b[None,:,None]*anchor[:,None,:]


def joint_covariance(f,b,cov,anchor_cov,cross):
    # cross[i] = Cov(c_i, anchor). Caller must supply correlated cross terms.
    return f@cov@f.T+np.outer(b,b)*anchor_cov+np.outer(f@cross,b)+np.outer(b,f@cross)


def run():
    data=Data();c=data.x[:,-1,:140].reshape(-1,4,35).astype(float)[:,:,:2];target=data.y[:,:,4:].reshape(-1,3,4,35).astype(float)[:,:,:,:2];raw=np.load(OLD/'recent_observation.npy');ok=np.load(OLD/'input_mask.npy');val=data.indices('validation',usable=False);val=val[ok[val]]
    anchors={'functional_endpoint':c.sum(1),'latest_observation':raw[:,-1,:2],'causal_position_estimate':data.x[:,-1,140:142].astype(float)};records=[];operators=[];metric=np.sqrt(2/(2*np.arange(4)+1))
    for hi,h in enumerate(C['horizons_seconds']):
        f,b=overlap(h);m=f+np.outer(b,np.ones(4));operators.append(dict(horizon_seconds=h,retention_norm=float(np.linalg.norm(metric[:,None]*f/metric[None,:],2)),complete_endpoint_fill_norm=float(np.linalg.norm(metric[:,None]*m/metric[None,:],2))))
        for name,anchor in anchors.items():
            pred=update(c,anchor,h).sum(1)
            for participant in C['validation_participants']:
                ix=val[(data.participants[val]==participant)&data.mask[val,hi]];error=pred[ix]-target[ix,hi].sum(1)
                records.append(dict(anchor=name,horizon_seconds=h,participant=participant,queries=len(ix),functional_endpoint_rmse=float(np.sqrt(np.mean(np.sum(error**2,1)))),mean_latest_observation_age_seconds=float(raw[ix,-1,2].mean())))
    # Linear-Gaussian synthetic check with correlated endpoint exactly computed from coefficients.
    cov=np.diag([.1,.2,.3,.4]);cross=cov@np.ones(4);variance=float(np.ones(4)@cross);checks=[]
    for h in C['horizons_seconds']:
        f,b=overlap(h);m=f+np.outer(b,np.ones(4));propagated=joint_covariance(f,b,cov,variance,cross)
        checks.append(dict(horizon_seconds=h,correlated_covariance_identity_error=float(np.max(abs(propagated-m@cov@m.T)))))
    atomic_json(R/'memory_anchors.json',dict(operators=operators,records=records,synthetic_covariance_checks=checks,
        units='All anchors and raw position coefficients are intrinsic-normalized 2D; geometry channels untouched.',covariance='No independent-anchor assumption. Real joint coefficient-anchor cross covariance unavailable, so no real calibrated propagated covariance is asserted.',
        scope='Uniform projection diagnostic on irregular fitted coefficients. Functional endpoints differ from cached kinematic prediction targets.'))
