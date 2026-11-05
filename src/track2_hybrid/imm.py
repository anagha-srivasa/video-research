"""Stationary/CV IMM: row-stochastic Markov transitions, Gaussian interaction.

State order is x,y,vx,vy. The covariance is a conditional moment approximation,
not a calibrated physical-motion distribution. See Dingler, arXiv:2207.04875.
"""
import copy
import numpy as np
from track1_lab.state_space import checked_covariance, transition as cv_transition


def covariance(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
        raise ValueError('Invalid covariance')
    if np.max(abs(value-value.T)) > 1e-8*max(float(abs(value).max()), 1e-15):
        raise ValueError('Asymmetric covariance')
    return checked_covariance(value)


def transition_probability(dt, rate):
    if not np.isfinite([dt,rate]).all() or dt < 0 or rate < 0:
        raise ValueError('Invalid Markov interval/rate')
    stay = .5*(1+np.exp(-2*rate*dt))
    return np.array([[stay,1-stay],[1-stay,stay]])  # row=old mode, column=new mode


def moments(means, covariances, weights):
    means, covariances, weights = map(lambda x: np.asarray(x,dtype=np.float64), (means,covariances,weights))
    if means.shape != (2,4) or covariances.shape != (2,4,4) or weights.shape != (2,):
        raise ValueError('Two compatible four-dimensional modes required')
    if not all(np.isfinite(x).all() for x in (means,covariances,weights)) or np.any(weights<0) or not np.isclose(weights.sum(),1.):
        raise ValueError('Invalid mixture')
    mean = weights@means
    delta = means-mean
    cov = sum(w*(covariance(p)+np.outer(d,d)) for w,p,d in zip(weights,covariances,delta))
    return mean, covariance(cov)


def interaction(means, covariances, weights, markov):
    markov = np.asarray(markov,float)
    if markov.shape != (2,2) or not np.isfinite(markov).all() or np.any(markov<0) or not np.allclose(markov.sum(1),1.):
        raise ValueError('Rows of Markov matrix must sum to one')
    moments(means,covariances,weights)
    prior = np.asarray(weights)@markov
    mixed_x, mixed_p = [], []
    for j in range(2):
        if prior[j] == 0:
            # An impossible mode has no effect on the posterior mixture.
            x,p = means[j].copy(),covariances[j].copy()
        else:
            mixing = np.asarray(weights)*markov[:,j]/prior[j]
            x,p = moments(means,covariances,mixing)
        mixed_x.append(x); mixed_p.append(p)
    return np.array(mixed_x),np.array(mixed_p),prior


def initial(position, variance, parameters, mode='imm'):
    p,r = np.asarray(position,float),np.asarray(variance,float)
    if p.shape!=(2,) or r.shape!=(2,) or not np.isfinite([p,r]).all() or np.any(r<0):
        raise ValueError('Invalid initial measurement')
    q,scale,rate = (parameters[k] for k in ['process_density','measurement_multiplier','switch_rate'])
    if not np.isfinite([q,scale,rate]).all() or q<=0 or scale<=0 or rate<0:
        raise ValueError('Invalid filter parameters')
    if mode not in ('imm','kalman_cv','kalman_stationary'):
        raise ValueError('Unknown motion mode')
    state = np.r_[p,0.,0.]
    cov = np.diag(np.r_[np.maximum(r*scale,1e-10),[1.,1.]])
    weights = np.array([.5,.5]) if mode=='imm' else np.array([float(mode=='kalman_stationary'),float(mode=='kalman_cv')])
    return dict(means=np.array([state,state]),covariances=np.array([cov,cov]),weights=weights,
                parameters=dict(parameters),mode=mode)


def propagate(state, dt):
    if not np.isfinite(dt) or dt<0:
        raise ValueError('Noncausal propagation')
    if dt == 0:
        return copy.deepcopy(state)
    par = state['parameters']
    x,p,mu = state['means'],state['covariances'],state['weights']
    if state['mode']=='imm':
        x,p,mu = interaction(x,p,mu,transition_probability(dt,par['switch_rate']))
    f_cv,q_cv = cv_transition(dt,par['process_density'])
    f_stop = np.diag([1.,1.,0.,0.])
    q_stop = np.diag([par['process_density']*.001*dt]*2+[0.,0.])
    means = np.array([f_stop@x[0],f_cv@x[1]])
    covs = np.array([covariance(f_stop@p[0]@f_stop.T+q_stop),covariance(f_cv@p[1]@f_cv.T+q_cv)])
    return dict(state,means=means,covariances=covs,weights=mu.copy())


def update(state, dt, position, variance):
    value = propagate(state,dt)
    point,var = np.asarray(position,float),np.asarray(variance,float)
    if point.shape!=(2,) or var.shape!=(2,) or not np.isfinite([point,var]).all() or np.any(var<0):
        raise ValueError('Invalid measurement')
    r = np.diag(np.maximum(var*state['parameters']['measurement_multiplier'],1e-10))
    h = np.c_[np.eye(2),np.zeros((2,2))]
    likelihood = []
    for i in range(2):
        x,p = value['means'][i],value['covariances'][i]
        error = point-x[:2]
        innovation = p[:2,:2]+r
        chol = np.linalg.cholesky(innovation)
        whitened = np.linalg.solve(chol,error)
        likelihood.append(-.5*(whitened@whitened+2*np.log(np.diag(chol)).sum()+2*np.log(2*np.pi)))
        gain = np.linalg.solve(innovation,h@p).T
        value['means'][i] = x+gain@error
        a = np.eye(4)-gain@h
        value['covariances'][i] = covariance(a@p@a.T+gain@r@gain.T)  # Joseph update
    prior = value['weights']
    logs = np.full(2,-np.inf)
    present = prior>0
    logs[present] = np.log(prior[present])+np.asarray(likelihood)[present]
    if not np.isfinite(logs[present]).all():
        raise FloatingPointError('Nonfinite measurement likelihood')
    weights = np.exp(logs-np.max(logs))
    value['weights'] = weights/weights.sum()
    return value


def filter_history(times, positions, variances, cutoff, parameters, mode='imm'):
    times,positions,variances = (np.asarray(x,float) for x in (times,positions,variances))
    if positions.shape!=(len(times),2) or variances.shape!=positions.shape or not np.isfinite(cutoff):
        raise ValueError('Invalid history shape/cutoff')
    keep = (times<=cutoff)&(times>=cutoff-1.-1e-9)
    t,p,v = times[keep],positions[keep],variances[keep]
    if len(t)<3 or not all(np.isfinite(x).all() for x in (t,p,v)) or np.any(v<0) or np.any(np.diff(t)<=0):
        raise ValueError('Insufficient/invalid retained observations')
    if t[-1]-t[0]<.04-1e-9 or cutoff-t[-1]>.1+1e-9 or np.diff(t).max()>.35+1e-9:
        raise ValueError('Unsupported observation span/age/gap')
    state = initial(p[0],v[0],parameters,mode)
    for i in range(1,len(t)):
        state = update(state,float(t[i]-t[i-1]),p[i],v[i])
    state = propagate(state,float(cutoff-t[-1]))
    state.update(latest_input_time=float(t[-1]),cutoff=float(cutoff),observation_count=len(t))
    return state


def forecast(state, horizon):
    if not np.isfinite(horizon) or not 0<=horizon<=1.000000001:
        raise ValueError('Forecast horizon outside bounded experiment')
    value = copy.deepcopy(state)
    remaining = float(horizon)
    while remaining>1e-12:
        step = min(remaining,.25)
        value = propagate(value,step)
        remaining -= step
    mean,cov = moments(value['means'],value['covariances'],value['weights'])
    return mean,cov,value['weights']
