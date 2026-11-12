"""Dependency-aware reconstruction and bounded causal source-position filtering.

These estimators describe supplied 2D evidence. No physical identity, depth,
contact, or independent calibration is inferred. Shared group bias is explicit;
measurement_cov is the independent covariance *in addition* to that bias.
"""
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .memory import MemoryConfig, transition


def psd(matrix,*,positive=False):
    a=np.asarray(matrix,float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all():raise ValueError('Invalid covariance shape/values')
    scale=max(float(abs(a).max()),1e-12)
    if np.max(abs(a-a.T))>scale*1e-8:raise ValueError('Asymmetric covariance')
    a=(a+a.T)/2;e=np.linalg.eigvalsh(a)
    if e[0]<-scale*1e-9 or (positive and e[0]<=0):raise ValueError('Covariance is not positive (semi)definite')
    if e[0]<0:a=a+np.eye(len(a))*(-e[0]+np.finfo(float).eps*scale)
    return a


def _observations(times,positions,covariance,groups):
    t=np.asarray(times,float);y=np.asarray(positions,float);r=np.asarray(covariance,float);g=np.asarray(groups,dtype=str)
    if t.ndim!=1 or not len(t) or y.shape!=(len(t),2) or r.shape!=(len(t),2,2) or g.shape!=(len(t),):raise ValueError('Observation arrays must be N, N by 2, N by 2 by 2, N')
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<0):raise ValueError('Finite chronological observations required')
    r=np.stack([psd(a,positive=True) for a in r]);return t,y,r,g


def unique_observations(times,positions,covariance,groups,provenance_keys=None):
    t,y,r,g=_observations(times,positions,covariance,groups)
    keys=list(provenance_keys) if provenance_keys is not None else [(float(ti),str(gi),tuple(yi)) for ti,gi,yi in zip(t,g,y)]
    if len(keys)!=len(t):raise ValueError('Provenance key length mismatch')
    seen={};unique_at={};ids=[];inverse=[]
    for i,key in enumerate(keys):
        if key in seen:
            j=seen[key]
            if t[i]!=t[j] or g[i]!=g[j] or not np.array_equal(y[i],y[j]) or not np.array_equal(r[i],r[j]):raise ValueError('Conflicting reuse of a provenance key')
        else:seen[key]=i;unique_at[key]=len(ids);ids.append(i)
        inverse.append(unique_at[key])
    return np.array(ids,int),np.array(inverse,int)


def time_mass(times):
    """Unit time mass split equally among independent observations at one time."""
    t=np.asarray(times,float);unique,inv,counts=np.unique(t,return_inverse=True,return_counts=True)
    if len(unique)==1:return np.ones(len(t))/len(t)
    edges=np.r_[unique[0],(unique[:-1]+unique[1:])/2,unique[-1]];mass=np.diff(edges)/(unique[-1]-unique[0])
    return mass[inv]/counts[inv]


def structured_covariance(covariance,groups,shared_bias_covariance=None):
    r=np.asarray(covariance,float);g=np.asarray(groups,dtype=str);n=len(g)
    shared=np.zeros((2,2)) if shared_bias_covariance is None else psd(shared_bias_covariance)
    result=np.zeros((2*n,2*n))
    for i in range(n):
        for j in range(n):
            if g[i]==g[j]:result[2*i:2*i+2,2*j:2*j+2]=shared
        result[2*i:2*i+2,2*i:2*i+2]+=r[i]
    return psd(result,positive=True)


def fit_history(times_seconds,positions,measurement_cov,groups,*,end_time=None,
                shared_bias_covariance=None,provenance_keys=None,ancestor_max_time=None,
                mode='strict_causal',regularization=1e-4,endpoint_mean=None,
                endpoint_covariance=None,observation_endpoint_cross=None):
    """Time-mass GLS cubic with exact linear/sandwich covariance.

    A supplied endpoint [px,py,vx,vy] is pinned exactly. Its cross covariance
    with observations is required for a fully specified joint covariance; if
    absent, returned covariance is explicitly conditional/approximate.
    With no endpoint, all coefficients are fitted jointly. Equal timestamps
    are allowed for independent evidence, but duplicate provenance is removed.
    """
    t,y,r,g=_observations(times_seconds,positions,measurement_cov,groups)
    end=float(t[-1] if end_time is None else end_time)
    if not np.isfinite(end) or np.any(t>end):raise ValueError('History contains future observations')
    if mode not in ('strict_causal','retrospective'):raise ValueError('Unknown ancestry mode')
    ancestry=np.full(len(t),np.inf) if ancestor_max_time is None else np.asarray(ancestor_max_time,float)
    if ancestry.shape!=t.shape or np.any(np.isnan(ancestry)):raise ValueError('Invalid ancestry')
    if mode=='strict_causal' and np.any(ancestry>end):raise ValueError('Future or unknown ancestry is unavailable causally')
    ids,inverse=unique_observations(t,y,r,g,provenance_keys);t,y,r,g=t[ids],y[ids],r[ids],g[ids]
    if len(np.unique(t))<4:raise ValueError('Four distinct history times required')
    if not np.isfinite(regularization) or regularization<0:raise ValueError('Invalid regularization')
    sigma=structured_covariance(r,g,shared_bias_covariance);root=np.sqrt(np.repeat(time_mass(t),2));precision=cho_solve(cho_factor(sigma,lower=True),np.eye(len(sigma)));weight=root[:,None]*precision*root[None,:]
    u=t-end;phi=np.stack([np.ones(len(t)),u,u*u,u*u*u],axis=1);design=np.kron(phi,np.eye(2));z=y.ravel()
    if endpoint_mean is None:
        penalty=regularization*np.diag([0,0,0,0,1,1,9,9]);gram=design.T@weight@design+penalty
        factor=cho_factor(gram,lower=True);linear=cho_solve(factor,design.T@weight);coef=linear@z;cov=linear@sigma@linear.T;conditional=False
        joint_input=sigma
    else:
        endpoint=np.asarray(endpoint_mean,float)
        if endpoint.shape!=(4,) or not np.isfinite(endpoint).all() or endpoint_covariance is None:raise ValueError('Finite endpoint and covariance required')
        epcov=psd(endpoint_covariance)
        if epcov.shape!=(4,4):raise ValueError('Endpoint covariance shape')
        conditional=observation_endpoint_cross is None
        if conditional:cross=np.zeros((len(z),4))
        else:
            oldcross=np.asarray(observation_endpoint_cross,float)
            if oldcross.shape!=(2*len(inverse),4):raise ValueError('Observation/endpoint cross covariance shape')
            cross=oldcross.reshape(-1,2,4)[ids].reshape(-1,4)
        joint_input=psd(np.block([[sigma,cross],[cross.T,epcov]]))
        free=design[:,4:];fixed=design[:,:4];gram=free.T@weight@free+regularization*np.diag([1,1,9,9]);factor=cho_factor(gram,lower=True);b=cho_solve(factor,free.T@weight)
        linear=np.zeros((8,len(z)+4));linear[:4,-4:]=np.eye(4);linear[4:,:len(z)]=b;linear[4:,-4:]=-b@fixed;coef=linear@np.r_[z,endpoint];cov=linear@joint_input@linear.T
    return dict(coefficients=coef.reshape(4,2),coefficient_covariance=psd(cov),linear_map=linear,joint_input_covariance=joint_input,
                end_time=end,coefficient_basis='physical_seconds_powers_about_endpoint',unique_indices=ids,inverse=inverse,
                quadrature_mass=time_mass(t),conditional_covariance=conditional,ancestry_mode=mode,
                ancestry_verified=bool(np.all(ancestry<=end)),regularized_bias_in_covariance=False,
                physical_uncertainty_calibrated=False)


@dataclass(frozen=True)
class FilterConfig:
    acceleration_spectral_density: float = .01
    jump_spectral_density: float = 1.
    shared_bias_variance: float = 1e-4
    initial_velocity_variance: float = 1.
    maximum_gap: float = .35
    minimum_span: float = .04
    minimum_observations: int = 3
    maximum_bias_groups: int = 4
    student_df: float = 4.
    robust_measurement: bool = True
    process_mode: str = 'coherent_jump'
    jump_nis_threshold: float = 9.21
    event_speed_threshold: float = .05
    memory_timescales: tuple[float,...] = (.125,.5,2.)

    def __post_init__(self):
        for name in ['acceleration_spectral_density','jump_spectral_density','shared_bias_variance','initial_velocity_variance','minimum_span','event_speed_threshold']:
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError('Invalid '+name)
        if self.maximum_gap<=0 or not np.isfinite(self.maximum_gap) or self.student_df<=0 or not np.isfinite(self.student_df) or self.jump_nis_threshold<=0 or not np.isfinite(self.jump_nis_threshold):raise ValueError('Invalid filter threshold')
        if self.minimum_observations<1 or self.maximum_bias_groups<1:raise ValueError('Invalid support/bias bound')
        if self.process_mode not in ('gaussian','coherent_jump'):raise ValueError('Unknown process mode')
        MemoryConfig(self.memory_timescales)


def process(times_seconds,positions,measurement_cov,groups,config=None,*,provenance_keys=None,
            ancestor_max_time=None,mode='strict_causal',coherence=None,shared_covariance=None):
    """Causal bounded CV + persistent group-bias + joint exponential memories.

    coherent_jump requires caller-supplied independent multi-anchor agreement;
    it never derives coherence from one centroid. shared_covariance[N,2,2] uses
    principal PSD-root loadings on a persistent unit group-bias latent; thus
    cross-time shared covariance is L_i L_j.T within a group. The configured
    isotropic shared_bias_variance is added before constructing those roots.
    Unknown ancestry is invalid
    in strict mode. Retrospective mode is explicitly a source proxy.
    """
    cfg=FilterConfig() if config is None else config;t,y,r,g=_observations(times_seconds,positions,measurement_cov,groups);n=len(t)
    if mode not in ('strict_causal','retrospective'):raise ValueError('Unknown ancestry mode')
    ancestry=np.full(n,np.inf) if ancestor_max_time is None else np.asarray(ancestor_max_time,float)
    coherent=np.zeros(n,bool) if coherence is None else np.asarray(coherence,bool)
    if ancestry.shape!=t.shape or np.any(np.isnan(ancestry)) or coherent.shape!=t.shape:raise ValueError('Invalid ancestry/coherence')
    shared=np.zeros_like(r) if shared_covariance is None else np.asarray(shared_covariance,float)
    if shared.shape!=r.shape:raise ValueError('Shared covariance shape mismatch')
    loadings=[]
    for a in shared:
        a=psd(a)+np.eye(2)*cfg.shared_bias_variance;values,vectors=np.linalg.eigh(a);loadings.append((vectors*np.sqrt(np.maximum(values,0)))@vectors.T)
    loadings=np.asarray(loadings)
    ids,inverse=unique_observations(t,y,r,g,provenance_keys)
    for i,j in enumerate(ids[inverse]):
        if not np.array_equal(shared[i],shared[j]) or ancestry[i]!=ancestry[j] or coherent[i]!=coherent[j]:raise ValueError('Conflicting duplicate shared covariance, ancestry or coherence')
    unique=set(ids.tolist());base=4+2*len(cfg.memory_timescales);m=None;p=None;active=OrderedDict();evicted=set();last_time=None;first_time=None;count=0;last_evidence=np.nan
    out=dict(mean=np.zeros((n,4)),covariance=np.zeros((n,4,4)),memory_mean=np.zeros((n,len(cfg.memory_timescales),2)),joint_covariance=np.zeros((n,base,base)),valid=np.zeros(n,bool),accepted=np.zeros(n,bool),ancestry_verified=ancestry<=t,measurement_weight=np.zeros(n),innovation_nis=np.zeros(n),process_jump=np.zeros(n,bool),event=np.zeros(n,bool),latest_evidence_time=np.full(n,np.nan),support_failure=np.full(n,'',dtype='<U48'))
    def store(i):
        if m is not None:
            out['mean'][i]=m[:4];out['covariance'][i]=p[:4,:4];out['memory_mean'][i]=m[4:base].reshape(-1,2);out['joint_covariance'][i]=p[:base,:base]
        out['latest_evidence_time'][i]=last_evidence
        out['valid'][i]=m is not None and count>=cfg.minimum_observations and t[i]-first_time>=cfg.minimum_span and t[i]-last_evidence<=cfg.maximum_gap
    for i in range(n):
        if i not in unique:
            out['support_failure'][i]='duplicate_provenance';store(i);continue
        if mode=='strict_causal' and ancestry[i]>t[i]:
            out['support_failure'][i]='future_or_unknown_ancestry';store(i);out['valid'][i]=False;continue
        if last_time is not None and t[i]-last_evidence>cfg.maximum_gap:
            m=p=None;active=OrderedDict();evicted=set();count=0
        if g[i] in evicted:
            out['support_failure'][i]='correlation_group_evicted';store(i);out['valid'][i]=False;continue
        if m is None:
            m=np.zeros(base+2);m[:2]=y[i]
            for j in range(len(cfg.memory_timescales)):m[4+2*j:6+2*j]=y[i]
            loading=loadings[i];b=loading@loading.T;p=np.zeros((base+2,base+2));p[2:4,2:4]=np.eye(2)*cfg.initial_velocity_variance;p[base:,base:]=np.eye(2)
            slots=[slice(0,2)]+[slice(4+2*j,6+2*j) for j in range(len(cfg.memory_timescales))]
            for a in slots:
                for c in slots:p[a,c]=r[i]+b
                p[a,base:]=-loading;p[base:,a]=-loading.T
            active[g[i]]=base;first_time=t[i];last_time=t[i];count=1;last_evidence=t[i];out['accepted'][i]=True;out['measurement_weight'][i]=1.;store(i)
            if not out['valid'][i]:out['support_failure'][i]='insufficient_recent_support'
            continue
        dt=t[i]-last_time;f,q=transition(dt,cfg.memory_timescales,cfg.acceleration_spectral_density);full=np.eye(len(m));full[:base,:base]=f;prior_mean=full@m;prior_cov=full@p@full.T;prior_cov[:base,:base]+=q;m,p=prior_mean,prior_cov;last_time=t[i]
        if g[i] not in active:
            if len(active)>=cfg.maximum_bias_groups:
                old,offset=active.popitem(last=False);evicted.add(old);keep=np.r_[np.arange(offset),np.arange(offset+2,len(m))];m=m[keep];p=p[np.ix_(keep,keep)];active=OrderedDict((key,at-2 if at>offset else at) for key,at in active.items())
            offset=len(m);active[g[i]]=offset;m=np.r_[m,np.zeros(2)];new=np.zeros((len(m),len(m)));new[:-2,:-2]=p;new[-2:,-2:]=np.eye(2);p=new
        h=np.zeros((2,len(m)));h[:,:2]=np.eye(2);offset=active[g[i]];h[:,offset:offset+2]=loadings[i];residual=y[i]-h@m;innovation=psd(h@p@h.T+r[i],positive=True);nis=float(residual@np.linalg.solve(innovation,residual));out['innovation_nis'][i]=nis
        jump=bool(cfg.process_mode=='coherent_jump' and coherent[i] and nis>cfg.jump_nis_threshold and dt>0)
        if jump:
            _,extra=transition(dt,cfg.memory_timescales,max(cfg.jump_spectral_density-cfg.acceleration_spectral_density,0.));p[:base,:base]+=extra
        weight=1. if not cfg.robust_measurement or jump else min(1.,(cfg.student_df+2)/(cfg.student_df+nis));effective=r[i]/max(weight,1e-6);innovation=psd(h@p@h.T+effective,positive=True);gain=np.linalg.solve(innovation,h@p).T;m=m+gain@residual;identity=np.eye(len(m))-gain@h;p=psd(identity@p@identity.T+gain@effective@gain.T)
        count+=1;last_evidence=t[i];out['accepted'][i]=True;out['measurement_weight'][i]=weight;out['process_jump'][i]=jump;out['event'][i]=bool(jump and np.linalg.norm(m[2:4])>=cfg.event_speed_threshold);store(i)
        if not out['valid'][i]:out['support_failure'][i]='insufficient_recent_support'
    out.update(ancestry_mode=mode,physical_uncertainty_calibrated=False,covariance_scope='conditional robust Gaussian approximation; group bias explicit; identity and camera errors not supplied',state_order=['px','py','vx','vy'],memory_timescales=cfg.memory_timescales)
    return out
