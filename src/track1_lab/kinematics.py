"""Exact synthetic motion references for causal cached-state stress tests.

No real-video quality or independent human-reference claim is made here.
"""
from collections import defaultdict
import copy,json,os
import numpy as np
from pipeline.common import ROOT
from track1.artifacts import save_index
from track1.numerics import basis
from .common import path,pool,report,identity
from .temporal import fit
from .architecture import polynomial_fit

KINDS=['stationary','linear','pickup','burst','reversal']

def truth(t,kind,onset=2.,duration=.25):
 t=np.asarray(t);u=(t-onset)/duration;p=np.zeros_like(t);v=np.zeros_like(t)
 if kind=='linear':p=180*(t-1);v[:]=180
 elif kind=='pickup':
  z=np.clip(u,0,1);p=100*(10*z**3-15*z**4+6*z**5);v=100/duration*(30*z**2-60*z**3+30*z**4)
 elif kind=='burst':
  active=(u>0)&(u<1);p[active]=80*np.sin(np.pi*u[active])**2;v[active]=80*np.pi/duration*np.sin(2*np.pi*u[active])
 elif kind=='reversal':
  z=(t-onset)/(duration/3);p=180*duration/3*(np.logaddexp(z,-z)-np.log(2));v=180*np.tanh(z)
 return np.c_[960+p,540+.3*p],np.c_[v,.3*v]

def episode(seed,include_residual=False,include_kalman=False):
 rng=np.random.default_rng(20261029+seed);kind=KINDS[seed%5];duration=[.1,.25,.5][seed//5%3];noise=[.5,2.,5.][seed//15%3];corruption=['clean','outlier','missing'][seed//3%3];onset=2.+rng.uniform(-.15,.15);clock=np.arange(0,4.201,.02);positions,_=truth(clock,kind,onset,duration);measured=positions+rng.normal(0,noise,positions.shape)
 if corruption=='outlier':
  index=np.argmin(abs(clock-(onset+.04)));measured[index]+=np.array([80.,-55.])
 retained=np.ones(len(clock),bool)
 if corruption=='missing':retained&=~((clock>=onset-.04)&(clock<=onset+.11))
 clock=clock[retained];measured=measured[retained];h=np.zeros((len(clock),35));h[:,:2]=(measured-[960,540])/1000;variance=np.full_like(h,1e-6);variance[:,:2]=(noise/1000)**2
 baseline=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());working=json.loads((path('derived')/'calibration/model.json').read_text());scale=np.asarray(working['feature_scale']);base=baseline['temporal']['options'];hybrid=working['options'];robust=copy.deepcopy(hybrid);robust['slope_robust_delta']=2.
 methods={'v2':(1.,3,base),'slope_hybrid':(1.,3,hybrid),'robust_slope':(1.,3,robust),'half_second':(.5,3,None),'slope_linear_tail':(1.,3,hybrid),'half_second_linear_tail':(.5,3,None),'quarter_second':(.25,3,None),'degree5':(1.,5,None),'constant_velocity':(.2,1,None),'constant_position':(.2,0,None)};query=np.arange(1.25,3.801,.02);target,velocity=truth(query,kind,onset,duration);true_speed=np.linalg.norm(velocity,axis=1);rows=[];traces=[]
 if include_residual:
  methods['causal_residual']=(1.,3,base);residual_parameters=json.loads((path('reports')/'residual_selection.json').read_text())['parameters']
 if include_kalman:
  methods.update(kalman_cv=(1.,1,None),kalman_cv_robust=(1.,1,None))
 for hop in [.25,.125]:
  filtered={}
  if include_kalman:
   from .state_space import filter_states
   for method,robust_mode in [('kalman_cv',False),('kalman_cv_robust',True)]:filtered[method]={round(float(cutoff),9):s for cutoff,s in zip(np.arange(1.,4.001,hop),filter_states(clock,h[:,:2],variance[:,:2],np.arange(1.,4.001,hop),robust=robust_mode))}
  for name,(window,degree,options) in methods.items():
   states=[];attempted=0
   for cutoff in np.arange(1.,4.001,hop):
    attempted+=1;keep=(clock>=cutoff-window-1e-9)&(clock<=cutoff+1e-9);tt=clock[keep];hh=h[keep];vv=variance[keep]
    if len(tt)<(3 if degree<2 else 8) or tt[-1]-tt[0]<(.08 if degree<2 else .6*window):continue
    if len(tt)>1 and np.diff(tt).max()>.175:continue
    if name in filtered:
     state=filtered[name][round(float(cutoff),9)]
     if not state['valid']:continue
     c=np.stack([state['state'][:2],state['state'][2:]])
    elif name=='constant_position':c=hh[-1:,:2]
    elif name=='constant_velocity':c=np.linalg.lstsq(np.c_[np.ones(len(tt)),tt-cutoff],hh[:,:2],rcond=None)[0]
    elif name=='causal_residual':
     from .residual import fit_residual
     c=fit_residual(tt-cutoff+1.,hh,vv,np.ones(len(tt)),np.ones(len(tt)),base,np.asarray(baseline['feature_scale']),**residual_parameters)
    else:
     relative=tt-cutoff+1.;prior=np.ones(len(tt));quality=np.ones(len(tt))
     fit_scale=np.asarray(baseline['feature_scale']) if name=='v2' else scale
     result=fit(relative,hh,vv,prior,quality,options,fit_scale) if options is not None else polynomial_fit(relative,hh,vv,prior,quality,window,degree,3. if window==.5 else 1/3,scale);c=result['coefficients'][:,:2]
    states.append((cutoff,c))
   predicted=[];predicted_velocity=[];ages=[];times=[];target_indices=[]
   for j,t in enumerate(query):
    past=[s for s in states if s[0]<=t+1e-9]
    if not past:continue
    cutoff,c=past[-1];age=t-cutoff
    if name=='constant_position':p=c[0];v=np.zeros(2)
    elif name=='constant_velocity' or name in filtered:p=c[0]+age*c[1];v=c[1]
    elif name=='causal_residual':
     from .residual import evaluate
     p=evaluate(c,np.asarray([1.+age]))[0,:2];v=evaluate(c,np.asarray([1.+age]),1)[0,:2]
    elif name.endswith('linear_tail'):
     v=2/window*basis(1.,1,degree=degree)@c;p=basis(1.,degree=degree)@c+age*v
    else:p=basis(1+2*age/window,degree=degree)@c;v=2/window*basis(1+2*age/window,1,degree=degree)@c
    predicted.append(1000*p+[960,540]);predicted_velocity.append(1000*v);ages.append(age);times.append(t);target_indices.append(j)
   p=np.asarray(predicted);v=np.asarray(predicted_velocity);times=np.asarray(times);gt=target[target_indices];gv=velocity[target_indices];speed=np.linalg.norm(v,axis=1);gs=true_speed[target_indices];active=speed>100;starts=times[active&~np.r_[False,active[:-1]]];true_active=gs>100;true_starts=times[true_active&~np.r_[False,true_active[:-1]]];event=kind in ['pickup','burst'] and len(true_starts)>0;matches=starts[abs(starts-true_starts[0])<=.25] if event else np.array([]);quiet=gs<1.
   rows.append(dict(seed=seed,split='synthetic_train' if seed<60 else 'synthetic_validation',kind=kind,duration_s=duration,noise_px=noise,corruption=corruption,method=name,hop_seconds=hop,samples=len(p),update_attempts=attempted,updates=len(states),update_yield=len(states)/attempted,maximum_state_age_s=max(ages),position_mse_px=float(np.mean(np.sum((p-gt)**2,axis=1))),velocity_mse_px_s=float(np.mean(np.sum((v-gv)**2,axis=1))),stationary_velocity_mse=float(np.mean(np.sum(v[quiet]**2,axis=1))) if quiet.any() else None,event_eligible=event,event_recalled=bool(len(matches)) if event else None,event_delay_s=float(matches[np.argmin(abs(matches-true_starts[0]))]-true_starts[0]) if len(matches) else None,false_alarm_fraction=float(np.mean(active[quiet])) if quiet.any() else None,peak_speed_ratio=float(speed.max()/gs.max()) if gs.max()>0 else None))
   if 60<=seed<65:
    traces.extend(dict(seed=seed,kind=kind,method=name,hop_seconds=hop,time_s=float(t),position_x=float(pp[0]),true_position_x=float(gg[0]),velocity_x=float(vv[0]),true_velocity_x=float(gvv[0])) for t,pp,gg,vv,gvv in zip(times,p,gt,v,gv))
 return rows,traces

def summarize(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['split'],r['hop_seconds'],r['method'])].append(r)
 out=[]
 for (part,hop,name),rs in sorted(groups.items()):
  events=[r for r in rs if r['event_eligible']];quiet=[r for r in rs if r['stationary_velocity_mse'] is not None];delays=[r['event_delay_s'] for r in events if r['event_delay_s'] is not None]
  out.append(dict(split=part,hop_seconds=hop,method=name,episodes=len(rs),position_rmse_px=float(np.sqrt(np.mean([r['position_mse_px'] for r in rs]))),velocity_rmse_px_s=float(np.sqrt(np.mean([r['velocity_mse_px_s'] for r in rs]))),stationary_velocity_rmse_px_s=float(np.sqrt(np.mean([r['stationary_velocity_mse'] for r in quiet]))) if quiet else None,event_count=len(events),event_recall=float(np.mean([r['event_recalled'] for r in events])) if events else None,median_signed_event_delay_s=float(np.median(delays)) if delays else None,false_alarm_fraction=float(np.mean([r['false_alarm_fraction'] for r in quiet])) if quiet else None,update_yield=float(np.mean([r['update_yield'] for r in rs]))))
 return out

def run(monitor,include_residual=False,include_kalman=False):
 from functools import partial
 prefix='kinematics_kalman' if include_kalman else 'kinematics_residual' if include_residual else 'kinematics'
 results=pool(partial(episode,include_residual=include_residual,include_kalman=include_kalman),list(range(90)),monitor,workers=min(16,os.cpu_count() or 1));rows=[r for rr,_ in results for r in rr];traces=[r for _,tt in results for r in tt];save_index(path('reports')/(prefix+'.parquet'),rows);save_index(path('reports')/(prefix+'_traces.parquet'),traces)
 groups={key:{str(value):summarize([r for r in rows if r[key]==value]) for value in sorted(set(r[key] for r in rows))} for key in ['kind','noise_px','corruption']}
 report(prefix+'.json',dict(state='experimental_not_finalized',reference='Exact analytic synthetic 2D motion; no real-video or human-reference certification.',episodes=90,summary=summarize(rows),breakdown=groups,protocol=dict(frame_hz=50,output_hops_s=[.25,.125],query_hz=50,causal_cached_states=True,speed_threshold_px_s=100,event_match_tolerance_s=.25,event_categories=['pickup','burst'],stationary_threshold_px_s=1,synthetic_split='Seeds 0..59 and 60..89; no fitting or model selection uses synthetic validation. This is a deterministic stress cohort, not a representative population split.',corruptions=['Gaussian noise','one displaced annotation','150 ms missing interval'],uncertainty_evaluated=False),scope='All real model options/scales are reused unchanged. Robust local slope delta=2 and linear endpoint tails are explicit unselected ablations. A linear tail changes only evaluation after the latest encoded cutoff. Shorter hops increase updates; no candidate is automatically promoted.',**identity()))
 plot(traces,prefix);monitor.check()

def plot(traces,prefix='kinematics'):
 os.environ['MPLCONFIGDIR']=str(ROOT/'work/mplconfig');import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(5,2,figsize=(14,15),constrained_layout=True)
 for i,kind in enumerate(KINDS):
  rs=[r for r in traces if r['kind']==kind and r['hop_seconds']==.25]
  methods=['v2','slope_hybrid','robust_slope','half_second','constant_velocity']+(['causal_residual'] if prefix!='kinematics' else [])
  if prefix=='kinematics_kalman':methods=['v2','constant_velocity','causal_residual','kalman_cv','kalman_cv_robust']
  for name in methods:
   rr=sorted([r for r in rs if r['method']==name],key=lambda r:r['time_s'])
   for ax,key in zip(axes[i],['position_x','velocity_x']):ax.plot([r['time_s'] for r in rr],[r[key] for r in rr],label=name,alpha=.8)
  truth_rows=sorted([r for r in rs if r['method']=='v2'],key=lambda r:r['time_s'])
  for ax,key in zip(axes[i],['position_x','velocity_x']):ax.plot([r['time_s'] for r in truth_rows],[r['true_'+key] for r in truth_rows],'k--',lw=2,label='analytic truth');ax.set(xlabel='Time (s)',ylabel='Native px' if key=='position_x' else 'Native px/s',title=kind+' / '+key);ax.legend(fontsize=7,ncol=2)
 fig.suptitle('Synthetic causal cached-state stress test — not real-video acceptance');fig.savefig(path('reports')/(prefix+'.png'),dpi=130);plt.close(fig)
