import copy
import numpy as np
from track1_finetune.fit import choose_family,candidates,predict
from track1_lab.history_detail import detail_basis

def test_selection_requires_participant_protocol_safety():
 rows=[]
 for v in ['P02_122','P03_112']:
  for p in ['single','gap100','endpoint100']:
   for trial,error in [(0,1.),(1,.8),(2,.5)]:
    rows.append(dict(video_id=v,protocol=p,trial=trial,position_mse=1.5 if trial==2 and v=='P03_112' and p=='endpoint100' else error))
 chosen,evidence=choose_family(rows,'position')
 assert chosen==1 and 2 not in evidence['admissible_trials']
 assert choose_family([dict(r,position_mse=1.) for r in rows],'position')[0]==0

def test_search_preserves_baseline_and_uses_past_inputs_only():
 from track1_retrain.temporal import prepare
 t=np.linspace(0,1,51);angle=np.arange(16)*2*np.pi/16;controls=np.c_[np.cos(angle),np.sin(angle)]
 from track1.numerics import wavelet_matrix
 h=np.array([np.r_[tt,tt*.2,(wavelet_matrix(16)@controls).ravel(),0.] for tt in t]);data=dict(h=h,variance=np.ones_like(h)*.01,controls=np.array([controls+[tt,tt*.2] for tt in t]))
 rows=[dict(timestamp_us=round(tt*1e6),centre_native=[tt,tt*.2],scale_native=1.,fit16_valid=True,prior=1.,quality=1.) for tt in t]
 case=dict(indices=list(range(51)),window_end_us=1000000);hidden=t>=.9
 p=prepare(case,rows,data,16,hidden);probe=dict(p=p,times=t[hidden]);options=dict(regularization=1/3,loss='l2');snapshot=copy.deepcopy(options)
 assert candidates(options)[0]==options and options==snapshot
 pred,_=predict(probe,options,np.ones(35),dict(count=3,strength=.01))
 changed=copy.deepcopy(data);changed['h'][hidden]+=1000;changed['controls'][hidden]*=-10
 q=prepare(case,rows,changed,16,hidden);new,_=predict(dict(p=q,times=t[hidden]),options,np.ones(35),dict(count=3,strength=.01))
 np.testing.assert_array_equal(pred,new)
 assert np.max(abs(detail_basis(np.array([1.]))))==0.
 assert np.max(abs(detail_basis(np.array([1.]),1)))==0.
