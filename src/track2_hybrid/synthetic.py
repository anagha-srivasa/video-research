"""Exact synthetic stress, separate event evidence, no independent-video claim."""
import json
import numpy as np
from pipeline.common import ROOT,atomic_json
from track1_encoder_revision.event_training import episode_inputs
from track1_encoder_revision.events import InnovationEvidence
from track1_lab.kinematics import truth
from track1_lab.causal_kinematics import estimate
from track1_reference.events import interval_scores
from .experiment import REPORT,filtered
from .imm import forecast


def episode(seed):
    params=json.loads((REPORT/'selection.json').read_text())['parameters']
    event_params=json.loads((ROOT/'reports/track1_encoder_revision/event_parameters.json').read_text())['parameters']
    t,p,v,targets,quiet,kind,corruption,onset,duration=episode_inputs(seed)
    records=[]
    for hop in (.125,.25):
        detector=InnovationEvidence(**event_params)
        for ti,point,var in zip(t,p,v):detector.update(ti,point,var)
        evidence=[dict(e,decision_time=float(1+np.ceil((e['evidence_time']-1)/hop-1e-10)*hop))
                  for e in detector.drain(4.) if e['kind']=='motion_onset' and e['evidence_time']>=1.]
        methods=('imm','kalman_cv','kalman_stationary','cv','nearest_observation')
        detections={m:[] for m in methods};active={m:False for m in methods};errors={m:[] for m in methods}
        attempted=0;supported=0
        for cutoff in np.arange(1.,3.001,hop):
            attempted+=1
            h=dict(times=(t-cutoff).tolist(),positions=((p-[960,540])/1000).tolist(),variances=(v/1e6).tolist())
            cv=estimate(t,p,v,float(cutoff))
            try: states={m:filtered(h,params,m) for m in methods[:3]}
            except ValueError:continue
            if not cv.support_valid:continue
            supported+=1;nearest=p[np.flatnonzero(t<=cutoff)[-1]]
            for m in methods:
                if m in states:x=forecast(states[m],0)[0]; pos=x[:2]*1000+[960,540];vel=x[2:]*1000
                elif m=='cv':pos=np.array(cv.position);vel=np.array(cv.velocity)
                else:pos=nearest;vel=np.zeros(2)
                moving=np.linalg.norm(vel)>=100
                if moving and not active[m]:detections[m].append(dict(decision_time=float(cutoff)))
                active[m]=moving
                p0,v0=truth(np.array([cutoff]),kind,onset,duration)
                row=[float(np.sum((pos-p0[0])**2)),float(np.sum((vel-v0[0])**2))]
                for horizon in (.25,.5):
                    target=truth(np.array([cutoff+horizon]),kind,onset,duration)[0][0]
                    pred=forecast(states[m],horizon)[0][:2]*1000+[960,540] if m in states else pos+horizon*vel
                    row.append(float(np.sum((pred-target)**2)))
                errors[m].append(row)
        # Score events only inside the actually audited [1,3] update interval.
        event_targets=[r for r in targets if r['start']<=3.]
        quiet_scored=[(max(a,1.),min(b,3.)) for a,b in quiet if min(b,3.)>max(a,1.)]
        for m in (*methods,'separate_innovation_evidence'):
            ev=evidence if m=='separate_innovation_evidence' else detections[m]
            ev=[e for e in ev if 1<=e['decision_time']<=3.]
            metric=np.mean(errors[m],axis=0).tolist() if m in errors and errors[m] else [None]*4
            records.append(dict(seed=seed,kind=kind,corruption=corruption,hop_seconds=hop,method=m,
                attempted=attempted,supported=supported,position_mse=metric[0],velocity_mse=metric[1],
                continuation250_mse=metric[2],continuation500_mse=metric[3],**interval_scores(ev,event_targets,quiet_scored)))
    return records


def summarize(records):
    rows=[r for group in records for r in group];summary=[]
    for hop in (.125,.25):
        for method in sorted({r['method'] for r in rows}):
            rs=[r for r in rows if r['method']==method and r['hop_seconds']==hop]
            events=sum(r['events'] for r in rs);seconds=sum(r['quiet_seconds'] for r in rs)
            delays=[x[0] for r in rs for x in r['detection_delay_intervals_seconds']]
            metrics={k.replace('_mse','_rmse'):float(np.sqrt(np.mean([r[k] for r in rs if r[k] is not None])))
                     if any(r[k] is not None for r in rs) else None
                     for k in ('position_mse','velocity_mse','continuation250_mse','continuation500_mse')}
            summary.append(dict(method=method,hop_seconds=hop,recall=sum(r['recalled'] for r in rs)/events if events else None,
                false_alarms_per_quiet_minute=60*sum(r['quiet_false_alarms'] for r in rs)/seconds if seconds else None,
                median_delay_seconds=float(np.median(delays)) if delays else None,**metrics))
    atomic_json(REPORT/'synthetic.json',dict(records=rows,summary=summary,seeds=list(range(60,90)),
        previously_exposed_synthetic_diagnostics=True,independent_real_reference=False,
        event_interval_seconds=[1,3],event_definition='rising speed >=100 px/s; separate innovation evidence uses locked training-only thresholds',
        missing_support_recorded=True,mode_weights_are_not_contact_or_event_probabilities=True))
