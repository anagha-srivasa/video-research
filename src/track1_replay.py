"""Read-only chronological replay audit for the existing Track 1 event channel.

This adapter feeds only observations received by a decision cutoff. It does not
retrain the detector, alter its thresholds or turn machine labels into references.
"""
import copy
import json
import numpy as np
from track1_encoder_revision.events import InnovationEvidence


def replay(times,positions,variances,cutoffs,parameters=None):
    times=np.asarray(times,float);positions=np.asarray(positions,float);variances=np.asarray(variances,float)
    cutoffs=np.asarray(cutoffs,float)
    if times.ndim!=1 or positions.shape!=(len(times),2) or variances.shape!=positions.shape:
        raise ValueError('Expected timestamp and 2D measurement arrays')
    if cutoffs.ndim!=1 or not np.isfinite(times).all() or not np.isfinite(cutoffs).all():
        raise ValueError('Finite one-dimensional clocks required')
    if np.any(np.diff(times)<=0) or np.any(np.diff(cutoffs)<=0):
        raise ValueError('Strictly increasing observation and decision clocks required')
    detector=InnovationEvidence(**(parameters or {}));cursor=0;state=None;rows=[]
    for cutoff in cutoffs:
        while cursor<len(times) and times[cursor]<=cutoff:
            state=detector.update(times[cursor],positions[cursor],variances[cursor],decision_time=cutoff)
            cursor+=1
        events=detector.drain(float(cutoff))
        if any(e['evidence_time']>e['decision_time'] or e['decision_time']>cutoff for e in events):
            raise AssertionError('Future evidence escaped replay')
        rows.append(dict(cutoff=float(cutoff),observations_received=cursor,
            latest_input_time=float(times[cursor-1]) if cursor else None,
            state=copy.deepcopy(state),events=copy.deepcopy(events)))
    return rows


def audit():
    from pipeline.common import ROOT,digest
    from track1_encoder_revision.event_training import episode_inputs
    parameters=json.loads((ROOT/'reports/track1_encoder_revision/event_parameters.json').read_text())['parameters']
    results=[]
    for seed in range(90):
        t,p,v,*_=episode_inputs(seed)
        offline=InnovationEvidence(**parameters)
        for ti,point,var in zip(t,p,v):offline.update(ti,point,var)
        reference=offline.drain(float(t[-1]))
        for hop in (.125,.25,.5):
            cuts=np.arange(1,int(np.ceil(t[-1]/hop))+1)*hop
            rows=replay(t,p,v,cuts,parameters)
            expected=[]
            for event in reference:
                r=dict(event);r['decision_time']=float(cuts[np.searchsorted(cuts,event['evidence_time'],side='left')]);expected.append(r)
            actual=[e for r in rows for e in r['events']]
            if actual!=expected:raise AssertionError('Batch/event-time result differs from chronological replay')
            prefix_cuts=cuts[cuts<=2.5]
            poisoned=p.copy();poisoned[t>prefix_cuts[-1]]=np.nan
            if replay(t,poisoned,v,prefix_cuts,parameters)!=rows[:len(prefix_cuts)]:
                raise AssertionError('Unreceived future measurements influenced prefix')
            results.append(dict(seed=seed,hop_seconds=hop,decisions=len(rows),events=len(actual),
                online_batch_equivalent=True,future_suffix_independent=True,
                decisions_after_evidence=True,event_delivered_once=True))
    return dict(records=results,episodes=90,schedules=len(results),thresholds_changed=False,
        parameters=parameters,source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [
            ROOT/'src/track1_replay.py',ROOT/'src/track1_encoder_revision/events.py',
            ROOT/'src/track1_encoder_revision/event_training.py',ROOT/'reports/track1_encoder_revision/event_parameters.json']},
        scientific_event_accuracy_certified=False,online_rgb_input_certified=False,
        meaning='Causality and scheduling equivalence only, not independent event accuracy')

if __name__=='__main__':
    import argparse
    from pathlib import Path
    from pipeline.common import ROOT,atomic_json
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Output must stay inside project root')
    atomic_json(output,audit())
    print('Passed 270 schedules over 90 episodes; no fitting or threshold changes',flush=True)
