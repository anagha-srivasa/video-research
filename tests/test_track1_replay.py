import json
import numpy as np
import pytest
from track1_replay import replay
from track1_encoder_revision.events import InnovationEvidence
from track1_encoder_revision.runtime import CompiledGeometry
from track1_lab.topology_certificate import certify
from track1.numerics import wavelet_matrix


def measurements():
    t=np.arange(0,1.001,.02)
    p=np.c_[np.maximum(t-.2,0)*220,np.zeros(len(t))]
    return t,p,np.ones_like(p)*.01


@pytest.mark.parametrize('hop',[.125,.25,.5])
def test_replay_matches_delayed_delivery_without_duplicate_events(hop):
    t,p,v=measurements();cuts=np.arange(1,int(np.ceil(t[-1]/hop))+1)*hop
    rows=replay(t,p,v,cuts)
    offline=InnovationEvidence()
    for ti,point,var in zip(t,p,v):offline.update(ti,point,var)
    events=offline.drain(1.);assert events
    expected=[dict(e,decision_time=float(cuts[np.searchsorted(cuts,e['evidence_time'])])) for e in events]
    assert [e for r in rows for e in r['events']]==expected
    for r in rows:
        assert r['latest_input_time']<=r['cutoff']
        assert r['observations_received']==np.count_nonzero(t<=r['cutoff'])


def test_future_mutation_cannot_change_any_prefix_output():
    t,p,v=measurements();cuts=[.1,.25,.5];original=p.copy()
    reference=replay(t,p,v,cuts);p[t>.5]=np.nan;v[t>.5]=-100
    assert replay(t,p,v,cuts)==reference
    np.testing.assert_array_equal(p[t<=.5],original[t<=.5])


def test_empty_decisions_and_no_arrived_observations():
    t,p,v=measurements()
    assert replay(t,p,v,[])==[]
    row=replay(t,p,v,[-1.])[0]
    assert row['state'] is None and row['latest_input_time'] is None and row['events']==[]


def test_nonmonotonic_clock_rejected():
    t,p,v=measurements()
    with pytest.raises(ValueError):replay(t,p,v,[.5,.25])
    with pytest.raises(ValueError):replay(t[::-1],p,v,[.5])


def test_late_event_is_not_released_before_decision_time():
    t,p,v=measurements();detector=InnovationEvidence()
    for ti,point,var in zip(t,p,v):detector.update(ti,point,var,decision_time=2.)
    assert detector.drain(1.)==[]
    assert detector.drain(2.)
    assert detector.drain(2.)==[]


def test_curve_certificate_does_not_assert_source_completeness():
    a=np.arange(16)*2*np.pi/16;p=np.c_[np.cos(a),np.sin(a)]
    c=np.zeros((4,35),np.float32);c[0,2:-1]=(wavelet_matrix(16)@p).ravel()
    cert=certify(c,[16],['outer'],[-1],window_seconds=1.)
    metadata=dict(allocation=[16],roles=['outer'],parents=[-1],certificate_json=json.dumps(cert),
        window_seconds=1.,window_start_us=0,source_complete=False)
    # Low-level curve evaluation deliberately has no source-completeness promise.
    decoder=CompiledGeometry(c,metadata,128)
    assert len(decoder.evaluate(500000))==1 and metadata['source_complete'] is False
    for time in (-1,1000001,np.nan):
        with pytest.raises(ValueError):decoder.evaluate(time)
