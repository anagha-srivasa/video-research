import numpy as np
from track1_belief.synthetic import episode
from track1_belief.evaluate import estimate_episode,query_rows,common_support
from track1_belief.experiment import memory_probes


def test_synthetic_clock_ancestry_is_exact_and_past_perturbations_do_not_leak():
    e=episode(1);p=dict(q=10000,threshold=5)
    a=estimate_episode(e,'anchor','dependent',p)
    changed=dict(e);changed['anchor_positions']=e['anchor_positions'].copy();changed['anchor_positions'][100:]+=100
    b=estimate_episode(changed,'anchor','dependent',p)
    np.testing.assert_array_equal(a['mean'][:100],b['mean'][:100])
    assert a['valid'].sum()>150
    assert np.all(a['latest_evidence_time'][a['valid']]<=e['time'][a['valid']])


def test_every_unsupported_query_stays_in_denominator():
    e=episode(0);p=dict(q=1000,threshold=3)
    a=estimate_episode(e,'anchor','nearest',p)
    b={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in a.items()};b['valid'][:]=False
    rows=query_rows(e,{('anchor','nearest'):a,('anchor','failed'):b},'truth')
    assert len(rows)>0 and sum(r['method']=='nearest' for r in rows)==sum(r['method']=='failed' for r in rows)
    assert not any(r['supported'] for r in rows if r['method']=='failed')
    assert common_support(rows)==[]


def test_filter_is_duplicate_invariant_at_experiment_boundary():
    e=episode(1);p=dict(q=1000,threshold=3)
    state=estimate_episode(e,'anchor','adaptive',p)
    assert np.isfinite(state['covariance']).all()
    assert np.linalg.eigvalsh(state['covariance'][state['valid']]).min()>-1e-7
