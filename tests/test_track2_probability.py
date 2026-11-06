import numpy as np
from track2_probability.core import arms,fit,scores


def test_score_baselines_and_ranking():
    y=np.array([0.,1.,0.,1.]);r=scores(np.full(4,.5),y)
    assert r['brier']==.25 and r['auc']==.5
    assert scores(np.array([.1,.9,.2,.8]),y)['auc']==1


def test_metadata_and_motion_arms_are_separate():
    rng=np.random.default_rng(42);raw=rng.normal(size=(8,4,21));c=rng.normal(size=(8,4,4,2));anchor=rng.normal(size=(8,2))
    a=arms(raw,c,anchor);changed=raw.copy();changed[:,:,2:4]+=100;b=arms(changed,c,anchor)
    np.testing.assert_array_equal(a['position_metadata'],b['position_metadata'])
    assert not np.array_equal(a['position_motion'],b['position_motion'])
    changed=raw.copy();changed[:,:,4:8]+=100;b=arms(changed,c,anchor)
    np.testing.assert_array_equal(a['position_motion'],b['position_motion'])
    assert not np.array_equal(a['position_metadata'],b['position_metadata'])


def test_heldout_targets_do_not_change_fits():
    rng=np.random.default_rng(13);x=rng.normal(size=(100,3));sq=rng.exponential(size=100);p=np.array(['a']*40+['b']*40+['c']*20);ids=np.arange(80)
    a=fit(x,sq,p,ids);sq[80:]+=1000;b=fit(x,sq,p,ids)
    for k in ['bp','bs','scale','probability','variance']:np.testing.assert_array_equal(a[k],b[k])
    assert a['threshold']==b['threshold']


def test_shared_anchor_noise_mechanism():
    rng=np.random.default_rng(91);e=rng.normal(size=(100000,4,2));past=np.sum((e[:,1]-e[:,0])**2,1);shared=np.sum((e[:,2]-e[:,1])**2,1);independent=np.sum((e[:,2]-e[:,3])**2,1)
    assert abs(np.corrcoef(past,shared)[0,1]-.25)<.015
    assert abs(np.corrcoef(past,independent)[0,1])<.015
