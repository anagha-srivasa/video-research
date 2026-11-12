import numpy as np
import pytest
import cv2

from track1_belief.anchors import PersistentAnchors, AnchorConfig, similarity_fit


def textured_scene():
    rng = np.random.default_rng(28)
    image = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
    mask = np.zeros(image.shape, np.uint8)
    mask[75:165, 110:210] = 1
    return image, mask


def translated(image, dx, dy):
    return cv2.warpAffine(image, np.float32([[1,0,dx],[0,1,dy]]), (image.shape[1],image.shape[0]))


def test_similarity_uses_supplied_subset_and_covariance_is_psd():
    rng=np.random.default_rng(7)
    points=rng.normal(size=(32,2))*20
    angle=.12
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    target=points@rotation.T+np.array([4.,-3.])
    result=similarity_fit(points,target,[0,0])
    np.testing.assert_allclose(result['position'],[4,-3],atol=1e-8)
    assert np.linalg.eigvalsh(result['covariance']).min()>0
    target[-1]+=1000
    other=similarity_fit(points,target,[0,0])
    np.testing.assert_allclose(other['position'],[4,-3],atol=1e-7)


def test_native_tracking_camera_only_translation_and_disjoint_roles():
    image,mask=textured_scene()
    tracker=PersistentAnchors(image,mask,0)
    reference=tracker.reference.copy()
    output=tracker.update(translated(image,3,2),20_000)
    assert output['valid'] and output['proxy_valid'] and output['camera_valid']
    np.testing.assert_allclose(output['position'],reference+[3,2],atol=.1)
    np.testing.assert_allclose(output['camera_relative_position'],reference,atol=.15)
    assert {0,1,2}==set(tracker.roles)
    assert output['latest_evidence_us']==20_000


def test_heldout_points_do_not_enter_fitted_pose():
    image,mask=textured_scene()
    tracker=PersistentAnchors(image,mask,0)
    tracker.update(translated(image,2,1),20_000)
    before=tracker._summarize()
    tracker.points[tracker.roles==1]+=[30,40]
    after=tracker._summarize()
    np.testing.assert_array_equal(before['position'],after['position'])
    np.testing.assert_array_equal(before['covariance_local'],after['covariance_local'])
    assert not np.allclose(before['proxy_position'],after['proxy_position'])


def test_prefix_causality_and_gap_fail_closed():
    image,mask=textured_scene()
    first=PersistentAnchors(image,mask,0)
    prefix=first.update(translated(image,2,1),20_000)['position'].copy()
    first.update(translated(image,-8,6),40_000)
    second=PersistentAnchors(image,mask,0)
    np.testing.assert_array_equal(prefix,second.update(translated(image,2,1),20_000)['position'])
    output=second.update(image,220_000)
    assert not output['valid'] and not output['proxy_valid']
    assert not second.update(image,240_000)['valid']
    with pytest.raises(ValueError):
        second.update(image,240_000)


def test_no_texture_is_unsupported_not_a_fake_zero_measurement():
    image=np.zeros((120,160),np.uint8)
    mask=np.ones(image.shape,np.uint8)
    tracker=PersistentAnchors(image,mask,0)
    assert not tracker.last['valid']
    assert tracker.last['reason']=='insufficient_consistent_persistent_anchors'
    assert not tracker.update(image,20_000)['valid']
