import cv2
import numpy as np
import pytest
from track1_fmr.dorsal_flux import anti_alias, flux, proposal_nms, explained_density


def test_ranked_nms_suppresses_duplicates_and_caps_birth_proposals():
    proposals = [dict(saliency=float(i), box=np.array([i / 100, 0, i / 100 + .005, .1]))
                 for i in range(30)]
    proposals.append(dict(saliency=100., box=proposals[-1]['box']))
    kept = proposal_nms(proposals)
    assert len(kept) == 15 and kept[0]['saliency'] == 100
    assert not any(p['saliency'] == 29 for p in kept)


def moving_patch():
    rng = np.random.default_rng(77)
    background = rng.integers(80, 100, (160, 160, 3), dtype=np.uint8)
    texture = rng.integers(90, 105, (50, 50, 3), dtype=np.uint8)
    before, after = background.copy(), background.copy()
    before[50:100, 40:90] = texture
    after[50:100, 44:94] = texture
    return before, after


def test_dis_detects_low_contrast_motion_and_excludes_known_density():
    previous, current = moving_patch()
    # All RGB differences remain below the previous absolute-difference threshold.
    assert np.abs(current.astype(float) - previous).mean(axis=2).max() < 25
    result = flux(previous, current)
    assert result['supported'] and 1 <= len(result['clouds']) <= 10
    assert result['motion_mask'][55:95, 50:85].any()
    known = np.zeros((160, 160), np.float32)
    known[40:110, 30:110] = 1
    gated = flux(previous, current, excluded=known,
                 known_regions=[dict(density=known, displacement=np.array([.025, 0.]))])
    assert gated['supported']
    assert gated['explained_density'][60:90, 55:80].mean() > .5
    assert gated['residual_flow'].shape == (160, 160, 2)
    with pytest.raises(ValueError):
        flux(previous, current, excluded=known + 1)


def test_dis_affine_background_does_not_become_novel_motion():
    rng = np.random.default_rng(91)
    previous = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    current = cv2.warpAffine(previous, np.array([[1, 0, 3], [0, 1, 1]], float), (160, 160),
                             borderMode=cv2.BORDER_REFLECT)
    result = flux(previous, current)
    assert result['supported']
    assert result['motion_mask'][20:-20, 20:-20].mean() < .05
    assert len(result['clouds']) <= 10


def test_antialias_preserves_native_and_rejects_upsampling():
    rgb = np.repeat(((np.indices((320, 640)).sum(axis=0) % 2) * 255)[..., None], 3, axis=2).astype(np.uint8)
    assert anti_alias(rgb) is rgb
    grid = anti_alias(rgb, 160)
    assert grid.shape == (160, 160, 3) and grid.std() < 2
    with pytest.raises(ValueError):
        anti_alias(rgb, 800)


def test_velocity_exclusion_preserves_onset_but_suppresses_matching_motion():
    residual = np.zeros((20, 20, 2), np.float32)
    residual[..., 0] = .02
    background = np.zeros_like(residual)
    background[..., 0] = .01
    density = np.ones((20, 20), np.float32)
    # An image velocity of 1.5 normalized units/s over .02s explains .02 residual + .01 camera flow.
    matched = explained_density(residual, background, [dict(density=density, displacement=np.array([1.5, 0]) * .02)])
    onset = explained_density(residual, background, [dict(density=density, displacement=np.zeros(2))])
    assert matched.min() > .999 and onset.max() < .001
    # The proposed sigma is not a guarantee that all motion just above v_min survives.
    subtle = np.full((20, 20, 2), 0., np.float32)
    subtle[..., 0] = .004
    explained = explained_density(subtle, np.zeros_like(subtle), [dict(density=density, displacement=np.zeros(2))])
    assert (.004 * (1 - explained)).max() < .0035


def test_dis_effective_patch_scale_is_explicit():
    previous, current = moving_patch()
    for size, scale in [(160, 0), (320, 1)]:
        before = cv2.resize(previous, (size, size))
        after = cv2.resize(current, (size, size))
        result = flux(before, after)
        assert result['supported'] and result['finest_scale'] == scale
