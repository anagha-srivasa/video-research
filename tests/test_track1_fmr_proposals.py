import cv2
import numpy as np
import pytest
from track1_fmr.dorsal_flux import flux, proposal_nms


def test_ranked_nms_suppresses_duplicates_and_caps_birth_proposals():
    proposals = [dict(saliency=float(i), box=np.array([i / 100, 0, i / 100 + .005, .1]))
                 for i in range(30)]
    proposals.append(dict(saliency=100., box=proposals[-1]['box']))
    kept = proposal_nms(proposals)
    assert len(kept) == 15 and kept[0]['saliency'] == 100
    assert not any(p['saliency'] == 29 for p in kept)


def test_normalized_flux_yields_similar_regions_at_two_resolutions():
    rng = np.random.default_rng(77)
    background = rng.integers(0, 100, (160, 160, 3), dtype=np.uint8)
    current = background.copy()
    current[40:80, 50:90] = 255
    counts = []
    for size in (160, 640):
        previous = cv2.resize(background, (size, size), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(current, (size, size), interpolation=cv2.INTER_NEAREST)
        result = flux(previous, image)
        assert result['supported']
        assert 1 <= len(result['clouds']) <= 15
        counts.append(len(result['clouds']))
        main = max(result['clouds'], key=lambda p: p['measure'].mass)['measure']
        np.testing.assert_allclose(main.centroid, [69.5 / 160, 59.5 / 160], atol=.02)
    assert counts[0] == counts[1]
    with pytest.raises(ValueError):
        flux(background, current, min_area_frac=-1)
