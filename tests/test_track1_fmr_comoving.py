from types import SimpleNamespace
import numpy as np
import pytest
from track1_fmr.dorsal_flux import DorsalFluxExtractor, FluxProposal


def proposal(x0,x1,velocity=(.02,0.),energy=.02):
    mask=np.zeros((100,100),bool);mask[40:50,x0:x1]=True
    area=float(mask.mean())
    return FluxProposal((x0/100,.4,x1/100,.5),mask,area,energy,velocity,.8,
                        energy*np.sqrt(area)*.8,(velocity[0]+.01,velocity[1]))


def test_comoving_merge_preserves_union_weighted_velocity_and_limits():
    extractor=DorsalFluxExtractor()
    a,b=proposal(10,20),proposal(22,32,(.021,0.),.01)
    merged=extractor._merge_comoving_proposals([a,b],100,100)
    assert len(merged)==1
    out=merged[0]
    np.testing.assert_array_equal(out.mask_grid,a.mask_grid|b.mask_grid)
    assert not out.mask_grid[:,20:22].any()  # No invented pixels in the gap.
    assert out.box_norm==(.1,.4,.32,.5) and out.area_frac==pytest.approx(.02)
    assert out.mean_novel_energy==pytest.approx(.015)
    np.testing.assert_allclose(out.mean_vel_norm,(.0205,0.))
    np.testing.assert_allclose(out.mean_image_vel_norm,(.0305,0.))
    assert len(extractor._merge_comoving_proposals([a,proposal(22,32,(-.02,0.))],100,100))==2
    assert len(extractor._merge_comoving_proposals([a,proposal(70,80)],100,100))==2
    capped=DorsalFluxExtractor(max_area_frac=.015)
    assert len(capped._merge_comoving_proposals([a,b],100,100))==2


def test_border_guard_rejects_only_whole_rim_boxes():
    extractor=DorsalFluxExtractor()
    assert extractor._is_border_strip((0.,.3,.025,.8))
    assert extractor._is_border_strip((.975,.3,1.,.8))
    assert extractor._is_border_strip((.2,0.,.8,.025))
    assert extractor._is_border_strip((.2,.975,.8,1.))
    assert not extractor._is_border_strip((0.,.3,.04,.8))
    assert not extractor._is_border_strip((.2,.2,.8,.8))


def test_ego_scaled_floor_suppresses_pan_residual_but_keeps_steady_onset(monkeypatch):
    extractor=DorsalFluxExtractor(None)
    image=np.zeros((160,160,3),np.uint8)
    residual=np.zeros((160,160,2),np.float32);residual[50:100,40:90,0]=.0045
    background=np.zeros_like(residual)
    monkeypatch.setattr(extractor,'_ensure_dis',lambda h,w:SimpleNamespace(
        calc=lambda previous,current,initial:(residual+background)*np.array([160,160],np.float32)))
    monkeypatch.setattr(extractor,'_estimate_ego_residual',
                        lambda observed,known:(observed-background,100,background,'bg_lattice'))
    extractor.step(image)
    steady,_=extractor.step(image)
    assert steady.proposals and steady.effective_v_min==pytest.approx(.0035)
    background[...,0]=.025
    pan,_=extractor.step(image)
    assert pan.bg_speed_median==pytest.approx(.025)
    assert pan.effective_v_min==pytest.approx(.0065)
    assert not pan.raw_motion_mask.any() and not pan.proposals
    with pytest.raises(ValueError):
        DorsalFluxExtractor(ego_speed_scale=-.1)
    with pytest.raises(ValueError):
        DorsalFluxExtractor(border_margin_norm=.5)
