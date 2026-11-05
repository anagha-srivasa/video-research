import copy
import numpy as np
import pytest
from track1_retrain.transport import anchor_geometry,moved_curves
from track1.numerics import wavelet_matrix


def test_transport_preserves_certified_shape_and_rejects_changed_anchor():
 angle=np.arange(16)*2*np.pi/16
 controls=np.c_[np.cos(angle),np.sin(angle)]
 h=np.r_[0.,0.,(wavelet_matrix(16)@controls).ravel(),0.].astype(np.float32)
 row=dict(source_rings=1,mask_uid='fixture',timestamp_us=0)
 anchor=anchor_geometry(row,h,16)
 assert anchor['certificate']['certified']
 original=anchor['coefficients'].copy()
 camera=dict(intrinsic_native=[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]],native_size=[100,100],distortion=[0.,0.,0.,0.,0.])
 a=moved_curves(anchor,np.array([0.,0.]),camera)
 b=moved_curves(anchor,np.array([.1,.2]),camera)
 np.testing.assert_allclose(b[0]-a[0],np.tile([10.,20.],(len(a[0]),1)),atol=1e-6)
 np.testing.assert_array_equal(anchor['coefficients'],original)
 changed=copy.deepcopy(anchor);changed['coefficients'][0,2]+=.01
 with pytest.raises(ValueError,match='binding'):moved_curves(changed,np.zeros(2),camera)
 with pytest.raises(ValueError,match='pose'):moved_curves(anchor,np.array([np.nan,0]),camera)
