from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from eth3d_matching_common import project_visible, world_to_camera


def test_identity_half_pixel_and_invalid_depth():
    depth=np.full((12,16),2.)
    depth[3,4]=np.nan
    xy=np.asarray([[1.,2.],[4.,3.],[15.,11.],[-1.,3.]])
    projected,valid=project_visible(xy,depth,depth,[10,11,8,6],[10,11,8,6],np.eye(4))
    np.testing.assert_array_equal(valid,[True,False,True,False])
    np.testing.assert_allclose(projected[valid],xy[valid],atol=1e-12)


def test_translation_occlusion_and_backward_camera():
    depth=np.full((12,16),2.)
    transform=np.eye(4); transform[0,3]=.2
    xy=np.asarray([[4.,5.]])
    projected,valid=project_visible(xy,depth,depth,[10,10,8,6],[10,10,8,6],transform)
    np.testing.assert_allclose(projected,[[5.,5.]])
    assert valid[0]
    occluded=depth.copy(); occluded[5,5]=1.
    assert not project_visible(xy,depth,occluded,[10,10,8,6],[10,10,8,6],transform)[1][0]
    transform[2,3]=-3
    assert not project_visible(xy,depth,depth,[10,10,8,6],[10,10,8,6],transform)[1][0]


def test_rounded_quaternion_is_normalized():
    record={"quaternion_wxyz":[.707107,0,.707107,0],"translation":[1,2,3]}
    matrix=world_to_camera(record)
    np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-12)
    np.testing.assert_allclose(matrix@np.asarray([1,0,0,1]),[1,2,2,1],atol=1e-12)
