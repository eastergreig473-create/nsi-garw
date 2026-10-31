import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("md_revision",SCRIPTS/"run_megadepth_revision.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_image_slope_is_scale_invariant_and_is_not_a_camera_normal():
    h,w=20,30
    y,x=np.mgrid[:h,:w]
    depth=np.exp(.2*x/w+.3*y/h)
    image=np.full((h,w,3),128,dtype=np.uint8)
    g,a,keep=module.image_slope_attributes(image,depth)
    second,_,same=module.image_slope_attributes(image,depth*7)
    np.testing.assert_allclose(g[keep],second[same],atol=1e-6)
    expected=np.asarray([-.2,-.3,1.]);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(g[keep],np.broadcast_to(expected,g[keep].shape),atol=1e-6)
    np.testing.assert_allclose(a,0)


def test_missing_depth_rejects_full_difference_stencil():
    depth=np.ones((8,8));depth[4,4]=0
    _,_,keep=module.image_slope_attributes(np.zeros((8,8,3),dtype=np.uint8),depth)
    assert not keep[4,4] and not keep[4,3] and not keep[3,4]
    assert keep[2,2] and not keep[0,0]


def test_cache_key_separates_multiple_reconstructions_of_same_photo():
    a=module.DATA/"datasets/MegaDepth/MegaDepth_v1/0012/dense0/depths/photo.h5"
    b=module.DATA/"datasets/MegaDepth/MegaDepth_v1/0012/dense1/depths/photo.h5"
    assert module.image_key(a)!=module.image_key(b)
