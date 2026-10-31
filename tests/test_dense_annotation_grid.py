import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("dense_source_sensitivity",SCRIPTS/"run_eth3d_dense_sensitivity.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_colmap_dense_header_and_channel_order(tmp_path):
    expected=np.arange(24,dtype=np.float32).reshape(2,4,3)
    path=tmp_path/"normal.bin"
    encoded=expected.transpose(1,0,2).reshape(-1,order="F")
    path.write_bytes(b"4&2&3&"+encoded.astype("<f4").tobytes())
    np.testing.assert_array_equal(module.read_dense(path),expected)


def test_nearest_common_grid_preserves_unknowns():
    original=np.arange(16,dtype=np.float32).reshape(4,4)
    original[1,1]=np.nan
    coarse=module.common_grid_laser(original,(2,2))
    np.testing.assert_allclose(coarse,[[np.nan,7],[13,15]],equal_nan=True)


def test_plane_normals_same_rays_after_resize_and_hole_rejection():
    original=np.full((40,60),3,dtype=np.float32)
    colmap=np.full((20,30),3,dtype=np.float32)
    laser=module.common_grid_laser(original,colmap.shape)
    xy=np.asarray([[19.5,19.5],[.2,.2]])
    a,b,keep=module.common_normals(laser,colmap,xy,[40,42,30,20],original.shape)
    np.testing.assert_array_equal(keep,[True,False])
    np.testing.assert_allclose(a[0],[0,0,1],atol=1e-12)
    np.testing.assert_allclose(a,b,atol=1e-12)
    laser[10,11]=np.nan
    assert not module.common_normals(laser,colmap,xy,[40,42,30,20],original.shape)[2][0]


def test_slanted_plane_checks_corner_origin_and_scaled_intrinsics():
    # Analytic camera-z depth on the actual20x30 output grid. Its calibrated
    # pixel centers are index+0.5 with K scaled from the40x60 source image.
    y,x=np.mgrid[:20,:30]
    normal=np.asarray([.2,-.1,1.]);normal/=np.linalg.norm(normal)
    rays=np.stack([(x+.5-15)/20,(y+.5-10)/21,np.ones_like(x)],axis=-1)
    depth=3./(rays@normal)
    first,second,keep=module.common_normals(depth,depth,np.asarray([[19.5,19.5],[39.5,27.5]]),[40,42,30,20],(40,60))
    assert keep.all()
    np.testing.assert_allclose(first,np.broadcast_to(normal,first.shape),atol=1e-6)
    np.testing.assert_allclose(second,first,atol=1e-12)
