"""Calibrated zero-based pixel-center projection helpers for ETH3D evaluation."""
import numpy as np
from scipy.spatial.transform import Rotation


def world_to_camera(record):
    q = np.asarray(record["quaternion_wxyz"], dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = record["translation"]
    return matrix


def project_visible(source_xy, source_depth, target_depth, source_params, target_params,
                    target_from_source, relative_depth_tolerance=.02):
    """Forward laser-GT projection; depth agreement only establishes query visibility.

    Candidate labels use forward image error and do not demand target depth
    at the candidate location. Missing target depth makes visibility unknown.
    """
    xy = np.asarray(source_xy, dtype=np.float64)
    count = len(xy)
    output = np.full((count, 2), np.nan)
    finite = np.isfinite(xy).all(axis=1)
    ij = np.zeros((count, 2), dtype=np.int64)
    ij[finite] = np.floor(xy[finite]+.5).astype(np.int64)
    valid = finite & (ij[:, 0]>=0) & (ij[:, 0]<source_depth.shape[1])
    valid &= (ij[:, 1]>=0) & (ij[:, 1]<source_depth.shape[0])
    rows = np.flatnonzero(valid)
    z = source_depth[ij[rows, 1], ij[rows, 0]]
    keep = np.isfinite(z)&(z>0)
    rows,z = rows[keep],z[keep]
    fx,fy,cx,cy = source_params
    cloud = np.column_stack([(xy[rows,0]+.5-cx)*z/fx, (xy[rows,1]+.5-cy)*z/fy, z, np.ones(len(z))])
    target = cloud @ np.asarray(target_from_source).T
    valid_z = np.isfinite(target).all(axis=1)&(target[:,2]>1e-12)
    rows,target = rows[valid_z],target[valid_z]
    fx,fy,cx,cy = target_params
    corner = np.column_stack([fx*target[:,0]/target[:,2]+cx,fy*target[:,1]/target[:,2]+cy])
    inside = (corner[:,0]>=0)&(corner[:,0]<target_depth.shape[1])&(corner[:,1]>=0)&(corner[:,1]<target_depth.shape[0])
    rows,target,corner = rows[inside],target[inside],corner[inside]
    index = np.floor(corner).astype(np.int64)
    measured = target_depth[index[:,1],index[:,0]]
    visible = np.isfinite(measured)&(measured>0)&(np.abs(measured-target[:,2])<=relative_depth_tolerance*target[:,2])
    output[rows[visible]] = corner[visible]-.5
    return output, np.isfinite(output).all(axis=1)
