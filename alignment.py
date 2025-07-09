import numpy as np
from scipy.spatial import cKDTree

def get_3d_corresp(clicked_pts, pts2L, pts3):
    kdtree = cKDTree(pts2L.T)
    distances, indices = kdtree.query(clicked_pts.T)
    corresp_pts3 = pts3[:, indices]
    return corresp_pts3

def align_svd(P1, P2):
    X1 = P1
    X2 = P2
    
    m1 = np.mean(X1, axis=1, keepdims=True)
    m2 = np.mean(X2, axis=1, keepdims=True)
    
    X1c = X1 - m1
    X2c = X2 - m2
    
    H = X2c @ X1c.T
    
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T 

    R = V @ U.T
    
    if np.linalg.det(R) < 0:
        # smooths out reflections
        V[:, -1] *= -1 
        R = V @ U.T 
    
    t = m1 - R @ m2
    
    return R, t