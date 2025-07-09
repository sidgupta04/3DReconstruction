import numpy as np
from scipy.spatial import Delaunay

def create_and_clean_mesh(pts2L, pts3, colors, boxlimits, trithresh):
    # 1. Bounding Box Pruning
    print(f"  Pruning points outside box. Initial points: {pts3.shape[1]}")
    x, y, z = pts3[0, :], pts3[1, :], pts3[2, :]
    in_box = ((x >= boxlimits[0]) & (x <= boxlimits[1]) &
              (y >= boxlimits[2]) & (y <= boxlimits[3]) &
              (z >= boxlimits[4]) & (z <= boxlimits[5]))
    
    pts3 = pts3[:, in_box]
    pts2L = pts2L[:, in_box]
    colors = colors[:, in_box]
    print(f"  Points after pruning: {pts3.shape[1]}")

    if pts3.shape[1] < 3:
        print("  Not enough points to form a mesh. Aborting.")
        return np.empty((3,0)), np.empty((3,0)), np.empty((0,3))

    # 2. Delaunay Triangulation (on 2D points)
    tri = Delaunay(pts2L.T) 
    faces = tri.simplices

    # 3. Long Edge Pruning (on 3D points)
    v = pts3.T
    keep_faces = []
    for f in faces:
        p1, p2, p3 = v[f[0]], v[f[1]], v[f[2]]
        e1 = np.linalg.norm(p1 - p2)
        e2 = np.linalg.norm(p2 - p3)
        e3 = np.linalg.norm(p3 - p1)
        if max(e1, e2, e3) < trithresh:
            keep_faces.append(f)
            
    faces = np.array(keep_faces)

    if len(faces) == 0:
        print("  No triangles left after pruning. Aborting.")
        return np.empty((3,0)), np.empty((3,0)), np.empty((0,3))

    # 4. Unused Vertex Removal
    used_vertices = np.unique(faces.flatten())
    
    new_index_map = -np.ones(pts3.shape[1], dtype=int)
    new_index_map[used_vertices] = np.arange(len(used_vertices))
    
    clean_faces = new_index_map[faces]
    
    clean_pts3 = pts3[:, used_vertices]
    clean_colors = colors[:, used_vertices]
    
    return clean_pts3, clean_colors, clean_faces


def smooth_mesh(pts3, faces, n_iters=1):
    """
    Smooths a mesh using Laplacian smoothing.
    
    Parameters
    ----------
    pts3 : (3, M) np.array
        The 3D vertices of the mesh.
    faces : (F, 3) np.array
        The triangle faces of the mesh.
    n_iters : int
        Number of smoothing iterations to perform.
        
    Returns
    -------
    smoothed_pts3 : (3, M) np.array
        The smoothed vertex positions.
    """
    
    num_verts = pts3.shape[1]
    
    neighbors = [set() for _ in range(num_verts)]
    for f in faces:
        neighbors[f[0]].update([f[1], f[2]])
        neighbors[f[1]].update([f[0], f[2]])
        neighbors[f[2]].update([f[0], f[1]])

    # Perform smoothing
    smoothed_pts3 = np.copy(pts3)
    for i in range(n_iters):
        temp_pts3 = np.copy(smoothed_pts3)
        for v_idx in range(num_verts):
            neighbor_indices = list(neighbors[v_idx])
            if len(neighbor_indices) > 0:
                # Calculate the average position of the neighbors
                avg_pos = np.mean(smoothed_pts3[:, neighbor_indices], axis=1)
                temp_pts3[:, v_idx] = avg_pos
        smoothed_pts3 = temp_pts3
        
    print("  Smoothing complete.")
    return smoothed_pts3