import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from reconstruction import reconstruct, compute_object_mask
from meshing import create_and_clean_mesh, smooth_mesh
from meshutils import writeply

class Camera:
    def __init__(self, f, c, R, t):
        self.f = f; self.c = c; self.R = R; self.t = t

def main():
    with open("calibration.pickle", "rb") as f:
        calib = pickle.load(f)

    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    extrinsics = calib["extrinsics"]

    camL = Camera(f=np.array([fx, fy]), c=np.array([cx, cy]), R=extrinsics[0]["R"], t=extrinsics[0]["t"])
    camR = Camera(f=np.array([fx, fy]), c=np.array([cx, cy]), R=extrinsics[1]["R"], t=extrinsics[1]["t"])

    base_path = "/Users/siddharthgupta/Desktop/CS 117/koi"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    decode_threshold = 0.05
    masking_threshold = 20
    boxlimits = np.array([0, 250, 0, 200, -100, 100])
    trithresh = 50
    smoothing_iterations = 2

    for i in range(5):
        grab_name = f"grab_{i}"
        grab_dir = os.path.join(base_path, grab_name)
        
        if not os.path.isdir(grab_dir):
            print(f"Directory not found, skipping: {grab_dir}")
            continue
            
        print(f"\n{'='*10} Processing {grab_name} {'='*10}")

        reconstruction_file = os.path.join(output_dir, f"{grab_name}_reconstruction.pkl")
        
        if os.path.exists(reconstruction_file):
            print(f"Loading existing reconstruction from {reconstruction_file}")
            with open(reconstruction_file, "rb") as f:
                data = pickle.load(f)
            pts2L, pts2R, pts3, colors = data['pts2L'], data['pts2R'], data['pts3'], data['colors']
        else:
            print("Performing new reconstruction...")
            background_path = os.path.join(grab_dir, "color_C0_00_u.png")
            object_path = os.path.join(grab_dir, "color_C0_01_u.png")
            background_img = cv2.imread(background_path)
            object_img = cv2.imread(object_path)

            if background_img is None or object_img is None:
                print(f"Error: Could not load color images for {grab_name}. Skipping.")
                continue

            pts2L, pts2R, pts3, colors = reconstruct(
                imprefixL=os.path.join(grab_dir, "frame_C0_"),
                imprefixR=os.path.join(grab_dir, "frame_C1_"),
                threshold=decode_threshold,
                camL=camL,
                camR=camR,
                object_img=object_img,
                background_img=background_img,
                color_img=object_img,
                mask_threshold=masking_threshold
            )

            if pts3.shape[1] > 0:
                print(f"Saving reconstruction data to {reconstruction_file}")
                with open(reconstruction_file, "wb") as f:
                    pickle.dump({'pts2L': pts2L, 'pts2R': pts2R, 'pts3': pts3, 'colors': colors}, f)

        if pts3.shape[1] == 0:
            print("No points to process. Skipping mesh generation.")
            continue

        print("\nStarting mesh generation...")
        mesh_pts, mesh_colors, mesh_faces = create_and_clean_mesh(
            pts2L, pts3, colors, boxlimits, trithresh
        )
        
        if mesh_pts.shape[1] == 0:
            print("  Mesh generation resulted in 0 vertices. Skipping smoothing and saving.")
            continue

        smoothed_pts = smooth_mesh(mesh_pts, mesh_faces, n_iters=smoothing_iterations)

        ply_filename = os.path.join(output_dir, f"{grab_name}_final_mesh.ply")
        print(f"Saving final mesh to {ply_filename}")
        writeply(smoothed_pts, mesh_colors, mesh_faces, ply_filename)
        print("Finished")

if __name__ == "__main__":
    main()