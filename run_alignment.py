import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from selectpoints import select_k_points
from alignment import get_3d_corresp, align_svd
from meshutils import writeply

NUM_SCANS = 5 
NUM_CORRESPONDENCE_POINTS = 7 

OUTPUT_DIR = "output"
BASE_PATH = "/Users/siddharthgupta/Desktop/CS 117/koi"

def main():
    base_scan_id = 0
    base_recon_file = os.path.join(OUTPUT_DIR, f"grab_{base_scan_id}_reconstruction.pkl")
    
    try:
        with open(base_recon_file, "rb") as f:
            base_data = pickle.load(f)
    except FileNotFoundError:
        return
        merged_pts = base_data['pts3']
    merged_colors = base_data['colors']
    
    for i in range(1, NUM_SCANS):
        moving_scan_id = i
        
        fixed_img_path = os.path.join(BASE_PATH, f"grab_{base_scan_id}", "color_C0_01_u.png")
        moving_img_path = os.path.join(BASE_PATH, f"grab_{moving_scan_id}", "color_C0_01_u.png")
        
        fixed_img = cv2.cvtColor(cv2.imread(fixed_img_path), cv2.COLOR_BGR2RGB)
        moving_img = cv2.cvtColor(cv2.imread(moving_img_path), cv2.COLOR_BGR2RGB)
        
        moving_recon_file = os.path.join(OUTPUT_DIR, f"grab_{moving_scan_id}_reconstruction.pkl")
        try:
            with open(moving_recon_file, "rb") as f:
                moving_data = pickle.load(f)
        except FileNotFoundError:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.imshow(fixed_img)
        ax1.set_title(f"Reference View (grab_{base_scan_id})")
        ax2.imshow(moving_img)
        ax2.set_title(f"Scan to Align (grab_{moving_scan_id})")
        
        print(f"\nPlease click {NUM_CORRESPONDENCE_POINTS} corresponding points.")
        print("Click on a feature in the LEFT image, then the SAME feature in the RIGHT image.")
        sp1 = select_k_points(ax1, NUM_CORRESPONDENCE_POINTS)
        sp2 = select_k_points(ax2, NUM_CORRESPONDENCE_POINTS)
        plt.show()

        if len(sp1.xs) != NUM_CORRESPONDENCE_POINTS or len(sp2.xs) != NUM_CORRESPONDENCE_POINTS:
            print("Incorrect number of points selected. Skipping this scan.")
            continue
            
        clicked_pts1 = np.vstack((sp1.xs, sp1.ys))
        clicked_pts2 = np.vstack((sp2.xs, sp2.ys))

        X1 = get_3d_corresp(clicked_pts1, base_data['pts2L'], base_data['pts3'])
        X2 = get_3d_corresp(clicked_pts2, moving_data['pts2L'], moving_data['pts3'])
        
        R, t = align_svd(X1, X2)
        aligned_pts = (R @ moving_data['pts3']) + t
        
        merged_pts = np.hstack((merged_pts, aligned_pts))
        merged_colors = np.hstack((merged_colors, moving_data['colors']))

    final_cloud_filename = os.path.join(OUTPUT_DIR, "final_merged_cloud.ply")
    print(f"\nfinal merged point cloud to {final_cloud_filename}")
    
    dummy_faces = np.empty((0, 3), dtype=int)
    writeply(merged_pts, merged_colors, dummy_faces, final_cloud_filename)

if __name__ == "__main__":
    main()