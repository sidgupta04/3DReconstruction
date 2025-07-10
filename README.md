# Structured Light 3D Reconstruction

A complete Python-based pipeline for 3D reconstruction using structured light scanning. This project transforms a sequence of 2D images into a high-fidelity, fully aligned, and colorized 3D mesh, showcasing techniques foundational to reverse engineering, cultural heritage preservation, and generating assets for VR/computer graphics.

## Features

- **Structured Light Processing**: Converts 2D structured light scan images into point clouds and meshes.
- **Robust Scan Alignment**: Aligns multiple scans using the SVD-based Kabsch algorithm, with point correspondences via KD-Tree nearest-neighbor search and user-selected feature points.
- **Mesh Cleaning & Surface Reconstruction**: Cleans, aligns, and merges point clouds, then generates watertight surfaces using Poisson surface reconstruction (via MeshLab).
- **Visualization**: Exports high-resolution 3D meshes for visualization and further refinement in MeshLab.
- **End-to-End Pipeline**: Handles all steps from raw image input to final mesh output.

## Applications

- Reverse engineering and inspection
- Cultural heritage and artifact digitization
- Asset creation for VR, AR, and computer graphics

## Requirements

- **Python 3**
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [MeshLab](https://www.meshlab.net/) (for Poisson surface reconstruction and mesh visualization, external)

Install Python dependencies via pip:

```bash
pip install numpy opencv-python scipy matplotlib
```

MeshLab should be installed separately for mesh surface reconstruction and visualization.

## Usage

1. **Calibration**

    - Ensure you have a pre-computed `calibration.pickle` file in your project directory.  
      _Calibration must be performed in advance (external to this repo)._

2. **Prepare Input Scans**

    - Place each scan sequence in its own folder, named with the prefix `grab_` (e.g., `grab_01`, `grab_02`, ...).
    - Each folder should contain the structured light images for that scan angle or position.

3. **Run the Pipeline**

    ```bash
    python main.py
    ```

    - The script will process all `grab_` folders, generate cleaned meshes, and cache reconstruction data.
    - Outputs include aligned point clouds and 3D meshes ready for further refinement or visualization in MeshLab.

## Output

- **3D Meshes**: Exported in standard formats (e.g., `.ply`, `.obj`) for compatibility with MeshLab and other tools.
- **Reconstruction Data**: Intermediate files for debugging or further processing.

## Notes

- User interaction (using Matplotlib) may be required for feature point selection.
- For best results, ensure all input images are pre-calibrated and consistently named.

## Example

_(Add example images, data, or commands here if available.)_

## License

_(Specify your preferred license here, e.g., MIT, GPL, etc.)_

## Author

sidgupta04

---

Feel free to contribute, report issues, or suggest improvements!
