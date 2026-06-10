# 3DGS-to-PC Improvement: Dense‑only Point Cloud Extraction & Robust Mesh Generation

This repository is a fork of the official [3DGS-to-PC](https://github.com/Lewis-Stuart-11/3DGS-to-PC) framework.  
We introduce two methodological improvements that address fundamental limitations not discussed in the original paper:

1. **Dense‑only point cloud extraction** – removes sparse background points (e.g., distant ground, floating artifacts) using DBSCAN clustering.  
2. **Robust mesh generation** – replaces the fixed‑radius ball pivoting with an adaptive pipeline (adaptive downsampling + PCA normals + adaptive radii) that never hangs and produces high‑quality meshes.

All improvements are implemented in the `improvement/` folder and are fully compatible with the original code.

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHA5MXptbjBjOGY1MzVwczFyejIydW1zdmdmejQ0aThkOG8wMXE2YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7UknswhXAHe88S93OY/giphy-downsized-large.gif" width="45%" />
</p>

---

## How to install

Clone this repository:
```bash
git clone https://github.com/LLDD12345/3DGS-to-PC-improved
cd 3DGS-to-PC-improved
```

Follow the same installation steps as the original:
```bash
pip install ./gaussian-pointcloud-rasterization   # optional, CUDA renderer
pip install open3d                                 # required for meshing
```

All improvements are located in the `improvement/` folder. You can either copy the improved scripts to the root directory, or run them directly from `improvement/`.

---

## How to run – Baseline vs. Improved

### Baseline (original code)
```bash
python gauss_to_pc.py --input_path bike.ply --transform_path bike_c2w.json --output_path baseline.ply --num_points 30000000
```

### Improved version (dense‑only + robust mesh)
```bash
python improvement/gauss_to_pc_improvement.py \
  --input_path bike.ply \
  --transform_path bike_c2w.json \
  --output_path improved.ply \
  --num_points 30000000 \
  --min_opacity 0.03 \
  --colour_quality high \
  --visibility_threshold 0.0 \
  --keep_dense_only \
  --dbscan_eps 0.15 \
  --dbscan_min_points 100 \
  --dense_voxel_size 0.1 \
  --generate_mesh \
  --mesh_output_path improved_mesh.ply
```

> **Note**: The improved version automatically handles `cam2world` camera poses (no manual conversion needed). The original code expects `world2cam`.

---

## New Functionality (Improvements)

| New Argument                    | Default Value  | Description |
| :---                            |  :----:        |          ---: |
| `keep_dense_only`               | False          | Enable dense‑only extraction: removes sparse background points using DBSCAN clustering. |
| `dbscan_eps`                    | 0.15           | Clustering radius (in meters) for DBSCAN. Increase for larger scenes (e.g., 0.3–0.5). |
| `dbscan_min_points`             | 100            | Minimum points to form a cluster. |
| `dense_voxel_size`              | 0.1            | Voxel size for pre‑downsampling before DBSCAN (reduces memory). |

In addition, the improved mesh generation (enabled by `--generate_mesh` with the improved `mesh_handler.py`) now uses:
- **Adaptive downsampling** (voxel size = scene diagonal × 0.002)
- **PCA normal estimation** (instead of Gaussian shortest‑axis)
- **Adaptive ball pivoting radii** (based on average point spacing)

These features require **no extra arguments** and work seamlessly with the original `--generate_mesh` flag.

---

## Functionality (All original arguments are still supported)

The following table lists all original arguments (unchanged) plus the new ones. For a detailed description of the original arguments, please refer to the [original README](https://github.com/Lewis-Stuart-11/3DGS-to-PC).

| Argument                        | Default Value  | Description |
| :---                            |  :----:        |          ---: |
| `input_path`                    | -              | Path to ply or splat file. |
| `output_path`                   | `3dgs_pc.ply`  | Output point cloud file. |
| `transform_path`                | -              | Camera poses (now auto‑detects `cam2world` / `world2cam`). |
| ... (all original arguments)    | ...            | ... |
| `keep_dense_only` (new)         | False          | Remove background points via DBSCAN. |
| `dbscan_eps` (new)              | 0.15           | DBSCAN radius (meters). |
| `dbscan_min_points` (new)       | 100            | Minimum points per cluster. |
| `dense_voxel_size` (new)        | 0.1            | Pre‑downsampling voxel size for DBSCAN. |

---

## Tips

### Removing background noise (point cloud)
- Use `--keep_dense_only` to automatically keep only the largest connected component (the main object).  
- Adjust `--dbscan_eps` to match your scene scale: for a large outdoor scene (e.g., garden) use `0.3–0.5`; for small objects use `0.05–0.1`.  
- You can combine `--keep_dense_only` with `--clean_pointcloud` for even cleaner results.

### Mesh generation – no more hangs!
- The original fixed‑radius ball pivoting often hangs on medium‑scale point clouds (>3M points).  
- Our improved mesh generation (used automatically when `--generate_mesh` is set with the improved `mesh_handler.py`) never hangs and produces smooth, watertight meshes.  
- If you need even higher quality, set `--laplacian_iterations` to 10–20.

### Speed
- The dense‑only extraction adds only a few seconds of overhead.  
- The robust mesh generation is **faster** than the original because of adaptive downsampling (tested: original hangs, improved finishes in <150 seconds).  
- To speed up colour rendering, you can still use `--camera_skip_rate` and `--colour_quality` as in the original.

---

## Citation

If you use this improved framework, please cite the original paper:

```bibtex
@InProceedings{A_G_Stuart_2025_ICCV,
    author    = {A G Stuart, Lewis and Morton, Andrew and Stavness, Ian and Pound, Michael P},
    title     = {3DGS-to-PC: 3D Gaussian Splatting to Dense Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {3730-3739}
}
```

For the methodological improvements presented in this fork, please refer to the accompanying **SLAM Final Project Report** (Sections 5–6).

---

## Repository Structure (Improvement branch)

```
.
├── improvement/                     # All improved code
│   ├── gauss_to_pc_improvement.py   # Main point cloud generator (with dense‑only)
│   ├── mesh_handler_improved.py     # Robust mesh generation (adaptive)
│   ├── mesh_handler_downsample_only.py
│   ├── mesh_handler_pca_only.py
│   ├── run_A_original.py
│   ├── run_B_downsample_only.py
│   ├── run_C_pca_only.py
│   ├── run_D_improved.py
│   └── ...
├── README_IMPROVEMENT.md            # This file
└── (original files remain unchanged)
```

---

## Author & Course

**Author:** Shuhan Lu  
**Course:** SLAM Final Project  
**Date:** June 2026

---

## License

Same as the original 3DGS-to-PC repository.
```