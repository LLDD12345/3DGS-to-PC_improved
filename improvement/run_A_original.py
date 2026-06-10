import torch
import numpy as np
import open3d as o3d
import mesh_handler_original as mesh

def load_ply_tensors(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.from_numpy(np.asarray(pcd.points)).double()
    colours = torch.from_numpy(np.asarray(pcd.colors) * 255).int()
    normals = torch.from_numpy(np.asarray(pcd.normals)).double() if pcd.has_normals() else None
    return points, colours, normals

pts, cols, nmls = load_ply_tensors("bike_baseline_only.ply")
mesh.generate_mesh(pts, cols, nmls, "A_original_mesh.ply", laplacian_iters=10, std_ratio=3)