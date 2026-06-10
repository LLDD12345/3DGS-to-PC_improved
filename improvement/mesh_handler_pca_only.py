import numpy as np
import torch
import open3d as o3d
import threading
import time
import sys

# 全局变量控制加载动画
animation_running = False

def loading_animation():
    chars = ['|', '/', '-', '\\']
    i = 0
    while animation_running:
        sys.stdout.write('\r   正在计算中... ' + chars[i % len(chars)])
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write('\r   正在计算中... 完成！\n')
    sys.stdout.flush()

def generate_ball_pivoting_mesh(point_cloud, laplacian_iters=20):
    global animation_running
    print(f"\n[3/5] 正在进行球旋转法表面重建")
    print("   ⚠️  预计耗时：10-30秒")
    animation_running = True
    animation_thread = threading.Thread(target=loading_animation)
    animation_thread.start()
    radii = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05] 
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud, o3d.utility.DoubleVector(radii)
    )
    animation_running = False
    animation_thread.join()
    print(f"\n[4/5] 正在进行拉普拉斯平滑（迭代次数：{laplacian_iters}）...")
    try:
        cleaned_mesh = mesh.filter_smooth_laplacian(number_of_iterations=laplacian_iters, filter_scope=o3d.geometry.FilterScope.Vertex)
        cleaned_mesh.compute_vertex_normals()
    except Exception:
        cleaned_mesh = mesh
    return cleaned_mesh

def convert_pytorch_to_o3d_pointcloud(points, colours, normals):
    point_cloud = o3d.geometry.PointCloud()
    colours = torch.clamp(colours, min=0, max=255).to(torch.int32)
    point_cloud.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    point_cloud.colors =  o3d.utility.Vector3dVector(colours.detach().cpu().numpy()/255)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())
    return point_cloud

def convert_o3d_to_pytorch_pointcloud(point_cloud, device="cuda:0"):
    points = torch.from_numpy(np.asarray(point_cloud.points)).type(torch.double).to(device)
    colours = torch.from_numpy(np.asarray(point_cloud.colors)*255).type(torch.int).to(device)
    if point_cloud.normals is not None:
        normals = torch.from_numpy(np.asarray(point_cloud.normals)).type(torch.double).to(device)
    return points, colours, normals

def clean_point_cloud(points, colours, normals, std_ratio=10, device="cuda:0"):
    point_cloud = convert_pytorch_to_o3d_pointcloud(points, colours, normals)
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    return convert_o3d_to_pytorch_pointcloud(point_cloud)

# ==================== 仅PCA法线版（C） ====================
def generate_mesh(points, colours, normals, output_path, depth=12, laplacian_iters=10, std_ratio=3):
    """
    仅添加PCA法线重估计，不进行降采样，固定半径
    """
    print("\n" + "="*60)
    print("🚀 [消融C] 仅PCA法线（无降采样，固定半径）")
    print("="*60)

    print("\n[1/5] 转换点云格式...")
    point_cloud = convert_pytorch_to_o3d_pointcloud(points, colours, normals)
    print(f"   原始点数: {len(point_cloud.points):,}")

    print(f"[2/5] 统计离群点去除（std_ratio={std_ratio}）...")
    original_count = len(point_cloud.points)
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    print(f"   去除 {original_count - len(point_cloud.points)} 个离群点")
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=10, radius=0.08)
    print(f"   半径去噪后点数: {len(point_cloud.points):,}")

    print("[3/5] PCA法线重估计...")
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_cloud.orient_normals_consistent_tangent_plane(30)

    print("[4/5] 球旋转重建（固定半径）...")
    mesh = generate_ball_pivoting_mesh(point_cloud, laplacian_iters=laplacian_iters)

    print("[5/5] 保存网格...")
    o3d.io.write_triangle_mesh(output_path, mesh)

    print("\n" + "="*60)
    print("✅ 网格生成完成！")
    print(f"   顶点数: {len(mesh.vertices)}")
    print(f"   三角面数: {len(mesh.triangles)}")
    print(f"   保存路径: {output_path}")
    print("="*60 + "\n")