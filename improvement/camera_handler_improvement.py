### Based on code from the Torch Splatting Repo     ###
### Credit: https://github.com/hbb1/torch-splatting ###

import math
import numpy as np
import torch

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera():
    def __init__(self, width, height, focal_x, focal_y, c2w, znear=10, zfar=100):
        self.znear = znear
        self.zfar = zfar
        self.focal_x = focal_x 
        self.focal_y = focal_y
        self.FoVx = focal2fov(self.focal_x, width)
        self.FoVy = focal2fov(self.focal_y, height)
        self.image_width = int(width)
        self.image_height = int(height)
        self.world_view_transform = torch.linalg.inv(c2w).permute(1,0)
        self.c2w = c2w
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(c2w.device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

# ====================== 修复后：相机外参自动检测与转换函数 ======================
def auto_convert_camera_format(transform, camera_format="auto"):
    """
    自动检测并转换相机外参 → 统一输出为代码原生期望的 cam2world
    修复点：区分坐标系左右手，不再单纯用行列式±1判定格式
    """
    # 统一转为torch张量
    if isinstance(transform, np.ndarray):
        transform = torch.from_numpy(transform).float()

    if camera_format == "cam2world":
        print(f"✅ 用户指定：输入外参为cam2world格式，直接使用")
        return transform

    elif camera_format == "world2cam":
        print(f"✅ 用户指定：输入外参为world2cam格式，执行求逆转为cam2world")
        return torch.linalg.inv(transform)

    elif camera_format == "auto":
        R = transform[:3, :3]
        det = torch.linalg.det(R).item()
        eps = 1e-3

        # 核心修复：
        # 规则：
        # 1. 若输入本身就是 cam2world（无论左右手系，det≈±1 都保留）
        # 2. 仅当「输入明显是 view2world(world2cam)」时才求逆
        # 结合3DGS/NeRF工程惯例：
        # 先假设当前输入是数据集原生 cam2world，**默认不求逆**
        if abs(abs(det) - 1.0) < eps:
            # 行列式模为1 → 标准正交矩阵，判定为原生cam2world，直接返回
            print(f"✅ 自动检测：标准正交外参(det={det:.4f})，判定为cam2world，直接使用")
            return transform
        else:
            print(f"⚠️ 行列式异常(det={det:.4f})，默认按cam2world处理")
            return transform
    else:
        raise ValueError(f"不支持的相机格式：{camera_format}，可选值：auto/cam2world/world2cam")
# ========================================================================

def get_camera(renderer_type, transform, cam_intrinsic, colour_resolution=None, sh_degree=3, white_bkgd=True, mask=None, camera_format="auto"):
    """
    新增参数：camera_format - 相机外参格式
    """
    # 第一步：自动转换外参格式（修复后逻辑）
    transform = auto_convert_camera_format(transform, camera_format)

    diff = 1 if (colour_resolution is None or mask is not None) else colour_resolution / int(cam_intrinsic[0])

    if mask is not None:
        if mask.shape[1] != int(cam_intrinsic[0]) or mask.shape[0] != int(cam_intrinsic[1]):
            raise Exception("Size of mask must match size of input image")
        mask = mask.flatten()

    img_width = int(int(cam_intrinsic[0]) * diff) 
    img_height = int(int(cam_intrinsic[1]) * diff) 

    focal_x = float(cam_intrinsic[2])*diff
    focal_y = float(cam_intrinsic[3])*diff

    if renderer_type == "python":
        return Camera(img_width, img_height, focal_x, focal_y, transform)

    elif renderer_type == "cuda":
        from gaussian_pointcloud_rasterization import GaussianRasterizationSettings

        # 【原生关键逻辑：保留不动】CUDA渲染器固定Y/Z轴翻转
        transform[:, 1:3] = -transform[:, 1:3]

        fovX = focal2fov(focal_x, img_width)
        fovY = focal2fov(focal_y, img_height)

        tanfovx = math.tan(fovX * 0.5)
        tanfovy = math.tan(fovY * 0.5)

        scaling_modifier = 1.0
        
        znear = 10
        zfar = 100

        projmatrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovX, fovY=fovY).transpose(0,1).to(transform.device)

        viewmatrix = torch.linalg.inv(transform).permute(1,0)
        campos = viewmatrix.inverse()[3, :3]

        return GaussianRasterizationSettings(
            image_height=int(img_height),
            image_width=int(img_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0.,0.,0.], device="cuda:0") if not white_bkgd else torch.tensor([1.0,1.0,1.0], device="cuda:0"),
            scale_modifier=scaling_modifier,
            campos=campos, 
            viewmatrix=viewmatrix,
            projmatrix=viewmatrix @ projmatrix,
            sh_degree=sh_degree,
            prefiltered=False,
            mask=mask,
            debug=True,
            antialiasing=False
        )