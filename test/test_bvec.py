import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dipy.io import read_bvals_bvecs
from os.path import join
import nibabel as nib

def save_bvecs_to_nifti(bvecs, filename):
    """
    将 bvec 向量保存为 3D 的 NIfTI 文件
    参数：
        bvecs: numpy array，形状为 (n, 3)，表示 bvec 向量
        filename: 保存的文件名，建议扩展名为 .nii.gz
    """
    # 将 (n, 3) 数据重新组织为 3D 数据
    n = bvecs.shape[0]
    grid_size = int(np.ceil(np.cbrt(n)))  # 计算可以容纳 n 个点的立方体尺寸
    padded_bvecs = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)

    # 填充 bvec 数据到前 n 个位置
    indices = np.unravel_index(range(n), (grid_size, grid_size, grid_size))
    for i in range(3):  # 填充每个通道（x, y, z）
        padded_bvecs[indices[0], indices[1], indices[2], i] = bvecs[:, i]

    # 创建 NIfTI 图像
    affine = np.eye(4)  # 单位仿射矩阵
    nifti_img = nib.Nifti1Image(padded_bvecs, affine)

    # 保存为 .nii.gz 文件
    nib.save(nifti_img, filename)
    print(f"Saved bvecs to {filename}")

def visualize_bvec_sphere_with_views(bvecs):
    """
    可视化 bvec 向量为 3D 球面图，从不同视角观察
    参数：
        bvecs: numpy array，形状为 (n, 3)，表示 bvec 向量
    """
    # 创建 3D 图像
    fig = plt.figure(figsize=(16, 8))

    # 定义视角
    view_angles = [(30, 30), (30, 60), (30, 90), (60, 30), (90, 0), (120, 30)]

    for i, (elev, azim) in enumerate(view_angles, 1):
        ax = fig.add_subplot(2, 3, i, projection='3d')

        # 绘制 bvecs 点
        x, y, z = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
        ax.scatter(x, y, z, c='b', s=50, label='bvec directions')

        # 绘制单位球
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="lightgray", alpha=0.5, linewidth=0.5)

        # 设置视角
        ax.view_init(elev=elev, azim=azim)

        # 设置坐标轴和标题
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"View: elev={elev}, azim={azim}")
        ax.set_box_aspect([1, 1, 1])  # 保持球体比例一致

    plt.tight_layout()
    plt.show()

# 示例 bvec 数据
# bvecs = np.array([
#     [0.7071, 0.0000, -0.7071],
#     [0.0000, 1.0000,  0.0000],
#     [0.7071, 0.0000,  0.7071],
#     [-0.7071, 0.0000, 0.7071],
#     [0.0000, -1.0000, 0.0000],
#     [-0.7071, 0.0000, -0.7071]
# ])

sub_path = '/data/wtl/HCP_MGH/mgh_1002/'
bvals, bvecs = read_bvals_bvecs(join(sub_path, 'dwi_b10k.bval'), join(sub_path, 'dwi_b10k.bvec'))
print(bvecs.shape)
# print(bvecs)
# 可视化
# visualize_bvec_sphere_with_views(bvecs[:60])

# save_bvecs_to_nifti(bvecs[:64], "bvecs_3D.nii.gz")