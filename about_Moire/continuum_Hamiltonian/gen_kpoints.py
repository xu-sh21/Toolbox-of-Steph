from pymatgen.io.vasp.inputs import Kpoints
import numpy as np
import matplotlib.pyplot as plt

# 设置 Monkhorst-Pack 网格的参数
mesh = [8, 8, 1]  # 在 x, y, z 方向上的网格点数

# 创建 Monkhorst-Pack 网格的 Kpoints 对象
kpoints = Kpoints.monkhorst_automatic(mesh)

# 获取所有 k 点的列表
kpoints_list = kpoints.kpts

# 只考虑二维 k 点（忽略 z 分量）
kpoints_2d = np.array(kpoints_list)[:, :2]

# 绘制二维布里渊区
fig, ax = plt.subplots()
# 绘制布里渊区边界
bz_boundaries = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5], [0, 0]])
ax.plot(bz_boundaries[:, 0], bz_boundaries[:, 1], 'k-')

# 绘制 k 点
for k in kpoints_2d:
    ax.plot(k[0], k[1], 'ro')

# 设置图表的坐标轴范围和标签
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)
ax.set_xlabel('kx')
ax.set_ylabel('ky')

# 显示图表
plt.show()
