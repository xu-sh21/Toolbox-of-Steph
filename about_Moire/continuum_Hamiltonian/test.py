# import numpy as np

# # # 假设你已经有了 num_path, density, 2 等变量
# # H_tot = np.zeros((3, 4, 2, 2))

# # # 定义一个函数，它接受i和j的索引，并返回对应的二维矩阵
# # def calculate_matrix(i, j):
# #     # 这里你需要定义如何根据i和j的值来计算矩阵
# #     # 以下是一个示例，假设矩阵是由i和j的值决定的
# #     matrix = np.ones((i,j))
# #     return matrix

# # # 创建一个包含所有i和j索引的数组
# # i_indices, j_indices = np.meshgrid(np.arange(3), np.arange(4), indexing='ij')

# # # 使用向量化的方法来计算所有矩阵
# # # np.vectorize 可以将函数向量化，这样我们就可以在数组上调用它


# # for i, j in enumerate(zip(i_indices.flatten(), j_indices.flatten())):
# #     print(f"i:{i}, j:{j}")

# l = [(1,2),(3,4),(5,6)]
# l_arr = np.array(l, dtype=int)
# s = np.zeros((2,3,4,2))
# s[:,:,0,:] = np.ones((2,), dtype=int)
# # print(s)
# s[:,:,1:,:] = l_arr
# print(s)
import numpy as np

# # 创建一个包含复数的列表
# complex_list = [1+2j, 3+4j, 5+6j]

# # 使用 numpy.array 将列表转换为数组，并指定 dtype 为 complex
# complex_array = np.array(complex_list, dtype=np.complex_)

# # 打印结果
# print(complex_array)

# # 查看数组的数据类型
# print(complex_array.dtype)


# 定义一个函数，根据行索引和列索引计算元素值
# def value_at(i, j):
#     # 示例规则：元素的值等于行索引的平方加上列索引
#     return i**2 + j

# # 矢量化函数
# vectorized_value_at = np.vectorize(value_at)

# # 创建一个 4x4 的矩阵，并应用赋值规则
# matrix = vectorized_value_at(np.arange(4), np.arange(4)[:, np.newaxis])

# # 打印结果
# print(matrix)
# # print(matrix.T)
# print(matrix[1][2])


# k_group_comb = np.array([[0,0],[1,0],[0,1],[1,1]])
# n_1, n_2 = k_group_comb[:, 0], k_group_comb[:, 1]
# m_1, m_2 = k_group_comb[:, 0], k_group_comb[:, 1]
# print(m_1[:, None])
# # 创建索引矩阵，用于比较 m 和 n 的值
# index_matrix = m_1[None, :] - n_1[:, None] + m_2[None, :] * 1j - n_2[:, None] * 1j
# print(index_matrix)
# # 定义所有可能的条件
# # conditions = {
# #     1: 1 + 0j,  # n_1 + 1 == m_1 and n_2 == m_2
# #     2: -1 + 1j, # n_1 - 1 == m_1 and n_2 + 1 == m_2
# #     3: 0 + 1j,  # n_1 == m_1 and n_2 + 1 == m_2
# #     4: -1 - 0j, # n_1 - 1 == m_1 and n_2 == m_2
# #     5: 1 - 1j,  # n_1 + 1 == m_1 and n_2 - 1 == m_2
# #     6: 0 - 1j   # n_1 == m_1 and n_2 - 1 == m_2
# # }
# conditions = {
#     1: 0 + 0j,  # n_1 == m_1 and n_2 == m_2
#     2: 0 - 1j, # n_1 == m_1 and n_2 - 1 == m_2
#     3: 1 - 1j,  # n_1 + 1 == m_1 and n_2 - 1 == m_2
# }

# # 初始化非对角元素为零
# H_nondiag = np.zeros((4,4))
# H_nondiag[~np.eye(4, dtype=bool)] = 0

# # 应用条件并填充非对角元素
# for condition, value in conditions.items():
#     H_nondiag[index_matrix == value] = 2 if condition in (1, 2, 3) else -2

# print(H_nondiag)

# import numpy as np

# 假设 H_tot 是一个四维数组，形状为 (num_paths, num_locations, num_states, num_states)
# 其中 num_states 是矩阵的大小，对于每个 [i, j, :, :]，它都是一个方阵

# 使用 np.lib.stride_tricks.as_strided 来创建一个二维数组，其中每个元素都是一个方阵
# 这个二维数组的形状将是 (num_paths * num_locations, num_states, num_states)
# H_tot = np.array([[[1., 0],
#          [0, 1.]],

#         [[2, 0],
#          [0, 3]],

#         [[4, 0],
#          [0, 5]],]
# , dtype=int)
#(2,3,2,2)
# strides = H_tot.strides
# new_shape = (H_tot.shape[0] * H_tot.shape[1], H_tot.shape[2], H_tot.shape[3])
# reshaped_H_tot = H_tot.reshape(-1, *H_tot.shape[2:])
# print(reshaped_H_tot.shape)
# # print(reshaped_H_tot[0,:,:])
# # 现在，我们可以计算所有方阵的本征值
# eigenvalues, eigenvecotrs = np.linalg.eig(H_tot)
# print(H_tot.shape)
# print(eigenvalues)
# print(eigenvecotrs.shape)
# # 如果需要，可以将本征值重新排列成原始形状的前两个维度
# eigenvalues = eigenvalues.reshape(H_tot.shape[0], H_tot.shape[1], -1)
# # print(eigenvalues.shape)
# eigenvecotrs = eigenvecotrs.reshape(H_tot.shape[0], H_tot.shape[1], H_tot.shape[2], H_tot.shape[3])
# # print(eigenvecotrs)


# This function is to convert the loop k points to a list of combinations.
# axis_arr = np.arange(-np.abs(2), np.abs(2) + 1)
# X, Y = np.meshgrid(axis_arr, axis_arr)
# group = np.vstack((X.ravel(), Y.ravel())).T
# print(group.shape)
# # print(X)
# # print(Y)
# print(group.shape)
# for i, [m_1, m_2] in enumerate(group):
#     print(m_1,m_2)

# a = np.array([3,2,4,6,2])
# b = [sum(a[:i]) for i in range(1,len(a))]
# print(b)

# c = [0,1,2,3,4,5,6,7,8,9,10]
# # c.remove(i) for i in [3,6,8]
# print(c)

# d = [*a, *c]
# print(d)

# H_nondiag = np.zeros((4,4))
# k_group_comb = np.array([[1,1],[1,2],[2,2],[3,4]])
# m_1, m_2 = k_group_comb[:, 0], k_group_comb[:, 1]
# n_1, n_2 = k_group_comb[:, 0], k_group_comb[:, 1]
# index_matrix = m_1[None, :] - n_1[:, None] + m_2[None, :] * 1j - n_2[:, None] * 1j

# print(index_matrix)

# conditions = {
#     1: 0 + 0j,  # n_1 == m_1 and n_2 == m_2
#     2: 0 - 1j, # n_1 == m_1 and n_2 - 1 == m_2
#     3: 1 - 1j,  # n_1 + 1 == m_1 and n_2 - 1 == m_2
# }
# # for condition, value in conditions.items():
# #     if condition in (1, 2, 3):
# #         H_nondiag[index_matrix == value] = 1
# #     elif condition in (4, 5, 6): 
# #         H_nondiag[index_matrix == value] = -1

# #     # H_nondiag[index_matrix == value] = 1 if condition in (1, 2, 3) else -1
# # print(H_nondiag)
# Delta_T = np.zeros((4,4))
# for condition, value in conditions.items():
#     Delta_T[index_matrix == value] = 1 

# print(Delta_T)


# a = np.zeros((4,2))
# b = np.array([[1,3],[1,1]])
# a[:,:] = b[:,None,:]
# print(a)

# t = np.linspace(0, 1, 5)
# X, Y = np.meshgrid(t, t)
# coef = np.vstack((X.ravel(), Y.ravel())).T
# print(coef)

# a = np.array([2,0])
# b = np.array([0,2])
# c = np.array([[2,0],[0,2]])
# r = coef[:,]@c
# print(r)

# original_array = np.array([[1, 2], [3, 4]])

# # 方法1: 使用 np.repeat
# # 在第二个维度上重复2次
# expanded_array_repeat = np.repeat(original_array[:, np.newaxis, :], 2, axis=1)


# print("使用 np.repeat 扩充的数组:")
# print(expanded_array_repeat)
# print(np.zeros(expanded_array_repeat.shape))

import numpy as np

# # 假设 A 是一个形状为 (n, m, m) 的三维数组
# n, m = 3, 2  # 示例维度
# A = np.array([[[1,2],[2,3]],[[3,4],[2,1]],[[4,2],[1,2]]])  # 随机生成 n 个 m×m 矩阵
# B = np.array([[[1,0],[0,-2]],[[0,4],[2,0]],[[4,0],[0,1]]])
# print(A.shape) #
# # 使用 numpy.linalg.eig 计算所有矩阵的本征值和本征向量
# # 结果会是一个包含本征值的数组和一个包含本征向量的数组
# eigenvalues, eigenvectors = np.linalg.eig(B)
# print(eigenvalues) #

# # eigenvalues 和 eigenvectors 的形状分别是 (3, 2) 和 (3, 2, 2)

# # 对每个方阵的特征值进行排序，得到排序后的索引
# # sorted_indices = np.argsort(eigenvalues, axis=1)

# # 使用高级索引重组特征向量和特征值
# # 对于特征值
# sorted_indices = np.argsort(eigenvalues, axis=1)
# print(sorted_indices)
# # print(eigenvectors)
# # 使用高级索引重组特征向量和特征值
# # 对于特征值
# sorted_eigenvalues = np.take_along_axis(eigenvalues, sorted_indices, axis=1).squeeze()
# print(sorted_eigenvalues.shape)
# print(eigenvectors)
# # 对于特征向量
# # 由于特征向量是三维数组，我们需要为每个矩阵的每个特征值创建一个索引数组
# indices_for_eigenvectors = np.arange(sorted_indices.shape[1])[:, np.newaxis]
# # print(indices_for_eigenvectors)
# sorted_eigenvectors = np.take_along_axis(eigenvectors, sorted_indices[:, np.newaxis, :], axis=2)
# print(sorted_eigenvectors)
# print(sorted_eigenvectors[:,:,-1])
# # 现在sorted_eigenvalues 和 sorted_eigenvectors 是按照特征值大小排序的

A = np.array([[13, 1],[2, 2+1j]],dtype=np.complex_)
print(A)
# print(A.conj().T)
# B = np.dot(A.conj(), A.T).T

A = np.delete(A, [1], axis=0)
print(A)