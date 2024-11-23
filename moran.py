import pandas as pd
import numpy as np

def calculate_moran_index(avg_values_file="data/mean.xlsx", distance_matrix_file="data/distance_matrix.xlsx"):
    # 读取均值数据
    avg_values_df = pd.read_excel(avg_values_file)
    # 读取距离矩阵数据
    distance_matrix_df = pd.read_excel(distance_matrix_file, index_col=0)

    # 获取城市平均值数据
    avg_values = avg_values_df['均值'].values

    # 将距离矩阵转化为空间权重矩阵，使用反距离加权法（距离越小权重越大）
    distance_matrix = distance_matrix_df.values

    # 避免除以零的问题，将对角线元素（自身距离）设置为 np.inf
    np.fill_diagonal(distance_matrix, np.inf)

    # 使用反距离作为权重，权重矩阵
    weight_matrix = 1 / distance_matrix
    weight_matrix[np.isinf(weight_matrix)] = 0  # 自身的权重设为0

    # 莫兰指数计算公式的各部分
    N = len(avg_values)  # 城市数量
    mean_value = np.mean(avg_values)  # 均值
    W = np.sum(weight_matrix)  # 权重矩阵的总和

    # 分子部分：∑∑ w_ij * (x_i - mean) * (x_j - mean)
    numerator = 0
    for i in range(N):
        for j in range(N):
            numerator += weight_matrix[i, j] * (avg_values[i] - mean_value) * (avg_values[j] - mean_value)

    # 分母部分：∑ (x_i - mean)^2
    denominator = np.sum((avg_values - mean_value) ** 2)

    # 莫兰指数 I 的计算
    moran_I = (N / W) * (numerator / denominator)

    # 返回莫兰指数的结果
    return moran_I

# 示例调用
if __name__ == "__main__":
    moran_I = calculate_moran_index()
    print("Moran's I:", moran_I)
