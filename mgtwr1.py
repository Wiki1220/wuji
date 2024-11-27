import pandas as pd
import numpy as np
from mgtwr.model import MGTWR  # 修改为导入MGtwr模型
from sklearn.preprocessing import StandardScaler
from mgtwr.sel import SearchMGTWRParameter
# 读取数据
data = pd.read_csv('data/data0.csv')

# 选择时刻为0的数据
data['日期'] = pd.to_datetime(data['日期'])
data = data[(data['日期'].dt.day == 1) & (data['时刻'] == 0)]
print(data.shape)
# 将日期转化为时间戳（单位：秒）
data['日期'] = pd.to_datetime(data['日期']).astype(np.int64) // 10**9  # 转换为秒级时间戳

# 提取经纬度坐标和时间
coords = data[['经度', '纬度']]  # 经纬度
t = data[['日期']]  # 时间

# 对降水量做平移处理，避免为0的情况
data['降水量'] = data['降水量'] + 1

# 选择自变量和因变量
X = data[['温度', '边界层高度', '地表气压', '相对湿度',  '风速', '风向', '海拔']]
y = data[['PM2.5浓度']]  # 因变量

# 标准化处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)  # 标准化自变量
y_scaled = scaler_y.fit_transform(y)  # 标准化因变量

# 设置mGTWR模型的带宽和时间衰减参数（可以通过参数搜索来优化）
# bw = 5.9  # 这是经过搜索或调整得出的带宽
# tau = 7.6  # 时间衰减参数
sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
bws = sel_multi.search(multi_bw_min=[0.1], verbose=True, tol_multi=0.01)
# 初始化并拟合mGTWR模型（修改为MGtwr）
mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel='gaussian', fixed=True).fit()

# 获取模型的系数（通过mgtwr.betas获取系数）
betas = mgtwr.betas  # betas是一个数组，包含每个时间点对应的系数

# 检查betas和时间的形状
print(f"Shape of betas: {betas.shape}")
print(f"Shape of t['日期']: {t['日期'].shape}")

# 创建系数表，将每个时间点的系数与日期、经纬度一起输出
coeff_list = []

# 将系数按时间和自变量列出
for i, time in enumerate(t['日期']):
    coeff = betas[i, :]  # 每个时间点的系数
    coeff_dict = {
        '日期': pd.to_datetime(time, unit='s'),  # 转换回日期格式
        '经度': coords.iloc[i]['经度'],  # 经纬度的提取，这里假设每个时间点的坐标是唯一的
        '纬度': coords.iloc[i]['纬度']
    }
    coeff_dict.update(dict(zip(X.columns, coeff)))  # 添加系数到字典中
    coeff_list.append(coeff_dict)

# 将系数表转为DataFrame
coeff_df = pd.DataFrame(coeff_list)

# 保存到CSV文件
coeff_df.to_csv('variable_coefficients_with_coords_mgtwr.csv', index=False)

# 打印输出系数表
print(coeff_df)
