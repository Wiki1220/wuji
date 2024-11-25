import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from mgtwr.model import GTWR, MGTWR,GTWRResults
from mgtwr.sel import SearchGTWRParameter
from mgtwr.model import GTWR
from sklearn.preprocessing import StandardScaler
import joblib
data = pd.read_csv('data/endatamini.csv')
data['time'] = pd.to_datetime(data['time']).view(np.int64) // 10**9
coords = data[['Longitude', 'Latitude']]
t = data[['time']]
# X = data[['Temperature','BL_Height','Surface_Pressure','Precipitation','Humidity','U_WindSpeed','V_WindSpeed']]
X = data[['Temperature','BL_Height','Surface_Pressure','Humidity','U_WindSpeed','V_WindSpeed']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardizing X
y = data[['PM2_5']]#因变量
y_scaled = scaler.fit_transform(y) 
# sel = SearchGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
# bw, tau = sel.search(tau_max=20, verbose=True, time_cost=True)
bw=5.9
tau=7.6
gtwr = GTWR(coords, t, X_scaled, y_scaled, bw, tau, kernel='gaussian', fixed=True).fit()
print(gtwr.R2)

# joblib.dump(gtwr, 'GTWR_model2017.pkl')
# 后续过程，打印系数随时间变化的趋势图

import matplotlib.pyplot as plt



# 将时间字段转化为日期格式
data = pd.read_csv('data/endatamini.csv')
data['time'] = pd.to_datetime(data['time'])

# 获取 GTWR 模型的系数
coefficients = gtwr.betas  # 获取系数

# 获取时间字段
time = np.unique(data['time'])  # 提取唯一时间点

# 定义变量名
variable_names = ['Temperature', 'BL_Height', 'Surface_Pressure', 'Precipitation', 'Humidity', 'U_WindSpeed', 'V_WindSpeed']


coeff_df = pd.DataFrame(columns=['Time'] + variable_names)

# 遍历每个时间点，提取对应的系数
for i, t in enumerate(time):
    row = [t]  # 首列为时间
    for var_index in range(len(variable_names)):
        row.append(coefficients[i, var_index])  # 将每个系数添加到行中
    coeff_df.loc[i] = row  # 将每一行添加到 DataFrame
print(coeff_df.head(100))




variable_index = variable_names.index('Temperature')#修改这里看其他变量
main_coeff = coefficients[:, variable_index]

longitudes = data['Longitude']
latitudes = data['Latitude']

# 创建 GeoDataFrame 来绘制
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(longitudes, latitudes))

gdf['main_coeff'] = main_coeff
# fig, ax = plt.subplots(figsize=(10, 6))

# # 选择一种颜色映射来表示系数的大小


# 使用 GeoPandas 绘制地图
# gdf.plot(column='main_coeff', cmap='coolwarm', ax=ax, legend=True,
#          legend_kwds={'label': "Temperature Coefficient by Location",
#                       'orientation': "horizontal"})
# ax.set_title('Spatial Variation of Temperature Coefficients')
# plt.show()

output_data = gdf[['Longitude', 'Latitude', 'main_coeff','time']]

print(output_data.head(17))