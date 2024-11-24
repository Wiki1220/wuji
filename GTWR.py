import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from mgtwr.model import GTWR, MGTWR,GTWRResults
from mgtwr.sel import SearchGTWRParameter
from mgtwr.model import GTWR
from sklearn.preprocessing import StandardScaler
import joblib
data = pd.read_csv('data/endata2017.csv')
data['time'] = pd.to_datetime(data['time']).view(np.int64) // 10**9
coords = data[['Longitude', 'Latitude']]
t = data[['time']]
# X = data[['Temperature','BL_Height','Surface_Pressure','Precipitation','Humidity','U_WindSpeed','V_WindSpeed']]
X = data[['Temperature','BL_Height','Surface_Pressure','Humidity','U_WindSpeed','V_WindSpeed']]
# X = data[['U_WindSpeed','V_WindSpeed']]
y = data[['PM2_5']]#因变量
# sel = SearchGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
# bw, tau = sel.search(tau_max=20, verbose=True, time_cost=True)
bw=5.9
tau=7.6
gtwr = GTWR(coords, t, X, y, bw, tau, kernel='gaussian', fixed=True).fit()
print(gtwr.R2)
# joblib.dump(gtwr, 'GTWR_model2017.pkl')-

##SearchGTWRParameter：用于搜索 GTWR 模型的最佳参数。
##coords, t, X, y：分别是空间位置、时间变量、自变量和因变量。
##kernel='gaussian'：使用高斯核函数。
##fixed=True：使用固定带宽。
##tau_max=20：搜索的最大时间延迟。
##verbose=True：显示搜索过程的详细信息。
##time_cost=True：显示搜索所需时间。
##GTWR：创建 GTWR 模型实例。
##fit()：拟合模型。
##gtwr.R2：输出模型的判定系数 R2R2，表示模型的解释力。