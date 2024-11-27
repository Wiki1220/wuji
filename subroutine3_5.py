import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from statsmodels.tsa.stattools import adfuller
# 假设数据已经存储在一个名为 'data' 的DataFrame中，并且有一个名为 '浓度' 的列
data = pd.read_csv('data/new/dataforD.csv', parse_dates=['时间'], index_col='时间')

# 选择时间序列数据列
time_series = data['浓度']

# 绘制ACF图，设置最大滞后期为20
plt.figure(figsize=(10, 6))  # 设置图形大小
plot_acf(time_series, lags=20)
plt.title('ACF Plot')
plt.show()

# 绘制PACF图，设置最大滞后期为20
plt.figure(figsize=(10, 6))  # 设置图形大小
plot_pacf(time_series, lags=20)
plt.title('PACF Plot')

# 进行ADF检验
adf_result = adfuller(time_series)

# 输出结果
print(f"ADF检验的统计量: {adf_result[0]}")
print(f"p值: {adf_result[1]}")
print(f"滞后数: {adf_result[2]}")
print(f"样本数量: {adf_result[3]}")
print(f"临界值: {adf_result[4]}")
