from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datachange import split_city_data, add_time_index
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

def arima_analysis(data_path, city_key):

    # 获取不同城市的数据
    city_datasets = split_city_data(data_path)

    # 获取指定城市的数据并添加时间索引
    if city_key in city_datasets:
        data = city_datasets[city_key]
        data = add_time_index(data)
    else:
        print(f"城市 '{city_key}' 数据不存在")
        return

    # 确保时间索引列为时间类型，并按时间排序
    data['时间索引'] = pd.to_datetime(data['时间索引'], errors='coerce')
    data = data.sort_values('时间索引')



    # 设置时间索引为索引
    data = data.set_index('时间索引')
    
    # 选择PM2.5浓度列作为目标变量
    target_series = data['PM2.5浓度']
    
    # 使用auto_arima自动选择最优的p, d, q参数
    print("正在自动选择ARIMA模型的最优参数...")
    model_auto = auto_arima(target_series, seasonal=False, stepwise=True, trace=True, suppress_warnings=True)
    
    # 打印自动选择的最优参数
    print(f"自动选择的最佳ARIMA参数: (p, d, q) = {model_auto.order}")
    
    # 使用自动选择的参数训练ARIMA模型
    model = ARIMA(target_series, order=model_auto.order)
    model_fit = model.fit()

    # 打印模型摘要
    print(model_fit.summary())
    


# 示例调用
arima_analysis('data/data.csv', 'A')
