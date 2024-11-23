from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datachange import split_city_data
from datachange import add_time_index
data_path=('data/data.csv')
city_datasets=split_city_data(data_path)
dataA = city_datasets['A']
dataA=add_time_index(dataA)

def arima_analysis(dataframe):
    # 确保时间索引列为时间类型
    dataframe = dataframe.set_index('时间索引')
    
    # 选择PM2.5浓度列作为目标变量
    target_series = dataframe['PM2.5浓度']
    
    # 拟合ARIMA模型
    model = ARIMA(target_series, order=(1, 1, 1))  # 这里的 (1, 1, 1) 可以根据需要进行调整
    model_fit = model.fit()
    
    # 打印模型摘要
    print(model_fit.summary())
    
    # 绘制预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(target_series, label='Observed PM2.5 Concentration')
    plt.plot(model_fit.fittedvalues, color='red', label='Fitted Values')
    plt.legend()
    plt.title('ARIMA Model Fit for PM2.5 Concentration')
    plt.xlabel('Time Index')
    plt.ylabel('PM2.5 Concentration')
    plt.show()

arima_analysis(dataA)