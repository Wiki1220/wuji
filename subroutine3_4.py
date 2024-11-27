from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from datachange import split_city_data, add_time_index
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

def run_arima_analysis():
    data_path = 'data/data.csv'
    city_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    validation_size = 8

    city_datasets = split_city_data(data_path)
    rmse_list = []

    for city_key in city_keys:
        if city_key in city_datasets:
            data = city_datasets[city_key]
            data = add_time_index(data)
        else:
            print(f"城市 '{city_key}' 数据不存在")
            continue

        data['时间索引'] = pd.to_datetime(data['时间索引'], errors='coerce')
        data = data.sort_values('时间索引')
        data = data.set_index('时间索引')
        
        target_series = data['PM2.5浓度']
        target_series = target_series.tail(365)

        train_data = target_series[:-validation_size]
        validation_data = target_series[-validation_size:]

        model_auto = auto_arima(
            train_data,
            seasonal=True,
            m=12,
            stepwise=True,
            trace=True,
            suppress_warnings=True,
            max_p=2,
            max_q=2,
            max_order=6
        )

        model = ARIMA(train_data, order=model_auto.order, seasonal_order=model_auto.seasonal_order)
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=validation_size)
        rmse = mean_squared_error(validation_data, forecast, squared=False)
        print(f"城市 {city_key} 的RMSE: {rmse}")
        rmse_list.append(rmse)
    
    average_rmse = sum(rmse_list) / len(rmse_list)
    print(f"\n所有城市RMSE的均值: {average_rmse}")

if __name__ == "__main__":
    run_arima_analysis()
