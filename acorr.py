import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def analyze_pm25_autocorrelation(data):
    # 读取数据
    data['Date'] = pd.to_datetime(data['日期'])
    data = data.set_index('Date')
    pm25_series = data['PM2.5浓度'].resample('D').mean()  # 按天取平均，计算时间序列
    
    # 去除缺失值
    pm25_series = pm25_series.dropna()

    # 时间自相关性分析
    lag_acf = sm.tsa.acf(pm25_series, nlags=30)

    # 自相关性数据
    print("Autocorrelation values:")
    for lag, acf_value in enumerate(lag_acf):
        print(f"Lag {lag}: {acf_value}")

    # 绘制自相关性图
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(lag_acf)), lag_acf)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of PM2.5 Concentration')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        
        import argparse
        parser = argparse.ArgumentParser(description="Analyze PM2.5 concentration autocorrelation.")
        parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
        args = parser.parse_args()
        
        data = pd.read_csv(args.data_path)
        analyze_pm25_autocorrelation(data)
    except SystemExit:
        # 默认参数
        data = pd.read_csv('data/data.csv')  
        analyze_pm25_autocorrelation(data)
