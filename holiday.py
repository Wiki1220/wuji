import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.stats as stats
import argparse

def analyze_pm25_by_quarter(data, year, quarter, holiday_mode='weekday'):
    # 数据预处理
    data['Date'] = pd.to_datetime(data['日期'])
    data['Year'] = data['Date'].dt.year
    data['Quarter'] = data['Date'].dt.quarter if quarter != 0 else 0
    data['Weekday'] = data['Date'].dt.dayofweek
    
    # 设置是否工作日
    holidays = [  # 添加法定节假日（示例数据）
        '2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30',
        '2017-04-02', '2017-04-03', '2017-04-04',
        '2017-05-01', '2017-05-28', '2017-05-29', '2017-05-30',
        '2017-10-01', '2017-10-02', '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07'
    ]
    holidays = pd.to_datetime(holidays)
    
    if holiday_mode == 'weekday':
        # 使用周一到周五为工作日的标记方式
        data['Is_Workday'] = data['Weekday'].apply(lambda x: 0 if x >= 5 else 1)
    elif holiday_mode == 'holiday':
        # 使用法定节假日的标记方式，将法定节假日标记为非工作日
        data['Is_Workday'] = data['Date'].apply(lambda x: 0 if x in holidays else (1 if x.weekday() < 5 else 0))
    
    # 过滤数据，截取指定年和季度的数据
    if quarter == 0:
        quarter_data = data[data['Year'] == year].copy()
    else:
        quarter_data = data[(data['Year'] == year) & (data['Quarter'] == quarter)].copy()
    
    # 按照天进行PM2.5数据的聚合，计算每日均值
    daily_pm25 = quarter_data.groupby('Date')['PM2.5浓度'].mean().reset_index()
    daily_pm25 = pd.merge(daily_pm25, quarter_data[['Date', 'Is_Workday', 'Weekday']].drop_duplicates(), on='Date')

    # 绘制折线图，根据是否工作日、节假日和周末来区分颜色
    plt.figure(figsize=(10, 6))
    colors = []
    for _, row in daily_pm25.iterrows():
        if row['Date'] in holidays:
            colors.append('green')  # 节假日标记为绿色
        elif row['Weekday'] >= 5:
            colors.append('red')  # 周末标记为红色
        else:
            colors.append('blue')  # 工作日标记为蓝色

    plt.scatter(daily_pm25['Date'], daily_pm25['PM2.5浓度'], c=colors, label='PM2.5 Concentration')
    plt.plot(daily_pm25['Date'], daily_pm25['PM2.5浓度'], color='gray', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('PM2.5 Concentration')
    plt.title(f"Daily PM2.5 Concentration in {year}{' Q' + str(quarter) if quarter != 0 else ''}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 相关性分析
    if holiday_mode == 'weekday':
        # 工作日与PM2.5浓度的相关性分析
        corr_coef, p_value = stats.pearsonr(quarter_data['Is_Workday'], quarter_data['PM2.5浓度'])
        print(f"Correlation coefficient between Workday and PM2.5 Concentration: {corr_coef}")
        if p_value < 0.05:
            print("Workday and PM2.5 concentration are significantly correlated (p < 0.05)")
        else:
            print("Workday and PM2.5 concentration are not significantly correlated (p >= 0.05)")
    elif holiday_mode == 'holiday':
        # 节假日与PM2.5浓度的相关性分析
        quarter_data['Is_Holiday'] = quarter_data['Date'].apply(lambda x: 1 if x in holidays else 0)
        corr_coef, p_value = stats.pearsonr(quarter_data['Is_Holiday'], quarter_data['PM2.5浓度'])
        print(f"Correlation coefficient between Holiday and PM2.5 Concentration: {corr_coef}")
        if p_value < 0.05:
            print("Holiday and PM2.5 concentration are significantly correlated (p < 0.05)")
        else:
            print("Holiday and PM2.5 concentration are not significantly correlated (p >= 0.05)")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Analyze PM2.5 concentration by year and quarter.")
        parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
        parser.add_argument('--year', type=int, required=True, help='Year to analyze.')
        parser.add_argument('--quarter', type=int, required=True, help='Quarter to analyze (0 for full year).')
        parser.add_argument('--holiday_mode', type=str, choices=['weekday', 'holiday'], default='weekday', help='Holiday mode to use (weekday or holiday).')
        
        args = parser.parse_args()
        
        # 读取数据
        data = pd.read_csv(args.data_path)
        
        # 调用分析函数
        analyze_pm25_by_quarter(data, year=args.year, quarter=args.quarter, holiday_mode=args.holiday_mode)
    except SystemExit:
        # 当命令行参数未提供时，直接运行默认参数
        data = pd.read_csv('data/data.csv')  
        analyze_pm25_by_quarter(data, year=2017, quarter=1, holiday_mode='holiday')
