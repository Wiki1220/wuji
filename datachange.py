import pandas as pd

def split_city_data(file_path):
    # 读取数据文件，假设数据格式为CSV
    data = pd.read_csv(file_path)
    
    # 创建独立的城市数据集
    city_names = data['城市'].unique()  # 获取所有城市名称，例如 A, B, ..., M

    # 使用字典存储各城市的数据
    city_datasets = {}
    for city in city_names:
        city_datasets[city] = data[data['城市'] == city].copy()  # 筛选每个城市的数据并复制

    # 将字典中的数据集赋给单独的变量
    for city in city_names:
        globals()[f"data{city}"] = city_datasets[city]  # 动态生成变量，例如 dataA, dataB, ..., dataM

    # 返回城市数据集字典
    return city_datasets

def add_time_index(dataframe, start_date='2017-01-01', freq='3h'):
    # 创建一个DatetimeIndex作为时间索引，时间间隔为三小时
    datetime_index = pd.date_range(start=start_date, periods=len(dataframe), freq=freq)
    
    # 直接加入时间索引列
    dataframe['时间索引'] = datetime_index
    
    # 返回添加时间索引后的DataFrame
    return dataframe

