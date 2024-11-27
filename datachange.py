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



# 下面是另一个功能，一次性使用没封装


# # 假设你的数据已加载到DataFrame中
# data = pd.read_csv('data/mata.csv')

# # 合并“日期”和“时刻”列到新的“时间”列
# data['时间'] = pd.to_datetime(data['日期']) + pd.to_timedelta(data['时刻'], unit='h')

# data = data.drop(columns=['日期', '时刻'])

# # 重命名列名为英文
# data = data.rename(columns={
#     '城市': 'City',
#     '温度': 'Temperature',
#     '边界层高度': 'BL_Height',
#     '地表气压': 'Surface_Pressure',
#     '降水量': 'Precipitation',
#     '相对湿度': 'Humidity',
#     'U水平风速': 'U_WindSpeed',
#     'V方向风速': 'V_WindSpeed',
#     'PM2.5浓度': 'PM2_5',
#     '经度': 'Longitude',
#     '纬度': 'Latitude',
#     '时间': 'time',
# })

# # 将数据保存为CSV文件
# data.to_csv('endata.csv', index=False)

# # 输出结果，查看新数据
# print(data.head())

# 保留到日的数据，并加入整数索引。



df = pd.read_csv('data/endata.csv')

# 将'time'列转换为datetime格式
df['time'] = pd.to_datetime(df['time'])

# 筛选出每日0点的数据
df_00h = df[df['time'].dt.hour == 0]

# 计算'第一个时间点'距离当前行的天数
start_date = df_00h['time'].min()  # 获取最早的日期
df_00h['timeline'] = (df_00h['time'] - start_date).dt.days

# 查看处理后的数据
print(df_00h.head(20))

# 将处理后的数据保存为新文件
df_00h.to_csv('data/daydata.csv', index=False)
