import pandas as pd
import os

# 假设原始数据集为df，路径为'data/original_data.csv'
df = pd.read_csv('data/Edata.csv')

# 创建文件保存路径
save_path = 'data/new/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 拆分数据集，并按城市名（A-M）命名子集
cities = df['城市'].unique()

# 遍历每个城市进行拆分处理
for city in cities:
    # 拆分数据集
    city_data = df[df['城市'] == city].copy()
    
    # 创建新的时间列：假设‘日期’和‘时刻’列格式为‘YYYY-MM-DD’和‘HH’
    city_data['时间'] = pd.to_datetime(city_data['日期'].astype(str) + ' ' + city_data['时刻'].astype(str) + ':00')

    # 添加索引列
    city_data['索引'] = range(len(city_data))

    # 删除原来的列：‘日期’，‘时刻’，‘城市’
    city_data = city_data.drop(columns=['日期', '时刻', '城市'])
    
    # 文件名命名规则：以城市名首字母大写作为文件名
    file_name = f'datafor{city[0].upper()}.csv'
    
    # 保存数据到指定目录，如果同名则覆盖
    city_data.to_csv(os.path.join(save_path, file_name), index=False)
    
    print(f'数据已保存：{file_name}')
