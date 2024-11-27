import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

# 定义函数：计算 PM2.5 均值、处理 x、y 和 yh 分类变量并计算点二列相关系数及显著性，加入 hb、x、y 的 Pearson 相关性
def process_and_merge_data_with_normalization_and_pearson(pm25_file, hb_file, merge_column='城市', how='left'):
    """
    处理数据：计算 PM2.5 均值，处理 x、y 和 yh 分类变量，归一化 PM2.5，计算点二列相关系数以及 Pearson 相关系数及显著性。
    
    参数:
    pm25_file (str): 包含 PM2.5 数据的 CSV 文件路径
    hb_file (str): 包含 hb 数据的 CSV 文件路径
    merge_column (str): 用于合并的共同列，默认为 '城市'
    how (str): 合并方式，默认为 'left'，可选的有 'left'、'right'、'inner'、'outer'
    
    返回:
    pd.DataFrame: 处理后的合并数据集及相关性分析结果
    """
    # 1. 读取 PM2.5 数据并计算均值
    pm25_data = pd.read_csv(pm25_file)
    
    # 复制数据以避免修改原数据集
    pm25_data_copy = pm25_data.copy()
    
    # 确保日期列为 datetime 类型
    pm25_data_copy['日期'] = pd.to_datetime(pm25_data_copy['日期'], format='%Y/%m/%d', errors='raise')
    
    # 按 '日期' 和 '城市' 分组，计算每个标签的 PM2.5 均值
    grouped_pm25 = pm25_data_copy.groupby(['日期', '城市'])['PM2.5浓度'].mean().reset_index()
    
    # 按照日期排序
    grouped_pm25 = grouped_pm25.sort_values(by='日期')

    # 2. 读取 hb 数据并处理 x、y、yh 分类变量
    hb_data = pd.read_csv(hb_file)
    
    # 复制数据以避免修改原数据集
    hb_data_copy = hb_data.copy()
    
    # 将 x、y 和 yh 变量转化为二分类（0 和 1）
    for column in ['经度', '纬度', '沿海']:
        # 根据均值进行分类：比均值小赋值为 0，比均值大赋值为 1
        mean_value = hb_data_copy[column].mean()
        hb_data_copy[column] = hb_data_copy[column].apply(lambda val: 0 if val < mean_value else 1)
    
    # 提取需要合并的列（城市 和 分类后的 经度, 纬度, 沿海 和 海拔）
    result2 = hb_data_copy[['城市', '经度', '纬度', '沿海', '海拔']]

    # 3. 合并 PM2.5 数据和分类变量数据
    merged_result = pd.merge(grouped_pm25, result2, on=merge_column, how=how)
    
    # 4. 对 PM2.5 数据进行归一化
    scaler = MinMaxScaler()
    merged_result['PM2.5_normalized'] = scaler.fit_transform(merged_result[['PM2.5浓度']])
    
    # 5. 计算点二列相关系数并输出相关性和显著性
    results = {}
    for column in ['经度', '纬度', '沿海']:
        # 计算点二列相关系数
        corr, p_value = stats.pointbiserialr(merged_result[column], merged_result['PM2.5_normalized'])
        
        # 存储结果
        results[column] = {'Point-biserial Correlation': corr, 'p-value': p_value}
        
        # 输出相关性和显著性
        print(f"{column}与归一化后的PM2.5的点二列相关系数: {corr:.4f}")
        print(f"{column}与归一化后的PM2.5的p-value: {p_value:.4f}")
        
        # 判断显著性
        if p_value < 0.05:
            print(f"{column}与归一化后的PM2.5的相关性显著（p < 0.05）")
        else:
            print(f"{column}与归一化后的PM2.5的相关性不显著（p >= 0.05）")
        print("-" * 50)

    # 6. 计算连续变量（海拔、经度、纬度）与归一化后的 PM2.5 的 Pearson 相关系数
    continuous_columns = ['海拔', '经度', '纬度']
    for column in continuous_columns:
        # 计算 Pearson 相关系数
        corr, p_value = stats.pearsonr(merged_result[column], merged_result['PM2.5_normalized'])
        
        # 存储结果
        results[column] = {'Pearson Correlation': corr, 'p-value': p_value}
        
        # 输出相关性和显著性
        print(f"{column}与归一化后的PM2.5的Pearson相关系数: {corr:.4f}")
        print(f"{column}与归一化后的PM2.5的p-value: {p_value:.4f}")
        
        # 判断显著性
        if p_value < 0.05:
            print(f"{column}与归一化后的PM2.5的相关性显著（p < 0.05）")
        else:
            print(f"{column}与归一化后的PM2.5的相关性不显著（p >= 0.05）")
        print("-" * 50)

    # 返回合并后的结果和相关性分析的字典
    return merged_result, results

# 示例使用函数：
pm25_file = 'data/data.csv'  # 输入 PM2.5 数据的 CSV 文件路径
hb_file = 'data/location_additem.csv'  # 输入 hb 数据的 CSV 文件路径

# 调用函数并获取合并结果
result, correlation_results = process_and_merge_data_with_normalization_and_pearson(pm25_file, hb_file)

# 打印合并后的结果（显示在右侧数据模块）
print(result.head())