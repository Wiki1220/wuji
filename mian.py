# main.py

import pandas as pd
from holiday import analyze_pm25_by_quarter  # 导入子模块中的分析函数

def main():
    # 读取数据，这里假设数据文件路径为 'data/data.csv'
    data_path = 'data/data.csv'
    data = pd.read_csv(data_path)
    
    # 设置参数，例如要分析的年份和季度
    year = 2017
    quarter = 1
    holiday_mode = 'holiday'  # 可以是 'weekday' 或 'holiday'

    # 调用子模块中的分析函数
    analyze_pm25_by_quarter(data, year, quarter, holiday_mode)

if __name__ == "__main__":
    main()
