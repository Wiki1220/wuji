import pandas as pd
from subroutine1_1 import analyze_pm25_by_quarter
from subroutine1_4 import compute_local_moran
from subroutine1_2 import analyze_pm25_autocorrelation
from subroutine1_3 import plot_pm25_bs
from subroutine1_5 import process_and_merge_data_with_normalization_and_pearson
data_path = 'data/data.csv'
location_path = 'data/location.csv'
data = pd.read_csv(data_path)
def main():
    
    # 计算节假日与PM2.5浓度的相关性，修改holiday为weekday可改为计算工作日
    params1 = [data, 2017, 1, 'holiday']  
    analyze_pm25_by_quarter(*params1) 

    # 计算PM2.5浓度的时间自相关度
    params2 = [data]
    analyze_pm25_autocorrelation(*params2)

    # 输出时间与浓度的关系图
    params3 = [data_path,location_path]
    plot_pm25_bs(data_path, location_path, font_path='C:\\Windows\\Fonts\\msyh.ttc')

    # 计算莫兰指数
    moran_I = compute_local_moran()
    print("Moran's I:", moran_I)

    # 空间因素的相关度,使用添加了海拔数据后的location
    process_and_merge_data_with_normalization_and_pearson(data_path,'data/location_additem.csv')

    
if __name__ == "__main__":
    main()