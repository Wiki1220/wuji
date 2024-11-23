import pandas as pd
from holiday import analyze_pm25_by_quarter
from acorr import analyze_pm25_autocorrelation
data_path = 'data/data.csv'
data = pd.read_csv(data_path)
def main():
    params1 = [data, 2017, 1, 'holiday']  # 参数顺序为 data, year, quarter, holiday_mod
    analyze_pm25_by_quarter(*params1)  
    
    params2 = [data]
    analyze_pm25_autocorrelation(*params2)
if __name__ == "__main__":
    main()
