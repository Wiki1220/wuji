import pandas as pd
from holiday import analyze_pm25_by_quarter
from acorr import analyze_pm25_autocorrelation
from bs import plot_pm25_bs
data_path = 'data/data.csv'
location_path = 'data/location.csv'
data = pd.read_csv(data_path)
def main():
    
    params1 = [data, 2017, 1, 'holiday']  
    analyze_pm25_by_quarter(*params1)  
    
    params2 = [data]
    analyze_pm25_autocorrelation(*params2)

    params3 = [data_path,location_path]
    plot_pm25_bs(data_path, location_path, font_path='C:\\Windows\\Fonts\\msyh.ttc')
if __name__ == "__main__":
    main()