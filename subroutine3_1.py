import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mgtwr.model import GTWR

def run_gtwr_model():
    # 读取数据
    data = pd.read_csv('data/endata21.csv')
    data['time'] = pd.to_datetime(data['time']).astype(np.int64) // 10**9  # Convert to Unix timestamp in seconds

    coords = data[['Longitude', 'Latitude']]
    t = data[['time']]

    X = data[['Temperature', 'BL_Height', 'Surface_Pressure', 'Humidity', 'U_WindSpeed', 'V_WindSpeed', '海拔', '沿海']]
    y = data[['PM2_5']]

    X_train, X_val = X[:-104], X[-104:]
    y_train, y_val = y[:-104], y[-104:]
    coords_train, coords_val = coords[:-104], coords[-104:]
    t_train, t_val = t[:-104], t[-104:]

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # 设置 GTWR 模型的带宽和时间衰减参数
    bw = 5.9
    tau = 7.6

    # 创建 GTWR 模型并拟合
    gtwr = GTWR(coords_train, t_train, X_train_scaled, y_train_scaled, bw, tau, kernel='gaussian', fixed=True)
    gtwr_fit = gtwr.fit()
    print("R2 score:", gtwr_fit.R2)

    # 获取训练集上的所有预测值
    y_train_pred_scaled = gtwr_fit.predict_value

    # 将预测值逆变换回原始 PM2.5 数值
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1))

    # 计算最后104个验证集样本的 RMSE
    y_val_true = y_val.values  # 真实的 PM2.5 浓度
    y_val_pred = y_train_pred[-104:]  # 只获取验证集的预测值

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    print("RMSE:", rmse)

    # 将预测结果保存到 CSV 文件
    y_val_pred_df = pd.DataFrame(y_val_pred, columns=['Predicted_PM2_5'])
    y_val_pred_df.to_csv('y_val_pred.csv', index=False)

if __name__ == "__main__":
    run_gtwr_model()
