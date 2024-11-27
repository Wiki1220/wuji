import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('data/endata.csv')
data['time'] = pd.to_datetime(data['time'])

# 处理城市列，使用 one-hot 编码
data_encoded = pd.get_dummies(data, columns=['City'])

# 提取时间特征
data_encoded['year'] = data_encoded['time'].dt.year
data_encoded['month'] = data_encoded['time'].dt.month
data_encoded['day'] = data_encoded['time'].dt.day
data_encoded['hour'] = data_encoded['time'].dt.hour
data_encoded['weekday'] = data_encoded['time'].dt.weekday

# 删除原始时间列
data_encoded = data_encoded.drop('time', axis=1)

# 将目标变量设置为 PM2_5，其他为特征
X = data_encoded.drop(columns=['PM2_5'])  # 删除目标列
y = data_encoded['PM2_5']  # 目标变量

# 标准化特征数据
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # 需要将 y 转换为二维数组

# 转换为时间序列的窗口数据
def create_dataset(X, y, time_step=1):
    X_data, y_data = [], []
    for i in range(len(X) - time_step):
        X_data.append(X[i:(i + time_step), :])  # 每次取 time_step 个连续的样本
        y_data.append(y[i + time_step, 0])  # 目标值是下一个时间点的值
    return np.array(X_data), np.array(y_data)

time_step = 3  # 你可以根据需要调整时间步长
X_data, y_data = create_dataset(X_scaled, y_scaled, time_step)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)

# 加载已经训练好的 LSTM 模型
model = load_model('lstm_model.h5')

# 计算 LSTM 特征重要性
def compute_lstm_feature_importance(model, X_valid, y_valid, COLS):
    results = []
    for k in range(X_valid.shape[2]):  # 遍历每个特征
        if k > 0:
            save_col = X_valid[:, :, k-1].copy()
            np.random.shuffle(X_valid[:, :, k-1])  # 随机打乱当前特征列

        # 使用模型进行预测
        oof_preds = model.predict(X_valid, verbose=0).squeeze()
        mae_value = np.mean(np.abs(oof_preds - y_valid))  # 计算MAE
        results.append({'feature': COLS[k], 'mae': mae_value})

        if k > 0:
            X_valid[:, :, k-1] = save_col  # 恢复原来的特征列

    # 将结果按 MAE 排序并返回
    df = pd.DataFrame(results)
    df = df.sort_values('mae')
    return df

# 特征列名称
COLS = X.columns.tolist()

# 计算特征重要性
feature_importance_df = compute_lstm_feature_importance(model, X_test, y_test, COLS)

# 展示特征重要性
print("LSTM Feature Importance:")
print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 20))
plt.barh(np.arange(len(COLS)), feature_importance_df.mae)
plt.yticks(np.arange(len(COLS)), feature_importance_df.feature.values)
plt.title('LSTM Feature Importance', size=16)
plt.ylim((-1, len(COLS)))
plt.show()

# 保存特征重要性结果
feature_importance_df.to_csv('lstm_feature_importance.csv', index=False)
