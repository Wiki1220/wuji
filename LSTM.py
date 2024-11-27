import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input # type: ignore
import shap

# 读取数据
data = pd.read_csv('data/endata.csv')
data['time'] = pd.to_datetime(data['time'])

# 处理城市列，使用one-hot编码
data_encoded = pd.get_dummies(data, columns=['City'])

# 提取时间特征
data_encoded['year'] = data_encoded['time'].dt.year
data_encoded['month'] = data_encoded['time'].dt.month
data_encoded['day'] = data_encoded['time'].dt.day
data_encoded['hour'] = data_encoded['time'].dt.hour
data_encoded['weekday'] = data_encoded['time'].dt.weekday

# 删除原始时间列
data_encoded = data_encoded.drop('time', axis=1)

# 将目标变量设置为PM2_5，其他为特征
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

# 创建LSTM模型
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))  # 输出一个预测值

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
model.save('lstm_model.h5')
# 预测并反转标准化
predictions = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions)


test_loss = model.evaluate(X_test, y_test)
print(f"测试集损失：{test_loss}")

