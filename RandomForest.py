import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 加载数据
data_path = 'data/data0_processed.csv'
data = pd.read_csv(data_path)

# 选择需要的变量
selected_columns = ['温度', '边界层高度', '地表气压', '降水量', '相对湿度',
                    '风速', '风向', '人口', '第二产业所占百分比',
                    '人均产出', '经度', '纬度', 'PM2.5浓度']

# 筛选数据
analysis_data = data[selected_columns]

# 检查并处理缺失值
analysis_data = analysis_data.dropna()

# 定义特征和目标变量
X = analysis_data.drop(columns=['PM2.5浓度'])
y = analysis_data['PM2.5浓度']

# ----------------------随机森林模型训练----------------------
print("\n开始随机森林模型训练...")

# 初始化并训练随机森林模型
tree_model = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
tree_model.fit(X, y)

# 预测结果
y_pred_tree = tree_model.predict(X)

# 评估模型
print(f"随机森林 R²: {r2_score(y, y_pred_tree):.2f}")
print(f"随机森林均方误差 (MSE): {mean_squared_error(y, y_pred_tree):.2f}")

# 保存模型
joblib.dump(tree_model, 'tree_model.pkl')

# 输出特征重要性
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': tree_model.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\n特征重要性：")
print(importance_df)
