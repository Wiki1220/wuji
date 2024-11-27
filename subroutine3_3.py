import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def run_random_forest_model():
    data_path = 'data/data0_processed.csv'
    data = pd.read_csv(data_path)

    selected_columns = ['温度', '边界层高度', '地表气压', '降水量', '相对湿度',
                        '风速', '风向', '人口', '第二产业所占百分比',
                        '人均产出', '经度', '纬度', 'PM2.5浓度']

    analysis_data = data[selected_columns]
    analysis_data = analysis_data.dropna()

    X = analysis_data.drop(columns=['PM2.5浓度'])
    y = analysis_data['PM2.5浓度']

    X_train = X[:-104]
    y_train = y[:-104]
    X_val = X[-104:]
    y_val = y[-104:]

    print("\n开始随机森林模型训练...")

    tree_model = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
    tree_model.fit(X_train, y_train)

    y_pred_val = tree_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"验证集上的RMSE: {rmse:.2f}")

    joblib.dump(tree_model, 'tree_model.pkl')

    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': tree_model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\n特征重要性：")
    print(importance_df)

if __name__ == "__main__":
    run_random_forest_model()
