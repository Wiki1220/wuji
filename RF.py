import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import chardet

# 设置字体
plt.rc('font', family='Microsoft YaHei')

# 自动检测文件编码并加载数据
def load_data(file_path):
    try:
        # 检测编码
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        # 加载数据
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"数据加载成功 (编码: {encoding})")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

# 加载数据
data_path = 'data/data0_processed.csv'
data = load_data(data_path)

# 如果数据成功加载，进行分析
if data is not None:
    # 选择变量（假设 PM2.5 的列名为 'PM2.5浓度'）
    selected_columns = ['温度', '边界层高度', '地表气压', '降水量', '相对湿度',
                        '风速', '风向', '人口', '第二产业所占百分比',
                        '人均产出', '经度', '纬度', 'PM2.5浓度']

    # 检查数据是否包含选定列
    missing_columns = [col for col in selected_columns if col not in data.columns]
    if missing_columns:
        print(f"以下列在数据中缺失: {missing_columns}")
    else:
        # 筛选分析所需的列
        analysis_data = data[selected_columns]

        # 检查缺失值并处理
        if analysis_data.isnull().sum().any():
            print("数据包含缺失值，进行处理...")
            analysis_data = analysis_data.dropna()  # 简单处理方式：删除缺失值
            print("缺失值处理完成")
        else:
            print("数据无缺失值")

        # 数据描述
        print("数据的基本信息：")
        print(analysis_data.info())
        print(analysis_data.describe())

        # 计算相关系数
        correlation_matrix = analysis_data.corr(method='pearson')
        print("相关系数矩阵：")
        print(correlation_matrix)

        # 绘制相关系数热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap", fontsize=16)
        plt.show()

        # ----------------------线性回归分析----------------------
        print("\n开始线性回归分析...")
        X = analysis_data.drop(columns=['PM2.5浓度'])
        y = analysis_data['PM2.5浓度']
        
        # 线性回归模型
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        y_pred_linear = linear_model.predict(X)

        # 评估指标
        print(f"线性回归 R²: {r2_score(y, y_pred_linear):.2f}")
        print(f"线性回归均方误差 (MSE): {mean_squared_error(y, y_pred_linear):.2f}")

        # 线性回归回归系数可视化
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_model.coef_})
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, hue=None, dodge=False)
        plt.title("Linear Regression Coefficients")
        plt.show()

        # ----------------------树模型分析----------------------
        print("\n开始树模型分析...")
        tree_model = RandomForestRegressor(random_state=42, n_estimators=100,n_jobs=-1)
        tree_model.fit(X, y)
        y_pred_tree = tree_model.predict(X)
        import joblib
        joblib.dump(tree_model, 'tree_model.pkl')
        # 评估指标
        print(f"随机森林 R²: {r2_score(y, y_pred_tree):.2f}")
        print(f"随机森林均方误差 (MSE): {mean_squared_error(y, y_pred_tree):.2f}")

        # 随机森林特征重要性可视化
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': tree_model.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, hue=None, dodge=False)
        plt.title("Random Forest Feature Importance")
        plt.show()
else:
    print("数据加载失败，无法进行分析")
