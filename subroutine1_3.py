import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib.font_manager as fm

def plot_pm25_bs(data_path, location_path, font_path='C:\\Windows\\Fonts\\msyh.ttc'):
    """
    该函数绘制多个关于PM2.5浓度的图表，包括时序图、按月变化图、按季节变化图等。
    
    参数:
    data_path: str, 数据文件路径 (csv格式)
    location_path: str, 城市位置文件路径 (csv格式)
    font_path: str, 字体路径，默认为微软雅黑字体路径
    """
    # 设置中文字体路径
    font_prop = fm.FontProperties(fname=font_path)

    # Matplotlib 中文支持
    rcParams['font.sans-serif'] = ['Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 加载数据
    data = pd.read_csv(data_path)
    locations = pd.read_csv(location_path)

    # 清理列名空格
    data.columns = data.columns.str.strip()

    # 合并城市数据
    merged_data = pd.merge(data, locations, left_on='城市', right_on='城市')

    # 转换日期格式并设置索引
    merged_data['datetime'] = pd.to_datetime(merged_data['日期'])
    merged_data.set_index('datetime', inplace=True)

    # 提取月份信息
    merged_data['month'] = merged_data.index.month  # 从 datetime 列提取月份

    # 获取城市列表
    cities = merged_data['城市'].unique()

    # --------------------------- 绘制总体PM2.5浓度变化 ---------------------------
    fig, axes = plt.subplots(5, 3, figsize=(20, 15), sharex=False, sharey=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.92)

    for i, city in enumerate(cities):
        city_data = merged_data[merged_data['城市'] == city]

        # 绘制原始数据的时序图
        axes[i].plot(city_data.index, city_data['PM2.5浓度'], color='darkblue')  # 统一深蓝色

        axes[i].set_title(city, fontproperties=font_prop, fontsize=14, pad=15)
        for ax in axes[:len(cities)]:
            ax.set_xlabel('日期', fontproperties=font_prop)
        axes[i].set_ylabel('PM2.5 (μg/m³)', fontproperties=font_prop)

        # 设置Y轴范围，让变化更明显
        axes[i].set_ylim(0, city_data['PM2.5浓度'].max() * 1.1)  # 设置Y轴上限为数据最大值的1.1倍
        axes[i].grid(True)

    # 隐藏多余的子图
    for ax in axes[len(cities):]:
        ax.axis('off')

    # 添加主标题
    fig.suptitle('13个城市的PM2.5时序图', fontproperties=font_prop, fontsize=22, y=0.98)
    plt.tight_layout()  # 自动调整子图间距
    plt.show()

    # --------------------------- 按月绘制时序图 ---------------------------
    fig, axes = plt.subplots(5, 3, figsize=(20, 15), sharex=False)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.92)

    for i, city in enumerate(cities):
        city_data = merged_data[merged_data['城市'] == city]
        monthly_avg = city_data.groupby('month')['PM2.5浓度'].mean()
        axes[i].plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', color='darkblue')  # 统一深蓝色
        axes[i].set_title(city, fontproperties=font_prop, fontsize=14, pad=15)
        axes[i].set_xlabel('月份', fontproperties=font_prop)
        axes[i].set_ylabel('PM2.5 浓度 (μg/m³)', fontproperties=font_prop)
        axes[i].grid(True)

    # 隐藏多余的子图
    for ax in axes[len(cities):]:
        ax.axis('off')

    # 添加主标题
    fig.suptitle('13个城市的PM2.5按月变化', fontproperties=font_prop, fontsize=22, y=0.98)
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


    # --------------------------- 按季节绘制条形图 ---------------------------
    def get_season(month):
        """返回月份对应的季节"""
        if month in [3, 4, 5]:
            return '春'
        elif month in [6, 7, 8]:
            return '夏'
        elif month in [9, 10, 11]:
            return '秋'
        else:
            return '冬'


    # 应用季节函数
    merged_data['season'] = merged_data['month'].apply(get_season)

    # 定义季节顺序
    season_order = ['春', '夏', '秋', '冬']

    # 设定季节颜色（按季节分配颜色）
    season_colors = {'春': 'lightgreen', '夏': 'lightcoral', '秋': 'lightskyblue', '冬': 'gold'}

    # 创建子图：13个城市的 PM2.5 按季节变化
    fig, axes = plt.subplots(5, 3, figsize=(20, 15), sharex=False)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.92)

    for i, city in enumerate(cities):
        city_data = merged_data[merged_data['城市'] == city]
        seasonal_avg = city_data.groupby('season')['PM2.5浓度'].mean().reindex(season_order)

        # 按季节顺序绘制条形图，使用季节颜色
        axes[i].bar(seasonal_avg.index, seasonal_avg.values, color=[season_colors[season] for season in seasonal_avg.index],
                    alpha=0.7)

        axes[i].set_title(city, fontproperties=font_prop, fontsize=14, pad=15)
        axes[i].set_xlabel('季节', fontproperties=font_prop)
        axes[i].set_ylabel('PM2.5 浓度 (μg/m³)', fontproperties=font_prop)
        axes[i].grid(True, axis='y')

    # 隐藏多余的子图
    for ax in axes[len(cities):]:
        ax.axis('off')

    # 添加主标题
    fig.suptitle('13个城市的PM2.5按季节变化', fontproperties=font_prop, fontsize=22, y=0.98)
    plt.tight_layout()  # 自动调整子图间距
    plt.show()

    # --------------------------- 按时刻绘制时序图 ---------------------------
    hourly_avg_per_city = merged_data.groupby(['城市', '时刻'])['PM2.5浓度'].mean().unstack(level=0)

    # 自定义横坐标刻度
    xticks_labels = ['0', '3', '6', '9', '12', '15', '18', '21']
    xticks_positions = range(0, 24, 3)  # 每3小时一个位置

    fig, axes = plt.subplots(5, 3, figsize=(20, 15), sharex=False, sharey=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.92)

    for i, city in enumerate(cities):
        if city in hourly_avg_per_city.columns:
            axes[i].plot(hourly_avg_per_city.index, hourly_avg_per_city[city], marker='o', linestyle='-', color='darkblue')  # 统一深蓝色
            axes[i].set_title(city, fontproperties=font_prop, fontsize=14, pad=15)
            axes[i].set_xlabel('时刻 (每3小时)', fontproperties=font_prop)
            axes[i].set_ylabel('PM2.5 (μg/m³)', fontproperties=font_prop)
            axes[i].set_xticks(xticks_positions)  # 设置自定义刻度位置
            axes[i].set_xticklabels(xticks_labels, fontproperties=font_prop)  # 设置自定义刻度标签
            axes[i].grid(True)

    # 隐藏多余的子图
    for ax in axes[len(cities):]:
        ax.axis('off')

    # 添加主标题
    fig.suptitle('13个城市的PM2.5按时刻变化', fontproperties=font_prop, fontsize=22, y=0.98)
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


