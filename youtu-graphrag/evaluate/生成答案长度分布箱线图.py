import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. 数据加载 ===
# 请确保这两个 CSV 文件与脚本在同一目录下，或者修改为绝对路径
file_baseline = "老正确率结果/问答_结果_alpha1.00_beta0.00_实时保存.csv"
file_ours = "老正确率结果/问答_结果_alpha0.25_beta0.75_实时保存.csv"

try:
    df_base = pd.read_csv(file_baseline)
    df_ours = pd.read_csv(file_ours)

    # === 2. 数据处理：计算答案长度 ===
    # 将'回答'列转换为字符串长度
    df_base['len'] = df_base['回答'].apply(lambda x: len(str(x)))
    df_ours['len'] = df_ours['回答'].apply(lambda x: len(str(x)))

    # === 3. 绘图设置 ===
    # 设置字体以支持中文显示 (SimHei 为黑体, Arial Unicode MS 为 Mac 备选)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 准备绘图数据和标签
    data_to_plot = [df_base['len'], df_ours['len']]
    labels = ['Baseline\n(Vector-Only)', 'Ours\n(Topology-Aware)']

    # === 4. 绘制箱线图 (Box Plot) ===
    # patch_artist=True 允许填充颜色
    box = ax.boxplot(data_to_plot, patch_artist=True, labels=labels, widths=0.5,
                     medianprops={'color': '#333333', 'linewidth': 2},  # 中位数线颜色
                     boxprops={'linewidth': 1.5, 'edgecolor': '#333333'},  # 箱体边框
                     whiskerprops={'linewidth': 1.5, 'color': '#333333'},  # 须线
                     capprops={'linewidth': 1.5, 'color': '#333333'},  # 须帽
                     flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': '#333333'})  # 异常值点

    # === 5. 自定义配色 ===
    # Baseline 用灰色，Ours 用绿色，突出对比
    colors = ['#eeeeee', '#a5d6a7']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # === 6. 添加数据抖动散点 (Jitter Points) ===
    # 在箱线图上叠加真实的散点，增加数据透明度
    for i, data in enumerate(data_to_plot):
        y = data
        # 生成随机 x 坐标 (以 1 或 2 为中心，左右轻微偏移)
        x = np.random.normal(1 + i, 0.04, size=len(y))
        ax.plot(x, y, 'r.', alpha=0.4, markersize=8)

    # === 7. 标题与标签 ===
    # ax.set_title('图 4-10 生成答案长度分布箱线图', fontsize=14, y=1.02, fontweight='bold')
    ax.set_ylabel('答案字符数 (Character Count)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)  # 添加水平网格线

    # 优化刻度显示
    plt.tick_params(axis='both', which='major', labelsize=11)

    # === 8. 保存与展示 ===
    plt.tight_layout()
    save_path = 'answer_length_boxplot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    # plt.show()

except FileNotFoundError as e:
    print(f"错误: 找不到文件 {e.filename}。请检查文件名或路径。")
except Exception as e:
    print(f"发生未知错误: {e}")