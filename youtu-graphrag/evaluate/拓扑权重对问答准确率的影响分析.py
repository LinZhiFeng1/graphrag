import matplotlib.pyplot as plt
import numpy as np

# 实验数据
betas = [0.0,  0.25, 0.5, 0.75, 1.0]
accuracies = [0.80,  0.8333, 0.8, 0.9333, 0.9]

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 绘制折线
ax.plot(betas, accuracies, marker='o', markersize=8, linewidth=2.5, color='#1f77b4', label='QA Accuracy')


# 添加数值标签
for i, txt in enumerate(accuracies):
    ax.annotate(f"{txt:.2%}", (betas[i], accuracies[i]), xytext=(0, 8), textcoords='offset points', ha='center', fontweight='bold')

# 装饰
# ax.set_title('图 4-11 拓扑权重对问答准确率的影响分析', fontsize=14, fontweight='bold', y=1.02)
ax.set_xlabel('拓扑权重 ($\\beta$)', fontsize=12)
ax.set_ylabel('问答准确率 (Accuracy)', fontsize=12)
ax.set_ylim(0.65, 0.98)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('ablation_study_chart.png')
print("图表已生成：ablation_study_chart.png")