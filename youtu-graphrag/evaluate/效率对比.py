import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === 1. 真实实验数据配置 ===
# 从实际测试中获得的数据
# 全量构建各阶段时间（秒）
real_full_times = [59.73+180.40, 73.99+232.74, 82.25+247.29, 94.54+281.13, 112.42+354.34]  # 20%, 40%, 60%, 80%, 100%

# 增量构建各阶段时间（秒）
real_inc_times = [48.02+107.70, 59.09+154.74, 51.54+185.63, 62.79+226.25, 61.66+229.44]      # 20%, 40%, 60%, 80%, 100%

# === 2. 五个阶段的数据 ===
stages = np.array([20, 40, 60, 80, 100])  # 数据量百分比

# 使用真实数据
full_times = real_full_times
inc_times = real_inc_times

# === 3. 生成图表 ===
plt.figure(figsize=(10, 6), dpi=100)

x = np.arange(len(stages))
width = 0.35

# 画柱子
bars1 = plt.bar(x - width/2, full_times, width, label='Full Rebuild (Baseline)', color='#5470c6', alpha=0.9)
bars2 = plt.bar(x + width/2, inc_times, width, label='Ours (Incremental)', color='#91cc75', alpha=0.9)

# 设置标签和标题
plt.xlabel('Data Volume Percentage (%)', fontsize=12)
plt.ylabel('Construction Time (seconds)', fontsize=12)
plt.title('Efficiency Comparison: Incremental vs Full Construction', fontsize=14)
plt.xticks(x, [f"{s}%" for s in stages])
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 在柱子上标数值
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}s', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}s', ha='center', va='bottom', fontsize=9)

# 保存
plt.tight_layout()
plt.savefig('efficiency_chart_real.png')
plt.show()

# === 4. 生成论文表格数据 ===
df = pd.DataFrame({
    "Data Volume": [f"{s}%" for s in stages],
    "Full Rebuild (s)": [round(x, 1) for x in full_times],
    "Incremental (s)": [round(x, 1) for x in inc_times],
    "Speedup Ratio": [round(f/i, 2) for f, i in zip(full_times, inc_times)]
})
print("真实实验数据表格：")
print(df)
