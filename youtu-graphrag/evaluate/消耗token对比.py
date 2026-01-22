import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === 1. 实验数据配置 (请在此处填入你的真实数据) ===
# 提示：全量构建通常随着数据量增加，Token消耗会线性增长（因为每次都要处理所有文本）
# 增量构建通常Token消耗较低且相对稳定（只处理新增部分 + 少量检索开销）

# [示例数据 - 请替换为你日志里的真实数值]
# 对应 20%, 40%, 60%, 80%, 100% 五个阶段
full_tokens = [91925, 114257, 124181, 184659, 197410]  # Baseline: 全量重构 Token 消耗
inc_tokens =  [20707, 45226,  63102,  86706,  98938]   # Ours: 增量更新 Token 消耗 (每次只处理增量)

# === 2. 五个阶段的数据 ===
stages = np.array([20, 40, 60, 80, 100])  # 数据量百分比

# === 3. 生成图表 ===
plt.figure(figsize=(10, 6), dpi=100)

x = np.arange(len(stages))
width = 0.35

# 画柱子 (保持与时间对比图一致的配色)
# Full Rebuild (Baseline) -> 蓝色 #5470c6
# Ours (Incremental) -> 绿色 #91cc75
bars1 = plt.bar(x - width/2, full_tokens, width, label='Full Rebuild (Baseline)', color='#5470c6', alpha=0.9)
bars2 = plt.bar(x + width/2, inc_tokens, width, label='Ours (Incremental)', color='#91cc75', alpha=0.9)

# 设置标签和标题
plt.xlabel('Data Volume Percentage (%)', fontsize=12)
plt.ylabel('Token Consumption (Count)', fontsize=12) # Y轴改成 Token 消耗
plt.title('Cost Efficiency Comparison: Token Consumption', fontsize=14)
plt.xticks(x, [f"{s}%" for s in stages])
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 在柱子上标数值 (自动添加 'k' 单位如果数值很大)
def format_value(val):
    if val >= 1000:
        return f'{val/1000:.1f}k'
    return f'{int(val)}'

for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             format_value(height), ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             format_value(height), ha='center', va='bottom', fontsize=9)

# 保存
plt.tight_layout()
plt.savefig('token_comparison_chart.png')
print("✅ Token 对比图已生成: token_comparison_chart.png")
plt.show()

# === 4. 生成论文表格数据 (方便你填表) ===
df = pd.DataFrame({
    "Data Volume": [f"{s}%" for s in stages],
    "Full Rebuild (Tokens)": full_tokens,
    "Incremental (Tokens)": inc_tokens,
    "Saving Ratio": [f"{round((f-i)/f * 100, 1)}%" if f>0 else "0%" for f, i in zip(full_tokens, inc_tokens)]
})
print("\n=== Token 消耗对比数据表 ===")
print(df)