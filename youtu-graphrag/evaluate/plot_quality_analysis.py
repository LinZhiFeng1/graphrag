import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

# 设置支持中文的字体
import matplotlib.font_manager as fm

# 检查系统可用的中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist if 'Chinese' in f.name or 'Song' in f.name or 'Hei' in f.name]

# 设置中文字体（优先级顺序）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# === 1. 加载数据 ===
# 请确保文件名与你实际存放的路径一致
file_baseline = "evaluate/graph_baseline.json"
file_ours = "evaluate/graph_ours.json"

def load_edges(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 将边提取为元组 (起点, 关系, 终点) 以便比较
    edges = set()
    for item in data:
        s = item.get('start_node', {}).get('properties', {}).get('name', 'UNK')
        t = item.get('end_node', {}).get('properties', {}).get('name', 'UNK')
        r = item.get('relation', 'related')
        edges.add((s, r, t))
    return edges

print("正在分析图谱结构差异...")
edges_base = load_edges(file_baseline)
edges_ours = load_edges(file_ours)

# === 2. 计算集合重叠 ===
common = edges_base.intersection(edges_ours)
only_base = edges_base - edges_ours
only_ours = edges_ours - edges_base

n_common = len(common)
n_base_only = len(only_base)
n_ours_only = len(only_ours)

print(f"Baseline 总边数: {len(edges_base)}")
print(f"Ours 总边数: {len(edges_ours)}")
print(f"共同边数 (Common): {n_common}")
print(f"Baseline 独有 (Discarded): {n_base_only}")
print(f"Ours 独有 (Refined/New): {n_ours_only}")

# === 3. 绘制堆叠柱状图 ===
# 设置字体以支持中文 (如果服务器不支持中文，可改为英文标签)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

labels = ['Baseline (无拓扑)', 'Ours (拓扑感知)']
width = 0.5

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 定义颜色方案
color_common = '#a6cee3'  # 浅蓝 (共同部分)
color_discarded = '#fb9a99' # 浅红 (被遗弃/噪声)
color_new = '#b2df8a'     # 浅绿 (新发现/优化)

# 绘制 Baseline 柱子
p1 = ax.bar(labels[0], n_common, width, label='共同保留关系 (Common)', color=color_common, edgecolor='grey')
p2 = ax.bar(labels[0], n_base_only, width, bottom=n_common, label='基线独有/冗余关系 (Discarded)', color=color_discarded, hatch='//', edgecolor='grey')

# 绘制 Ours 柱子
p3 = ax.bar(labels[1], n_common, width, color=color_common, edgecolor='grey')
p4 = ax.bar(labels[1], n_ours_only, width, bottom=n_common, label='拓扑增强新关系 (Refined)', color=color_new, hatch='xx', edgecolor='grey')

# 添加数值标签
ax.bar_label(p1, label_type='center', fontsize=10)
ax.bar_label(p2, label_type='center', fontsize=10)
ax.bar_label(p3, label_type='center', fontsize=10)
ax.bar_label(p4, label_type='center', fontsize=10)

# 设置图表细节
ax.set_ylabel('知识图谱边数量 (Edges Count)', fontsize=12)
# ax.set_title('图 3-x 图谱融合前后的结构演化与重构质量对比', fontsize=13, pad=15)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 保存
output_file = "fusion_quality_analysis.png"
plt.tight_layout()
plt.savefig(output_file)
print(f"✅ 图表已生成: {output_file}")