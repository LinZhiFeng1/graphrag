import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter

# === 1. 加载 Ours 图谱 ===
file_path = 'evaluate/graph_ours.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === 2. 构建 NetworkX 图 & 提取社区 ===
G = nx.Graph()
community_map = {}  # 记录每个实体属于哪个社区

for item in data:
    s = item['start_node']['properties']['name']
    t = item['end_node']['properties']['name']

    # 你的数据里，如果连接到 "Community" 节点，那个关系往往是 "member_of" 或 "keyword_of"
    # 我们利用这个结构来统计社区大小
    s_label = item['start_node']['label']
    t_label = item['end_node']['label']

    # 构建基础图结构用于计算度
    if s_label != 'community' and t_label != 'community':
        G.add_edge(s, t)

    # 统计社区成员
    # 假设结构是：Entity --[member_of]--> Community
    if t_label == 'community':
        comm_name = t
        entity_name = s
        if comm_name not in community_map: community_map[comm_name] = set()
        community_map[comm_name].add(entity_name)
    elif s_label == 'community':
        comm_name = s
        entity_name = t
        if comm_name not in community_map: community_map[comm_name] = set()
        community_map[comm_name].add(entity_name)

# === 3. 分析指标 ===
# A. 社区规模分布
comm_sizes = [len(members) for members in community_map.values()]
comm_sizes.sort(reverse=True)

# B. 连通性指标 (对于 3.4.2 证明"融合质量"很有用)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

print(f"=== 3.4.2 图谱质量统计 ===")
print(f"检测到的语义社区数量: {len(comm_sizes)}")
print(f"最大社区成员数: {max(comm_sizes) if comm_sizes else 0}")
print(f"平均社区成员数: {np.mean(comm_sizes):.2f}")
print(f"实体节点平均度 (Average Degree): {avg_degree:.2f}")

# === 4. 绘图：社区规模分布 (直方图) ===
# 证明：图谱形成了有意义的聚集，而不是均匀的噪声
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 过滤掉太小的社区以免图太乱 (比如只画 Top 50 或 成员数>3的)
filtered_sizes = [s for s in comm_sizes if s > 1]

plt.hist(filtered_sizes, bins=30, color='#69b3a2', edgecolor='black', alpha=0.7)
# plt.title('图 3-x 增量构建后的图谱社区规模分布 (Community Size Distribution)', fontsize=14)
plt.xlabel('社区包含的实体数量 (Community Size)', fontsize=12)
plt.ylabel('社区频次 (Frequency)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 添加一段文字说明
text_str = f'Total Entities: {num_nodes}\nTotal Communities: {len(community_map)}\nAvg Degree: {avg_degree:.2f}'
plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('community_quality_analysis.png')
print("✅ 3.4.2 专用图表已生成: community_quality_analysis.png")