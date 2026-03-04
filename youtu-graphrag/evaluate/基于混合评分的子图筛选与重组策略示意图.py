import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

# 设置画布
fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# 字体设置
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK JP', 'sans-serif']
except:
    pass

# --- 1. 左侧：评分排序列表 ---
# 模拟数据：节点ID，总分，是否选中
nodes_data = [
    ('Node A', 0.95, True), ('Node B', 0.88, True), ('Node C', 0.82, True),
    ('Node D', 0.75, True), ('Node E', 0.71, True),  # Top 5
    ('Node F', 0.65, False), ('Node G', 0.45, False), ('Node H', 0.30, False)
]

start_x = 1.0
start_y = 5.0
bar_height = 0.4
gap = 0.2

ax.text(start_x + 1.0, 5.5, "1. 扩展候选集排序\n(Expanded Candidates)", ha='center', fontsize=10, fontweight='bold')

for i, (name, score, kept) in enumerate(nodes_data):
    y = start_y - i * (bar_height + gap)
    color = '#4caf50' if kept else '#bdbdbd'  # Green for kept, Grey for dropped

    # 画名字
    ax.text(start_x, y + bar_height / 2, name, va='center', ha='right', fontsize=9)
    # 画分数条
    rect = patches.Rectangle((start_x + 0.2, y), score * 2.5, bar_height, facecolor=color, edgecolor='none')
    ax.add_patch(rect)
    # 画数值
    ax.text(start_x + 0.2 + score * 2.5 + 0.1, y + bar_height / 2, f"{score:.2f}", va='center', fontsize=8)

# 画截断线
cut_y = start_y - 4.8 * (bar_height + gap)
ax.plot([start_x - 0.5, start_x + 3.5], [cut_y, cut_y], 'r--', lw=1.5)
ax.text(start_x + 3.8, cut_y, "Top-K\nCut-off", va='center', color='red', fontsize=9, fontweight='bold')

# --- 箭头 ---
ax.arrow(5.0, 3.0, 1.5, 0, head_width=0.2, head_length=0.2, fc='#616161', ec='#616161')
ax.text(5.75, 3.2, "筛选与\n重组", ha='center', fontsize=9)

# --- 2. 右侧：活跃子图 ---
# 构建一个简单的图
G = nx.Graph()
# 核心节点
core_nodes = ['A', 'B', 'C', 'D', 'E']
# 邻居节点
neighbor_nodes = ['n1', 'n2', 'n3']
G.add_nodes_from(core_nodes)
G.add_nodes_from(neighbor_nodes)

# 边：核心节点内部连接 (Paths) + 邻居连接 (1-hop)
edges = [
    ('A', 'B'), ('B', 'C'), ('A', 'E'),  # Paths between cores
    ('A', 'n1'), ('C', 'n2'), ('D', 'n3')  # 1-hop neighbors
]
G.add_edges_from(edges)

# 布局
pos = {
    'A': (9, 3.5), 'B': (10.5, 4.5), 'C': (12, 3.5),
    'D': (10.5, 2.5), 'E': (8, 2.5),
    'n1': (8, 4.5), 'n2': (13, 2.5), 'n3': (10.5, 1.0)
}

# 绘制
# 核心节点 (绿色)
nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_color='#4caf50', node_size=500, ax=ax, edgecolors='white',
                       linewidths=1.5)
nx.draw_networkx_labels(G, pos, labels={n: n for n in core_nodes}, font_color='white', font_weight='bold', ax=ax)

# 邻居节点 (浅绿)
nx.draw_networkx_nodes(G, pos, nodelist=neighbor_nodes, node_color='#a5d6a7', node_size=300, ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: n for n in neighbor_nodes}, font_size=8, ax=ax)

# 边
nx.draw_networkx_edges(G, pos, edge_color='#616161', width=1.5, ax=ax)

# 标注
ax.text(10.5, 5.5, "2. 活跃子图语境重组\n(Active Subgraph Context)", ha='center', fontsize=10, fontweight='bold')

# 图例说明
legend_x = 7.5
legend_y = 0.5
ax.add_patch(patches.Circle((legend_x, legend_y), 0.15, color='#4caf50'))
ax.text(legend_x + 0.3, legend_y, "Top-K 核心锚点 (V_core)", va='center', fontsize=9)

ax.add_patch(patches.Circle((legend_x + 3.5, legend_y), 0.15, color='#a5d6a7'))
ax.text(legend_x + 3.8, legend_y, "1-hop 关联邻居", va='center', fontsize=9)

plt.tight_layout()
plt.savefig('subgraph_filtering_strategy.png', dpi=300, bbox_inches='tight')
print("Image saved.")