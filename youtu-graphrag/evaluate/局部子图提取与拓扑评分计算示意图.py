import matplotlib.pyplot as plt
import networkx as nx

# 设置画布
fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
ax.axis('off')

# 字体设置
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK JP', 'sans-serif']
except:
    pass

# --- 1. 左侧：全图与候选点 ---
pos_base_x = 0
pos_base_y = 0

# 创建一个随机图作为背景
G_full = nx.erdos_renyi_graph(20, 0.2, seed=42)
pos = nx.spring_layout(G_full, center=(2, 3), scale=1.5, seed=42)

# 绘制背景图（灰色）
nx.draw_networkx_nodes(G_full, pos, node_size=100, node_color='#e0e0e0', ax=ax)
nx.draw_networkx_edges(G_full, pos, edge_color='#eeeeee', alpha=0.5, ax=ax)

# 标记候选点（红色）
candidates = [0, 4, 19]
nx.draw_networkx_nodes(G_full, pos, nodelist=candidates, node_size=150, node_color='#ef5350', label='Candidates', ax=ax)
ax.text(2, 0.5, "1. 初始候选节点定位\n(Initial Candidates)", ha='center', fontsize=10, fontweight='bold')

# --- 箭头指向中间 ---
ax.arrow(4.5, 3, 1, 0, head_width=0.2, head_length=0.2, fc='#757575', ec='#757575')

# --- 2. 中间：子图提取 ---
# 提取子图（候选点 + 邻居）
neighbors = set()
for n in candidates:
    neighbors.update(list(G_full.neighbors(n)))
subgraph_nodes = list(set(candidates) | neighbors)
G_sub = G_full.subgraph(subgraph_nodes)

# 平移位置到中间
pos_sub = {k: (v[0] + 6, v[1]) for k, v in pos.items() if k in subgraph_nodes}

# 绘制子图
# 邻居节点（蓝色）
nx.draw_networkx_nodes(G_sub, pos_sub, nodelist=list(neighbors - set(candidates)), node_size=100, node_color='#90caf9', ax=ax)
# 候选节点（红色）
nx.draw_networkx_nodes(G_sub, pos_sub, nodelist=candidates, node_size=150, node_color='#ef5350', ax=ax)
# 子图连边（深色）
nx.draw_networkx_edges(G_sub, pos_sub, edge_color='#616161', ax=ax)

ax.text(8, 0.5, "2. 局部诱导子图构建\n(Induced Subgraph)", ha='center', fontsize=10, fontweight='bold')

# --- 箭头指向右侧 ---
ax.arrow(10.5, 3, 1, 0, head_width=0.2, head_length=0.2, fc='#757575', ec='#757575')

# --- 3. 右侧：拓扑评分 ---
# 计算 PageRank
pr = nx.pagerank(G_sub, alpha=0.85)
# 根据 PR 值调整节点大小
node_sizes = [pr.get(n, 0) * 5000 for n in subgraph_nodes]
# 根据 PR 值调整颜色深浅 (Reds)
node_colors = [pr.get(n, 0) for n in subgraph_nodes]

# 平移位置到右侧
pos_score = {k: (v[0] + 6, v[1]) for k, v in pos_sub.items()}

nodes = nx.draw_networkx_nodes(G_sub, pos_score, nodelist=subgraph_nodes,
                       node_size=node_sizes,
                       node_color=node_colors,
                       cmap=plt.cm.Reds,
                       edgecolors='#424242', # 边框
                       ax=ax)
nx.draw_networkx_edges(G_sub, pos_score, edge_color='#bdbdbd', ax=ax)

# 标注高分节点
high_score_node = max(pr, key=pr.get)
x, y = pos_score[high_score_node]
ax.text(x, y+0.3, f"PR={pr[high_score_node]:.2f}", ha='center', color='#d50000', fontweight='bold', fontsize=9)

ax.text(14, 0.5, "3. PageRank 拓扑评分\n(Topology Scoring)", ha='center', fontsize=10, fontweight='bold')

# 添加 Colorbar
# cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
# cbar.ax.set_ylabel('Topology Score', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('topology_scoring_flow.png', dpi=300, bbox_inches='tight')
print("Image saved.")