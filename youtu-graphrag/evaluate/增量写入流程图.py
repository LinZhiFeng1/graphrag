import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

# === 1. 画布设置 ===
fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')

# 颜色定义
c_old_node = '#bdbdbd'   # 灰色旧节点
c_old_edge = '#e0e0e0'   # 浅灰旧边
c_new_node = '#f44336'   # 红色新节点
c_new_edge = '#ef5350'   # 红色新边 (虚线)
c_comm_1 = '#90caf9'     # 社区1颜色 (蓝)
c_comm_2 = '#a5d6a7'     # 社区2颜色 (绿)
c_comm_3 = '#ce93d8'     # 社区3颜色 (紫) - 合并后的新社区

# 字体
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS', 'sans-serif']
except:
    pass

# === 2. 绘制左侧：物理层局部写入 (Physical Layer) ===

# 基础图结构
G_left = nx.Graph()
# 旧节点 cluster 1
old_nodes_1 = [1, 2, 3, 4]
# 旧节点 cluster 2
old_nodes_2 = [5, 6, 7, 8]
G_left.add_nodes_from(old_nodes_1 + old_nodes_2)
# 内部边
G_left.add_edges_from([(1,2), (2,3), (3,4), (1,4), (1,3)])
G_left.add_edges_from([(5,6), (6,7), (7,8), (5,8), (6,8)])

# 布局
pos_left = {
    1: (2, 5), 2: (3, 5.5), 3: (3, 4.5), 4: (2, 4), # Cluster 1 (Top Left)
    5: (2, 2), 6: (3, 2.5), 7: (3, 1.5), 8: (2, 1)  # Cluster 2 (Bottom Left)
}

# 绘制旧部分
nx.draw_networkx_nodes(G_left, pos_left, nodelist=old_nodes_1+old_nodes_2, node_color=c_old_node, node_size=300, ax=ax, edgecolors='white')
nx.draw_networkx_edges(G_left, pos_left, edge_color=c_old_edge, width=2, ax=ax)

# 绘制新写入部分 (Local Update)
new_nodes = [9, 10]
pos_new = {9: (4.5, 3.5), 10: (3.5, 3.5)} # Bridge nodes
nx.draw_networkx_nodes(G_left, pos_new, nodelist=new_nodes, node_color=c_new_node, node_size=300, ax=ax, edgecolors='white', label='New Nodes')
# 新边 (连接两个旧簇)
new_edges = [(3, 10), (10, 9), (9, 6)]
nx.draw_networkx_edges(G_left, {**pos_left, **pos_new}, edgelist=new_edges, edge_color=c_new_edge, width=2, style='dashed', ax=ax)

# 标注
ax.text(3, 6.2, "Step 1: 物理层局部写入\n(Local Incremental Write)", ha='center', fontsize=11, fontweight='bold', color='#424242')
ax.text(3, 0.5, "状态: 仅更新局部邻接表\n(NetworkX add_node/edge)", ha='center', fontsize=9, color='#757575')

# === 3. 中间：算法处理 (Process) ===
ax.arrow(6, 3.5, 2, 0, head_width=0.2, head_length=0.2, fc='#607d8b', ec='#607d8b')
box = patches.FancyBboxPatch((6.5, 3.0), 1.5, 1.0, boxstyle='round,pad=0.1', fc='#fff9c4', ec='#fbc02d', linewidth=2)
ax.add_patch(box)
ax.text(7.25, 3.5, "Leiden算法\n全局重算", ha='center', va='center', fontsize=10, fontweight='bold')

# === 4. 右侧：语义层全局重组 (Semantic Layer) ===

# 结构是一样的，但是颜色变了
G_right = G_left.copy()
G_right.add_nodes_from(new_nodes)
G_right.add_edges_from(new_edges)

# 右侧位置平移
shift_x = 9
pos_right = {k: (v[0] + shift_x, v[1]) for k, v in {**pos_left, **pos_new}.items()}

# 重新染色：因为节点9,10的加入，上下两个社区合并成了一个大社区（紫色）
# 或者我们可以展示它变成了三个社区，或者维持两个。
# 这里的逻辑是：Leiden 全局跑了一遍，重新分配了颜色。
# 假设 1-4 还是社区1，5-8 还是社区2，9-10 归入社区2。或者全部合并。
# 为了展示“重组”，我们让它们全部合并成一个大社区（紫色），说明新节点带来了连通性变化。

all_nodes = old_nodes_1 + old_nodes_2 + new_nodes
nx.draw_networkx_nodes(G_right, pos_right, nodelist=all_nodes, node_color=c_comm_3, node_size=300, ax=ax, edgecolors='white')
nx.draw_networkx_edges(G_right, pos_right, edge_color='#bdbdbd', width=1.5, ax=ax)

# 标注
ax.text(3 + shift_x, 6.2, "Step 2: 语义层全局重组\n(Global Community Refresh)", ha='center', fontsize=11, fontweight='bold', color='#424242')
ax.text(3 + shift_x, 0.5, "状态: 社区结构全局更新\n(FastTreeComm / Leiden)", ha='center', fontsize=9, color='#757575')

# 标题
# ax.set_title("图 3-6 图谱增量写入与全局社区结构重组流程", fontsize=14, y=0.98)

# Legend示意
ax.legend([
    patches.Circle((0,0), color=c_old_node, label='历史节点'),
    patches.Circle((0,0), color=c_new_node, label='增量节点'),
    patches.Circle((0,0), color=c_comm_3, label='重组后的社区'),
], ['历史节点', '增量节点', '重组后的社区'], loc='upper right', bbox_to_anchor=(0.95, 0.9))

plt.tight_layout()
plt.savefig('incremental_update_flow_hd.png', dpi=300)
print("Image saved.")