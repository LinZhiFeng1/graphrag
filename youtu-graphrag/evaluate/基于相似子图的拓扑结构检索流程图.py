import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

# 设置中文字体，确保中文能正常显示
# Windows 一般使用 SimHei，Mac 可以使用 Arial Unicode MS 或 PingFang HK
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang HK']
plt.rcParams['axes.unicode_minus'] = False


def draw_box(ax, center_x, center_y, width, height, text, bg_color, edge_color, text_color='black', fontsize=10):
    """绘制圆角矩形和文字"""
    # FancyBboxPatch 锚点在左下角，因此要减去宽高的一半
    box = patches.FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width, height,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor=edge_color, facecolor=bg_color, lw=1.5
    )
    ax.add_patch(box)
    ax.text(center_x, center_y, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, linespacing=1.5)


def draw_circle(ax, center_x, center_y, radius, text, bg_color, edge_color, fontsize=10):
    """绘制圆形和文字"""
    circle = patches.Circle((center_x, center_y), radius,
                            edgecolor=edge_color, facecolor=bg_color, lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(center_x, center_y, text, ha='center', va='center',
            fontsize=fontsize, zorder=4, linespacing=1.5)


def draw_node(ax, x, y, label, node_type):
    """绘制知识图谱节点"""
    colors = {
        'anchor': {'face': '#FF6666', 'edge': '#CC0000', 'text': 'white'},  # 红色
        'neighbor': {'face': '#FFB366', 'edge': '#CC6600', 'text': 'black'},  # 橙色
        'unrelated': {'face': '#E6E6E6', 'edge': '#999999', 'text': 'black'}  # 灰色
    }
    style = colors.get(node_type, colors['unrelated'])

    circle = patches.Circle((x, y), 0.35, edgecolor=style['edge'],
                            facecolor=style['face'], lw=1.5, zorder=5)
    ax.add_patch(circle)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=9, color=style['text'], zorder=6)


# 1. 创建画布
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')  # 隐藏坐标轴

# 添加主标题

# ==================== 绘制顶部流程 ====================
# 输入新文本片段
draw_box(ax, 2, 9, 2.2, 1.2, "输入新文本片段",
         bg_color='#FFF2CC', edge_color='#D6B656')

# 语义编码器
draw_box(ax, 5, 9, 2.5, 1.2, "语义编码器\n(BGE-M3 Encoder)",
         bg_color='#DAE8FC', edge_color='#6C8EBF')

# 相似度计算
draw_circle(ax, 8, 9, 0.6, "$q \cdot E^V$\n相似度计算\n(Similarity)",
            bg_color='#E6E6E6', edge_color='#B3B3B3')

# 历史向量索引
draw_box(ax, 8, 7.5, 2.2, 0.8, "历史向量索引",
         bg_color='#D5E8D4', edge_color='#82B366')

# 锚点列表
draw_box(ax, 11.5, 9, 2.5, 1.2, "锚点节点 ID 列表\n(Anchor Nodes: [A, B, C])",
         bg_color='#FFF2CC', edge_color='#D6B656')

# 顶部箭头及文字
ax.annotate("", xy=(3.6, 9), xytext=(3.2, 9), arrowprops=dict(arrowstyle="->", color="black"))
ax.annotate("", xy=(7.3, 9), xytext=(6.3, 9), arrowprops=dict(arrowstyle="->", color="black"))
ax.text(6.8, 9.1, "生成查询向量 $q$", ha='center', fontsize=9)

ax.annotate("", xy=(8, 8), xytext=(8, 8.5), arrowprops=dict(arrowstyle="->", color="black"))

ax.annotate("", xy=(10.1, 9), xytext=(8.7, 9), arrowprops=dict(arrowstyle="->", color="black"))
ax.text(9.4, 9.1, "Top-K 排序", ha='center', fontsize=9)

# ==================== 绘制底部知识图谱区 ====================
# 背景虚线框
rect = patches.Rectangle((0.5, 1), 14, 5.5, linewidth=1, edgecolor='#B3B3B3',
                         facecolor='none', linestyle='--', zorder=0)
ax.add_patch(rect)
ax.text(0.7, 6.3, "历史知识图谱", fontsize=11, color='gray')

# 节点坐标定义
nodes = {
    'A': (3.5, 4.0), 'B': (7.5, 4.5), 'C': (11.5, 3.8),
    'N1': (2.5, 5.5), 'N2': (5.5, 5.0), 'N3': (8.5, 5.5), 'N4': (11.5, 5.0),
    'G1': (1.5, 3.8), 'G2': (4.5, 2.8), 'G3': (9.5, 2.8)
}

# 绘制边 (Edge)
edges = [
    # (起始节点, 结束节点, 颜色, 粗细)
    ('A', 'N1', '#FF9933', 2), ('A', 'N2', '#FF9933', 2),
    ('B', 'N2', '#FF9933', 2), ('B', 'N3', '#FF9933', 2),
    ('C', 'N4', '#FF9933', 2),
    ('A', 'G1', '#CCCCCC', 1.5), ('A', 'G2', '#CCCCCC', 1.5),
    ('B', 'G2', '#CCCCCC', 1.5), ('B', 'G3', '#CCCCCC', 1.5)
]

for start, end, color, lw in edges:
    ax.plot([nodes[start][0], nodes[end][0]], [nodes[start][1], nodes[end][1]],
            color=color, lw=lw, zorder=1)

# 绘制节点 (Node)
for name, pos in nodes.items():
    if name in ['A', 'B', 'C']:
        draw_node(ax, pos[0], pos[1], name, 'anchor')
        ax.text(pos[0], pos[1] - 0.6, "Anchor", ha='center', color='#FF6666', fontsize=9)
    elif name.startswith('N'):
        draw_node(ax, pos[0], pos[1], name, 'neighbor')
    else:
        draw_node(ax, pos[0], pos[1], "", 'unrelated')

# ==================== 绘制跨区域关联与右侧结果 ====================
# 1. 定位锚点 箭头 (曲线)
ax.annotate("", xy=(nodes['B'][0] + 0.3, nodes['B'][1] + 0.3), xytext=(11.5, 8.3),
            arrowprops=dict(arrowstyle="->", color="black", connectionstyle="arc3,rad=0.1"))
ax.text(11, 7.5, "1. 定位锚点", fontsize=10)

# 2. 拓扑扩展 箭头 (底部向上)
ax.annotate("", xy=(nodes['B'][0], nodes['B'][1] - 0.5), xytext=(nodes['B'][0], 2.5),
            arrowprops=dict(arrowstyle="->", color="black"))
ax.text(nodes['B'][0], 2.2, "2. 拓扑扩展 ",
        ha='center', fontsize=10, color='black')

# 3. 提取结构化三元组框
context_text = '3. 提取结构化三元组\n["[实体A, 关系1, 实体N1]",\n"[实体A, 关系2, 实体N2]",\n"[实体B, 关系3, 实体N2]",\n...]'
draw_box(ax, 14.2, 3.5, 2.5, 3.0, context_text,
         bg_color='#FFF2CC', edge_color='#D6B656', fontsize=9)

# 展示和保存图片
plt.tight_layout()
plt.savefig('knowledge_graph_retrieval.png', dpi=300, bbox_inches='tight')
# plt.show()