import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 辅助函数
def draw_doc_stack(x, y, title, color='#e0e0e0', highlight=False):
    rect = patches.Rectangle((x, y), 2.5, 3.5, facecolor=color, edgecolor='#616161', linewidth=1.5)
    ax.add_patch(rect)
    # Lines representing text
    for i in range(5):
        ax.plot([x + 0.3, x + 2.2], [y + 3.0 - i * 0.5, y + 3.0 - i * 0.5], color='#9e9e9e', linewidth=1)

    if highlight:
        rect.set_edgecolor('#d32f2f')
        rect.set_linewidth(2.5)
        ax.text(x + 1.25, y - 0.4, title, ha='center', fontweight='bold', color='#d32f2f', fontsize=10)
    else:
        ax.text(x + 1.25, y - 0.4, title, ha='center', fontsize=9)


# --- 左侧：Baseline ---
ax.text(3, 6.5, "Baseline (Vector Only)", ha='center', fontsize=12, fontweight='bold', color='#616161')

# 错误的文档堆
draw_doc_stack(1, 2.0, "燃油系统\n(无关)", '#eeeeee')
draw_doc_stack(1.5, 2.2, "", '#eeeeee')  # Shadow effect
draw_doc_stack(2.0, 2.4, "起动系统\n(无关)", '#eeeeee')

# 漏掉的文档 (虚线)
rect_miss = patches.Rectangle((3.5, 0.5), 2.0, 2.5, facecolor='none', edgecolor='#bdbdbd', linestyle='--', linewidth=1)
ax.add_patch(rect_miss)
ax.text(4.5, 1.75, "恒速传动\n(Missed)", ha='center', color='#bdbdbd', fontsize=8)

# 搜索光束 (发散)
ax.arrow(3, 5.5, -1, -2.5, head_width=0.2, color='#bdbdbd')
ax.arrow(3, 5.5, 0, -2.5, head_width=0.2, color='#bdbdbd')
ax.text(3, 5.8, "Query: 发电机...", ha='center', fontsize=9, style='italic')

# --- 分隔线 ---
ax.plot([6, 6], [0, 7], color='#cfd8dc', linestyle='--', linewidth=1.5)

# --- 右侧：Ours ---
ax.text(9, 6.5, "Ours (Topology Aware)", ha='center', fontsize=12, fontweight='bold', color='#2e7d32')

# 1. 锚点节点
circle = patches.Circle((9, 5.0), 0.6, facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=2)
ax.add_patch(circle)
ax.text(9, 5.0, "Entity:\n发电机", ha='center', va='center', fontsize=9, fontweight='bold', color='#1b5e20')

# 2. 拓扑连接线 (Red)
ax.annotate("", xy=(9, 3.5), xytext=(9, 4.4), arrowprops=dict(arrowstyle="->", lw=2.5, color='#d32f2f'))
ax.text(9.2, 4.0, "拓扑关联\n(PageRank)", fontsize=9, color='#d32f2f', fontweight='bold')

# 3. 被捞回的文档
draw_doc_stack(7.75, 0.5, "恒速传动系统\n(Correct Chunk)", '#fff3e0', highlight=True)

# 关键内容标注
ax.text(9, 2.0, "关键句: ...有两个传动来源...", ha='center', fontsize=9, color='#e65100', backgroundcolor='#fff3e0')

plt.tight_layout()
plt.savefig('case_study_viz.png', dpi=300, bbox_inches='tight')
print("Case study visualization generated.")