import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# 字体设置
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK JP', 'sans-serif']
except:
    pass

# 辅助绘图函数
def draw_box(x, y, w, h, text, color, edge):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                 linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#444444'))

# --- 1. 输入层 ---
draw_box(1.0, 6.5, 2.5, 1.0, "向量相似度\nVector Scores", '#bbdefb', '#1976d2')
draw_box(6.5, 6.5, 2.5, 1.0, "拓扑关联度\nTopology Scores", '#ffcdd2', '#d32f2f')

# --- 2. 归一化层 ---
draw_arrow(2.25, 6.5, 2.25, 5.5)
draw_arrow(7.75, 6.5, 7.75, 5.5)

draw_box(1.0, 4.5, 2.5, 1.0, "Min-Max 归一化\nNormalization", '#e1bee7', '#7b1fa2')
draw_box(6.5, 4.5, 2.5, 1.0, "Min-Max 归一化\nNormalization", '#e1bee7', '#7b1fa2')

# --- 3. 融合层 ---
draw_arrow(2.25, 4.5, 4.0, 3.0)
draw_arrow(7.75, 4.5, 6.0, 3.0)

# 加权融合图标
circle = patches.Circle((5.0, 3.0), 0.6, facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=2)
ax.add_patch(circle)
ax.text(5.0, 3.0, "+", ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(3.5, 3.5, "× α", ha='center', fontsize=12, color='#1976d2')
ax.text(6.5, 3.5, "× β", ha='center', fontsize=12, color='#d32f2f')

# --- 4. 排序与输出 ---
draw_arrow(5.0, 2.4, 5.0, 1.8)

draw_box(3.5, 0.8, 3.0, 1.0, "重排序与截断\nRe-ranking (Top-K)", '#c8e6c9', '#388e3c')

# 标题
# ax.text(5, 7.8, "图 4-4 混合评分与结果重排序流程", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('hybrid_scoring_flow.png', dpi=300, bbox_inches='tight')
print("Image saved.")