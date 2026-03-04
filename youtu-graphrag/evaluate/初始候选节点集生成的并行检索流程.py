import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# 字体设置 (尝试适配中文)
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK JP', 'sans-serif']
except:
    pass

# 辅助绘图函数
def draw_box(x, y, w, h, text, color, edge, style='round,pad=0.1'):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=style,
                                 linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#444444'))

# --- 1. 输入层 ---
draw_box(0.5, 3.0, 1.5, 1.0, "用户查询", '#fff9c4', '#fbc02d')

# --- 2. 并行处理层 ---
# 分支点
draw_arrow(2.0, 3.5, 2.55, 3.5)
# 向上：向量路径
draw_arrow(2.5, 3.5, 3.0, 5.0)
draw_box(3.1, 4.5, 2.0, 1.0, "语义编码\n(BGE-M3)", '#e3f2fd', '#1e88e5')
draw_arrow(5.1, 5.0, 5.5, 5.0)
draw_box(5.55, 4.5, 2.5, 1.0, "FAISS向量检索", '#e3f2fd', '#1e88e5')

# 向下：关键词路径
draw_arrow(2.5, 3.5, 3.0, 2.0)
draw_box(3.1, 1.5, 2.0, 1.0, "关键词提取", '#e1bee7', '#8e24aa')
draw_arrow(5.1, 2.0, 5.5, 2.0)
draw_box(5.55, 1.5, 2.5, 1.0, "关键词匹配", '#e1bee7', '#8e24aa')

# --- 3. 合并层 ---
draw_arrow(8.15, 5.0, 8.5, 3.8) # 上汇入
draw_arrow(8.15, 2.0, 8.5, 3.2) # 下汇入

draw_box(8.6, 2.8, 2.0, 1.4, "候选集合并\n&\n相似度批计算", '#c8e6c9', '#43a047')

# --- 4. 输出层 ---
draw_arrow(10.65, 3.5, 11.05, 3.5)
draw_box(11.1, 2.5, 0.8, 2.0, "初始\n候选\n集", '#ffccbc', '#d84315')

# 标题
# ax.text(6, 6.5, "图 4-2 初始候选节点集生成的并行检索流程", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('retrieval_process_flow.png', dpi=300, bbox_inches='tight')
print("Image saved.")