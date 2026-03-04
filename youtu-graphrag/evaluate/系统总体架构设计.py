import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 字体设置
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
except:
    pass


# 绘图辅助函数
def draw_layer(x, y, w, h, title, color, sub_boxes=[]):
    # Main Layer Box
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                  linewidth=1.5, edgecolor='#333333', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    # Title Tag
    ax.text(x - 0.8, y + h / 2, title, ha='center', va='center', fontsize=11, fontweight='bold', rotation=90)

    # Sub components
    for sb in sub_boxes:
        sx, sy, sw, sh, stext = sb
        # Adjust relative coords to absolute
        abs_x = x + sx
        abs_y = y + sy
        sub_rect = patches.Rectangle((abs_x, abs_y), sw, sh, linewidth=1, edgecolor='#555555', facecolor='white')
        ax.add_patch(sub_rect)
        ax.text(abs_x + sw / 2, abs_y + sh / 2, stext, ha='center', va='center', fontsize=9, wrap=True)


# === 1. 表现层 (Top) ===
presentation_subs = [
    (0.5, 0.4, 1.8, 0.8, "Web 浏览器\n(HTML5/JS)"),
    (2.8, 0.4, 1.8, 0.8, "ECharts\n图可视化"),
]
# 调整到顶部附近
draw_layer(1.5, 7.8, 7.5, 1.6, "表现层", '#e3f2fd', presentation_subs)

# Arrow Down - 调整箭头位置
ax.annotate("", xy=(5.25, 7.2), xytext=(5.25, 7.8), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(5.4, 7.3, "REST API / WebSocket", fontsize=8)

# === 2. 业务逻辑层 ===
logic_subs = [
    (0.5, 0.4, 1.8, 0.8, "FastAPI\n接口服务"),
    (2.8, 0.4, 1.8, 0.8, "WebSocket\n实时通信"),
    (5.1, 0.4, 1.8, 0.8, "用户鉴权")
]
# 均匀分布到中部偏上
draw_layer(1.5, 5.6, 7.5, 1.6, "业务逻辑层", '#fff3e0', logic_subs)

# Arrow Down - 调整箭头位置
ax.annotate("", xy=(5.25, 5.0), xytext=(5.25, 5.6), arrowprops=dict(arrowstyle="->", lw=2))

# === 3. 核心算法层 (Core) - Highlight ===
algo_subs = [
    # Left: Construction
    (0.3, 0.3, 3.2, 1.0, "图谱增量构建引擎"),
    # Right: Retrieval
    (3.8, 0.3, 3.4, 1.0, "拓扑感知检索引擎")
]
# 均匀分布到中部偏下
draw_layer(1.5, 3.4, 7.5, 1.8, "核心算法层", '#e8f5e9', algo_subs)

# Arrow Down - 调整箭头位置
ax.annotate("", xy=(5.25, 2.8), xytext=(5.25, 3.4), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(5.4, 2.9, "I/O", fontsize=8)


# === 4. 数据存储层 ===
data_subs = [
    (0.3, 0.3, 1.5, 0.8, "非结构化\n文档存储"),
    (2.1, 0.3, 1.5, 0.8, "FAISS\n向量索引\n"),
    (3.9, 0.3, 1.5, 0.8, "知识图谱\n存储\n(JSON)"),
    (5.7, 0.3, 1.4, 0.8, "用户\n数据库\n(SQL)")
]
# 均匀分布到底部
draw_layer(1.5, 1.2, 7.5, 1.6, "数据存储层", '#f3e5f5', data_subs)

# Title
# ax.text(5, 0.5, "图 5-3 系统总体架构设计", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('system_architecture_diagram.png', dpi=300, bbox_inches='tight')
print("Image generated.")