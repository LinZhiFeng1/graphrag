import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# 字体设置
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

def draw_arrow(x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#444444'))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', fontsize=9, color='#444444')

# --- 1. 左侧：活跃子图输入 ---
draw_box(0.5, 2.5, 2.0, 2.0, "活跃子图\n(Active Subgraph)", '#b3e5fc', '#1976d2')

# --- 2. 中间：双路处理 ---
# 上路：三元组提取
draw_arrow(2.5, 4.0, 3.5, 5.5)
draw_box(3.5, 5.0, 2.5, 1.0, "三元组线性化\n(Linearization)", '#fff9c4', '#fbc02d')
draw_arrow(6.0, 5.5, 7.0, 5.5, "提取边属性")

draw_box(7.0, 5.0, 2.5, 1.0, "结构化列表\nList[(S, R, O)]", '#fff9c4', '#fbc02d')

# 下路：文本溯源
draw_arrow(2.5, 3.0, 3.5, 1.5)
draw_box(3.5, 1.0, 2.5, 1.0, "源文本溯源\n(Chunk Retrieval)", '#e1bee7', '#8e24aa')
draw_arrow(6.0, 1.5, 7.0, 1.5, "索引 Chunk ID")

draw_box(7.0, 1.0, 2.5, 1.0, "原始文档片段\nList[Text Chunks]", '#e1bee7', '#8e24aa')

# --- 3. 右侧：Prompt 组装 ---
# 汇聚
draw_arrow(9.5, 5.5, 10.5, 4.0)
draw_arrow(9.5, 1.5, 10.5, 3.0)

# Prompt 框
prompt_bg = patches.Rectangle((10.5, 1.5), 3.0, 4.0, linewidth=1.5, edgecolor='#616161', facecolor='#f5f5f5', linestyle='--')
ax.add_patch(prompt_bg)
ax.text(12.0, 5.2, "Final Prompt", ha='center', fontweight='bold')

# 内容模拟
ax.text(10.7, 4.5, "[KG Context]:\n- (MKT-372, voltage, ...)\n- (Engine, speed, ...)", fontsize=8, va='top')
ax.text(10.7, 3.0, "[Doc Context]:\n\"...check MKT-372 when\nengine speed reaches...\"", fontsize=8, va='top')

# 标题
# ax.text(7, 6.5, "图 4-6 子图语境化与 Prompt 文本重组流程", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('contextualization_flow.png', dpi=300, bbox_inches='tight')
print("Image saved.")