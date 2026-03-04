import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# 字体设置
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK JP', 'sans-serif']
except:
    pass

# 辅助绘图函数
def draw_box(x, y, w, h, text, color, edge, fontsize=10, style='round,pad=0.1'):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=style,
                                 linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#444444'))
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.1, text, ha='center', fontsize=8, color='#444444', backgroundcolor='#ffffff80')

# --- 1. 左侧：问题分解 ---
draw_box(0.5, 6.5, 2.5, 1.0, "用户复杂查询", '#fff9c4', '#fbc02d')
draw_arrow(1.75, 6.5, 1.75, 5.5)

draw_box(0.5, 4.0, 2.5, 1.5, "Agentic Decomposer", '#e1bee7', '#8e24aa')
ax.text(1.75, 4.3, "子问题 1\n子问题 2...", ha='center', fontsize=8)

# --- 2. 中间：双模态 Prompt 构建 ---
draw_arrow(3.0, 4.75, 4.0, 4.75, "迭代检索")

# 大框：Prompt Context
prompt_bg = patches.Rectangle((4.0, 2.0), 5.0, 5.5, linewidth=2, edgecolor='#1565c0', facecolor='#e3f2fd', linestyle='--')
ax.add_patch(prompt_bg)
ax.text(6.5, 7.2, "提示词", ha='center', fontweight='bold', color='#0d47a1')

# 内部结构：Triple Layer
draw_box(4.5, 5.5, 4.0, 1.2, "逻辑层\n[Knowledge Graph Triples]\n(头节点A, caused_by, 尾节点B)...", '#bbdefb', '#1976d2', fontsize=9)

# 内部结构：Content Layer
draw_box(4.5, 3.5, 4.0, 1.2, "语义层\n[Original Text Chunks]\n\"...检查步骤：...\"", '#c8e6c9', '#388e3c', fontsize=9)

# 内部结构：Instruction
draw_box(4.5, 2.2, 4.0, 0.8, "归因指令 (Attribution)\n\"你是使用迭代检索和链式思维推理的知识助手\"", '#ffccbc', '#d84315', fontsize=9)

# --- 3. 右侧：LLM 推理 ---
draw_arrow(9.0, 4.75, 10.0, 4.75, "注入")

draw_box(10.0, 3.5, 3.0, 2.5, "大语言模型\n(IRCoT Reasoning)", '#fff9c4', '#fbc02d')

# 循环箭头 (Reflection)
ax.add_patch(patches.FancyArrowPatch((11.5, 6.0), (1.75, 5.5),
                                     connectionstyle="arc3,rad=0.4",
                                     arrowstyle='simple,head_length=10,head_width=10,tail_width=1', color='black', linestyle='--'))
ax.text(6.5, 8, "信息不足 -> 补充检索", ha='center', fontsize=10, fontweight='bold', color='#1565c0')
# 修正文字位置
ax.text(6.5, 6.5, "反思与新查询", ha='center', fontsize=9, color='#616161')


# 输出
draw_arrow(11.5, 3.5, 11.5, 2.0)
draw_box(10.5, 1.0, 2.0, 1.0, "最终答案", '#d1c4e9', '#512da8')

# 标题
# ax.text(7, 0.2, "图 4-7 基于 IRCoT 的拓扑结构注入与答案生成框架", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('ircot_framework_detailed.png', dpi=300, bbox_inches='tight')
print("Image saved.")