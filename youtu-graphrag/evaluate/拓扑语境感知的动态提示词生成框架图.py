import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === 1. Create Canvas ===
fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
c_input = '#fff9c4'   # Yellowish
c_prompt = '#e3f2fd'  # Light Blue
c_static = '#e0e0e0'  # Grey for static parts
c_dynamic = '#ffccbc' # Orange/Reddish for dynamic injection
c_llm = '#d1c4e9'     # Purple
c_output = '#c8e6c9'  # Green
c_arrow = '#424242'

# Font setup
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS', 'sans-serif']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

# Helper functions
def draw_box(x, y, w, h, text, color, edge_color, fontsize=10, fontweight='normal'):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                 linewidth=1.5, edgecolor=edge_color, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, wrap=True)
    return box

def draw_arrow(x1, y1, x2, y2, text=None, style="-|>", color=c_arrow):
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                    color=color, lw=1.5, mutation_scale=15)
    ax.add_patch(arrow)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', va='bottom', fontsize=9)

# === 2. Left Side: Inputs ===
# Box 1: Retrieved Triples
draw_box(0.5, 5.5, 3.5, 1.5, "检索到的子图三元组\n(Retrieved Subgraph Triples)\n\n[A, rel, B]\n[C, rel, A]...", c_input, '#fbc02d')

# Box 2: New Document Chunk
draw_box(0.5, 1.5, 3.5, 1.5, "新文档片段\n(New Document Chunk)\n\n\"...涡轮叶片...\"", c_input, '#fbc02d')

# === 3. Middle: Prompt Template Construction ===
# Big Container for Prompt
prompt_bg = patches.Rectangle((5.5, 0.5), 5.0, 7.0, linewidth=2, edgecolor='#1e88e5', facecolor='#f5f5f5', linestyle='--')
ax.add_patch(prompt_bg)
ax.text(8.0, 7.7, "提示词模板 (Prompt Template)", ha='center', fontsize=12, fontweight='bold', color='#1565c0')

# Layer 1: Role & Task (Static)
draw_box(6.0, 6.0, 4.0, 1.0, "1. 角色与任务指令 (Task)\n(Static: Expert Role)", c_static, '#9e9e9e')

# Layer 2: Schema (Static)
draw_box(6.0, 4.5, 4.0, 1.0, "2. Schema 约束定义\n(Static: JSON Schema)", c_static, '#9e9e9e')

# Layer 3: Topology Context (Dynamic) - The Injection Point
draw_box(6.0, 3.0, 4.0, 1.0, "3. 拓扑语境槽位 (Context)\n[Dynamic Injection Point]", c_dynamic, '#d84315', fontweight='bold')

# Layer 4: Input Data (Dynamic)
draw_box(6.0, 1.5, 4.0, 1.0, "4. 输入文本槽位 (Input)\n[Dynamic Injection Point]", c_dynamic, '#d84315', fontweight='bold')

# === 4. Arrows connecting Inputs to Prompt ===
# Arrow from Triples to Layer 3
draw_arrow(4.1, 6.25, 5.9, 3.5, color='#d84315')
ax.text(5.0, 5.0, "动态注入\n(Inject)", ha='center', color='#d84315', fontweight='bold')

# Arrow from Chunk to Layer 4
draw_arrow(4.1, 2.25, 5.9, 2.0, color='#d84315')

# === 5. Right Side: LLM & Output ===
# Arrow from Prompt to LLM
draw_arrow(10.6, 4.0, 12.0, 4.0, text="发送请求 (Send)")

# LLM Box
draw_box(12.0, 3.0, 1.5, 2.0, "LLM\n(大模型)", c_llm, '#673ab7', fontweight='bold')

# Arrow from LLM to Result
draw_arrow(13.6, 4.0, 14.5, 4.0)

# Output Box
draw_box(14.5, 3.25, 1.0, 1.5, "结构化\n结果", c_output, '#388e3c')

# Title
# ax.set_title("图 3-4 拓扑语境感知的动态提示词生成框架", fontsize=16, y=0.98)

# Save
plt.tight_layout()
plt.savefig('dynamic_prompt_framework_hd.png', dpi=300, bbox_inches='tight')
print("Image saved.")