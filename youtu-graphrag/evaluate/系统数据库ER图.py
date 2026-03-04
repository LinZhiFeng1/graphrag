import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# 字体设置
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK JP', 'sans-serif']
except:
    pass

# 绘图辅助函数：绘制实体（矩形）
def draw_entity(x, y, text, color='#e3f2fd'):
    rect = patches.FancyBboxPatch((x, y), 2.0, 1.0, boxstyle='round,pad=0.1',
                                  linewidth=1.5, edgecolor='#1565c0', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + 1.0, y + 0.5, text, ha='center', va='center', fontsize=10, fontweight='bold')
    return (x + 1.0, y + 0.5)

# 绘图辅助函数：绘制关系（菱形）
def draw_relation(x, y, text):
    # Diamond shape using Polygon
    diamond = patches.Polygon([[x, y+0.4], [x+0.8, y], [x, y-0.4], [x-0.8, y]],
                              closed=True, linewidth=1.5, edgecolor='#333333', facecolor='#fff9c4')
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=9)
    return (x, y)

# 绘图辅助函数：连接线
def connect(p1, p2, text=None):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1, zorder=0)
    if text:
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(mid_x, mid_y, text, fontsize=8, backgroundcolor='white')

# 1. 实体节点
pos_user = draw_entity(0.5, 4.0, "用户\n(User)")
pos_kb = draw_entity(4.0, 4.0, "知识库\n(KnowledgeBase)")
pos_doc = draw_entity(7.5, 4.0, "文档\n(Document)")
pos_session = draw_entity(4.0, 1.0, "问答会话\n(ChatSession)")
pos_msg = draw_entity(7.5, 1.0, "问答消息\n(ChatMessage)")

# 2. 关系节点
pos_rel_create = draw_relation(2.5, 4.5, "创建")
pos_rel_contain_doc = draw_relation(6.25, 4.5, "包含")
pos_rel_start = draw_relation(2.5, 2.5, "发起")
pos_rel_belong = draw_relation(4.0, 2.8, "隶属") # Session belongs to KB
pos_rel_contain_msg = draw_relation(6.25, 1.5, "包含")

# 3. 连线
# User - Create - KB
connect(pos_user, pos_rel_create, "1")
connect(pos_rel_create, pos_kb, "N")

# KB - Contain - Doc
connect(pos_kb, pos_rel_contain_doc, "1")
connect(pos_rel_contain_doc, pos_doc, "N")

# User - Start - Session
connect(pos_user, pos_rel_start, "1")
connect(pos_rel_start, pos_session, "N")

# KB - Belong - Session
connect(pos_kb, pos_rel_belong, "1")
connect(pos_rel_belong, pos_session, "N")

# Session - Contain - Message
connect(pos_session, pos_rel_contain_msg, "1")
connect(pos_rel_contain_msg, pos_msg, "N")

# 标题
# ax.text(5, 0.1, "图 5-5 系统数据库实体关系ER图", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('system_er_diagram.png', dpi=300, bbox_inches='tight')
print("Image generated.")