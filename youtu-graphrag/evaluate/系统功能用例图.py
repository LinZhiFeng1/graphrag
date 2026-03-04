import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# 字体设置
try:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK JP', 'sans-serif']
except:
    pass

# --- 绘图函数定义 ---
def draw_actor(x, y, label):
    # 头
    circle = patches.Circle((x, y), 0.2, edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(circle)
    # 身体
    ax.plot([x, x], [y-0.2, y-0.8], 'k-', linewidth=1.5)
    # 手
    ax.plot([x-0.3, x+0.3], [y-0.4, y-0.4], 'k-', linewidth=1.5)
    # 脚
    ax.plot([x, x-0.3], [y-0.8, y-1.2], 'k-', linewidth=1.5)
    ax.plot([x, x+0.3], [y-0.8, y-1.2], 'k-', linewidth=1.5)
    # 标签
    ax.text(x, y-1.5, label, ha='center', fontweight='bold', fontsize=10)

def draw_usecase(x, y, label, width=2.2, height=0.7):
    ellipse = patches.Ellipse((x, y), width, height, edgecolor='#1565c0', facecolor='#e3f2fd', linewidth=1.5)
    ax.add_patch(ellipse)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, wrap=True)
    return (x, y)

def connect(p1, p2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=0.8)

# --- 1. 绘制角色 ---
actor_user_pos = (1.5, 4.5)
draw_actor(*actor_user_pos, "机务维修人员\n(User)")

actor_admin_pos = (8.5, 4.5)
draw_actor(*actor_admin_pos, "系统管理员\n(Admin)")

# --- 2. 绘制系统边界 ---
rect = patches.Rectangle((2.3, 1.7), 5.4, 4.5, linewidth=1.5, edgecolor='#757575', facecolor='none', linestyle='--')
ax.add_patch(rect)
ax.text(5.0, 6.4, "领域知识图谱问答系统", ha='center', fontsize=12, fontweight='bold')

# --- 3. 绘制用户用例 ---
# 登录 (公共)
uc_login = draw_usecase(5.0, 5.5, "用户/管理员登录")

# 用户特有 - 调整位置避免重叠
uc_qa = draw_usecase(3.5, 4.2, "智能故障问答")      # 向左移动，降低高度
uc_trace = draw_usecase(3.5, 3.2, "查看溯源证据")   # 向左移动，降低高度
uc_viz = draw_usecase(3.5, 2.2, "图谱可视化交互")   # 向左移动，降低高度

# --- 4. 绘制管理员用例 ---
uc_upload = draw_usecase(6.5, 4.2, "上传维护手册")   # 向右移动，降低高度
uc_build = draw_usecase(6.5, 3.2, "图谱构建与更新")  # 向右移动，降低高度
uc_monitor = draw_usecase(6.5, 2.2, "系统状态监控")  # 向右移动，降低高度


# --- 5. 连线 ---
# 用户连线
user_hand = (1.8, 4.1) # 手部大概位置
connect(user_hand, (3.9, 5.5)) # Login
connect(user_hand, (2.9, 4.5)) # QA
connect(user_hand, (2.9, 3.5)) # Trace
connect(user_hand, (2.9, 2.5)) # Viz

# 管理员连线
admin_hand = (8.2, 4.1)
connect(admin_hand, (6.1, 5.5)) # Login
connect(admin_hand, (7.1, 4.5)) # Upload
connect(admin_hand, (7.1, 3.5)) # Build
connect(admin_hand, (7.1, 2.5)) # Monitor

# 包含关系 (例如 QA 包含 查看溯源)
# ax.text(4.0, 4.0, "<<include>>", ha='center', fontsize=8, rotation=90)
# ax.annotate("", xy=(4.0, 3.85), xytext=(4.0, 4.15), arrowprops=dict(arrowstyle="->", linestyle="dashed"))

# 标题
# ax.text(5, 0.1, "图 5-2 系统功能用例图", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('system_usecase_diagram.png', dpi=300, bbox_inches='tight')
print("Image generated.")