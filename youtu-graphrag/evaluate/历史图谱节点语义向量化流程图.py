import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import matplotlib.lines as lines

# =================配置区域=================
# 设置画布大小
fig, ax = plt.subplots(figsize=(22, 8))
ax.set_xlim(0, 22)
ax.set_ylim(0, 10)
ax.axis('off')  # 关闭坐标轴

# 【关键】设置中文字体
# 如果你在 Windows 上，通常可以使用 'SimHei' 或 'Microsoft YaHei'
# 如果在 Mac/Linux，可能需要指定具体路径，例如 '/System/Library/Fonts/PingFang.ttc'
try:
    # 尝试使用系统默认黑体
    font_title = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14) # Windows示例
    font_text = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)
    font_small = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=9)
except:
    # 如果找不到，回退到默认（中文可能会乱码，请手动指定上面的路径）
    font_title = FontProperties(size=14)
    font_text = FontProperties(size=10)
    font_small = FontProperties(size=9)

# 颜色定义
c_blue_light = '#DAE8FC'
c_blue_stroke = '#6C8EBF'
c_grey_light = '#F5F5F5'
c_grey_stroke = '#B3B3B3'
c_yellow_light = '#FFF2CC'
c_yellow_stroke = '#D6B656'
c_green_light = '#D5E8D4'
c_green_stroke = '#82B366'
c_purple_light = '#E1D5E7'
c_purple_stroke = '#9673A6'

# =================绘图函数封装=================

def draw_circle(x, y, radius, text, color, stroke, dashed=False):
    style = '--' if dashed else '-'
    circle = patches.Circle((x, y), radius, linewidth=1.5, edgecolor=stroke, facecolor=color, linestyle=style)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontproperties=font_text)
    return (x, y)

def draw_box(x, y, w, h, text, color, stroke, title=""):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                 linewidth=1.5, edgecolor=stroke, facecolor=color)
    ax.add_patch(box)
    # 居中文字
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontproperties=font_text)
    # 顶部标题（如果有）
    if title:
        ax.text(x + w/2, y + h + 0.3, title, ha='center', va='bottom', fontproperties=font_title)

def draw_arrow(x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    if text:
        ax.text((x1+x2)/2, y1 + 0.2, text, ha='center', va='bottom', fontproperties=font_small)

# =================第1部分：左侧 历史知识图谱 (History Graph)=================

# 虚线外框
rect_frame = patches.Rectangle((0.5, 2), 5, 5.5, linewidth=1, edgecolor='#CCCCCC', facecolor='none', linestyle='--')
ax.add_patch(rect_frame)
ax.text(3, 8, "输入: 历史知识图谱", ha='center', fontproperties=font_title)

# 节点
p1 = draw_circle(3, 6, 0.75, "实体\n高压压气机", c_blue_light, c_blue_stroke)
p2 = draw_circle(1.8, 3.5, 0.75, "实体\n燃油泵", c_blue_light, c_blue_stroke)
p3 = draw_circle(4.2, 3.5, 0.75, "属性\n12000rpm", 'white', c_grey_stroke, dashed=True)

# 节点连线
draw_arrow(p1[0], p1[1]-0.75, p2[0], p2[1]+0.75)
draw_arrow(p1[0], p1[1]-0.75, p3[0], p3[1]+0.75)


# =================第2部分：过滤器 (Filter)=================

# 漏斗形状 (用多边形绘制)
# 坐标点: 左上, 右上, 右下缩进点, 左下缩进点
funnel_x = 7
funnel_y = 6
funnel_w = 2.5
funnel_h = 2.5
verts = [
    (funnel_x, funnel_y), # 左上
    (funnel_x + funnel_w, funnel_y), # 右上
    (funnel_x + funnel_w*0.7, funnel_y - funnel_h), # 右下颈部
    (funnel_x + funnel_w*0.7, funnel_y - funnel_h - 0.5), # 右下底部
    (funnel_x + funnel_w*0.3, funnel_y - funnel_h - 0.5), # 左下底部
    (funnel_x + funnel_w*0.3, funnel_y - funnel_h), # 左下颈部
]
poly = patches.Polygon(verts, closed=True, linewidth=1.5, edgecolor=c_yellow_stroke, facecolor=c_yellow_light)
ax.add_patch(poly)

ax.text(funnel_x + funnel_w/2, funnel_y - 1, "节点过滤器", ha='center', fontproperties=font_text)
ax.text(funnel_x + funnel_w/2, funnel_y + 0.2, "筛选条件: label == 'entity'", ha='center', color='#D6B656', fontproperties=font_small)

# 丢弃的属性
draw_circle(funnel_x + funnel_w/2, 2.5, 0.35, "属性", 'white', c_grey_stroke, dashed=True)
ax.text(funnel_x + funnel_w/2 + 0.6, 2.5, "丢弃", ha='left', va='center', color='grey', fontproperties=font_small)
# 虚线连接漏斗和丢弃属性
ax.plot([funnel_x + funnel_w/2, funnel_x + funnel_w/2], [3.0, 2.85], color='grey', linestyle='--')

# 连接 Graph -> Filter
draw_arrow(5.5, 4.8, 7, 4.8)


# =================第3部分：文本序列化 (Serialization)=================

box_x = 10.5
box_y = 2.6
box_w = 3.3
box_h = 2.8
draw_box(box_x, box_y, box_w, box_h, "文本序列化\n['高压压气机','燃油泵',...]", c_green_light, c_green_stroke)

# 连接 Filter -> Serialization
draw_arrow(8.75, 4.0, box_x, 4.0, "提取Name")


# =================第4部分：语义编码器 (Encoder)=================

enc_x = 14.8
enc_y = 2.6
enc_w = 3.3
enc_h = 2.8
draw_box(enc_x, enc_y, enc_w, enc_h, "语义编码器\n[BGE-M3]", c_purple_light, c_purple_stroke)

# 连接 Serialization -> Encoder
draw_arrow(box_x + box_w, 4.0, enc_x, 4.0)


# =================第5部分：稠密向量索引 (Vector Cache)=================

cache_x = 18.5
cache_y = 2.5
cache_w = 3.3
cache_h = 3.3

# 外框
rect_cache = patches.Rectangle((cache_x, cache_y), cache_w, cache_h, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(rect_cache)

# 标题
ax.text(cache_x + cache_w/2, cache_y + cache_h + 0.3, "稠密向量索引\n(Vector Cache)", ha='center', va='bottom', fontproperties=font_title)
ax.text(cache_x + 0.5, cache_y - 0.3, "IDs", ha='center', fontproperties=font_small)
ax.text(cache_x + 2.0, cache_y - 0.3, "Vectors (1024d)", ha='center', fontproperties=font_small)

# 绘制内部表格行
row_h = 0.5
margin_x = 0.2
for i in range(5):
    y_pos = cache_y + 0.3 + i * 0.6
    # ID block
    rect_id = patches.Rectangle((cache_x + margin_x, y_pos), 0.6, 0.4, facecolor='#F0F0F0', edgecolor='none')
    ax.add_patch(rect_id)
    # Vector block (simulated by segmented purple blocks)
    for j in range(4):
        rect_vec = patches.Rectangle((cache_x + 1.0 + j*0.5, y_pos), 0.45, 0.4, facecolor='#E1D5E7', edgecolor=c_purple_stroke)
        ax.add_patch(rect_vec)

# 连接 Encoder -> Cache
draw_arrow(enc_x + enc_w, 4.0, cache_x, 4.0, "L2 正则化")

plt.tight_layout()
plt.savefig('历史图谱节点语义向量化流程图.png', dpi=300, bbox_inches='tight')
plt.savefig('历史图谱节点语义向量化流程图.svg', format='svg', bbox_inches='tight')