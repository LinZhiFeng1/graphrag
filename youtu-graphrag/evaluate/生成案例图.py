import json
import matplotlib.pyplot as plt
import networkx as nx
import textwrap
import matplotlib.font_manager as fm
import platform


# === 1. 设置中文字体 (核心步骤) ===
def find_chinese_font():
    """自动查找系统中可用的中文字体"""
    system = platform.system()

    if system == "Windows":
        # Windows 常见中文字体
        possible_fonts = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',  # 黑体
            'SimSun',  # 宋体
            'KaiTi',  # 楷体
            'NSimSun',  # 新宋体
        ]
        # 检查这些字体是否可用
        available_fonts = [f.name for f in fm.fontManager.ttflist if f.name in possible_fonts]
        if available_fonts:
            return available_fonts[0]

        # 如果上面的字体都不可用，尝试查找包含中文的字体
        chinese_fonts = [f.name for f in fm.fontManager.ttflist if
                         any(keyword in f.name.lower() for keyword in
                             ['chinese', 'song', 'hei', 'kai', 'yahei', 'sim'])]
        if chinese_fonts:
            return chinese_fonts[0]

    elif system == "Darwin":  # macOS
        # macOS 常见中文字体
        possible_fonts = [
            'PingFang SC',
            'Heiti SC',
            'Hiragino Sans GB',
            'STSong',
            'STHeiti'
        ]
        available_fonts = [f.name for f in fm.fontManager.ttflist if f.name in possible_fonts]
        if available_fonts:
            return available_fonts[0]

    else:  # Linux
        # Linux 常见中文字体
        possible_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Noto Sans CJK TC',
            'Source Han Sans SC',
            'Source Han Sans TC'
        ]
        available_fonts = [f.name for f in fm.fontManager.ttflist if f.name in possible_fonts]
        if available_fonts:
            return available_fonts[0]

    # 如果都没找到，返回 None
    return None


# 查找字体
font_name = find_chinese_font()

if font_name:
    print(f"✅ 找到中文字体: {font_name}")
    plt.rcParams['font.family'] = font_name
else:
    print("⚠️ 未找到合适的中文字体，使用默认字体")
    # 设置多个备选字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
    font_name = 'DejaVu Sans'  # 使用英文作为后备

plt.rcParams['axes.unicode_minus'] = False

# === 2. 加载数据 ===
# 请确保这两个 json 文件在当前目录下
with open('evaluate/graph_baseline.json', 'r', encoding='utf-8') as f:
    data_baseline = json.load(f)
with open('evaluate/graph_ours.json', 'r', encoding='utf-8') as f:
    data_ours = json.load(f)


def build_graph(data):
    G = nx.MultiDiGraph()
    for item in data:
        s = item['start_node']['properties']['name']
        t = item['end_node']['properties']['name']
        r = item['relation']
        s_label = item['start_node']['label']
        t_label = item['end_node']['label']
        G.add_node(s, label=s_label)
        G.add_node(t, label=t_label)
        G.add_edge(s, t, relation=r)
    return G


G_base = build_graph(data_baseline)
G_ours = build_graph(data_ours)

# 目标中心节点
TARGET_NODE = "ГТ60ПЧ6А额定转速"


# === 3. 辅助函数：提取子图 & 文本换行 ===
def get_ego_subgraph(G, root):
    if not G.has_node(root): return nx.MultiDiGraph()
    nodes = set([root]) | set(G.neighbors(root)) | set(G.predecessors(root))
    return G.subgraph(nodes)


def wrap_text(text, width=8):
    """超过 width 个字符就换行"""
    return "\n".join(textwrap.wrap(text, width=width))


# === 4. 绘图逻辑 ===
if G_base.has_node(TARGET_NODE) and G_ours.has_node(TARGET_NODE):
    sub_base = get_ego_subgraph(G_base, TARGET_NODE)
    sub_ours = get_ego_subgraph(G_ours, TARGET_NODE)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=300)  # 画布调大


    def draw_sub(ax, G, title):
        # 布局：k值越大，节点越分散
        pos = nx.spring_layout(G, k=2.5, seed=42)

        # 节点颜色
        node_colors = []
        for n in G.nodes():
            if n == TARGET_NODE:
                node_colors.append('#ff7f0e')  # 橙色中心
            elif G.nodes[n].get('label') == 'community':
                node_colors.append('#2ca02c')  # 绿色社区
            else:
                node_colors.append('#1f77b4')  # 蓝色普通

        # 画节点：增大尺寸
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500, alpha=0.9, edgecolors='white')

        # 画标签：应用换行函数
        labels = {n: wrap_text(n, width=8) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_family=font_name, font_size=6, font_weight='bold')

        # 画边
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.5, arrowsize=20,
                               connectionstyle="arc3,rad=0.1")

        # 边标签
        edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_family=font_name, font_size=6)

        # 这里也需要修复：my_font未定义，应该创建FontProperties对象
        import matplotlib.font_manager as fm_module
        try:
            font_prop = fm_module.FontProperties(family=font_name, size=20)
            ax.set_title(title, fontsize=20, fontproperties=font_prop, pad=20)
        except:
            ax.set_title(title, fontsize=20, fontfamily=font_name, pad=20)

        ax.margins(0.2)  # 增加边缘留白，防止被切
        ax.axis('off')


    draw_sub(axes[0], sub_base, f"Baseline (基准方法): {TARGET_NODE}")
    draw_sub(axes[1], sub_ours, f"Ours (本文方法): {TARGET_NODE}")

    plt.tight_layout()
    plt.savefig('case_study_comparison.png')  # 保存图片
    print("✅ 图片已保存为 case_study_comparison.png")
    plt.show()
else:
    print(f"❌ 图谱中未找到节点: {TARGET_NODE}")
