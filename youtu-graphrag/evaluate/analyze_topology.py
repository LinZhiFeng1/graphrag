import json
import networkx as nx


def evaluate_graph(name, file_path):
    print(f"正在分析: {name}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        g = nx.Graph()
        # 解析你的JSON结构
        if isinstance(data, list):
            for item in data:
                u = item.get("start_node", {}).get("properties", {}).get("name")
                v = item.get("end_node", {}).get("properties", {}).get("name")
                if u and v: g.add_edge(u, v)

        # 计算指标
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()

        # 1. 连通分量数 (越少越好，说明融合紧密)
        components = list(nx.connected_components(g))
        num_components = len(components)

        # 2. 最大连通子图占比 (越大越好，说明形成了核心知识网)
        largest_cc_size = len(max(components, key=len)) if components else 0
        largest_ratio = largest_cc_size / num_nodes if num_nodes > 0 else 0

        print(f"  - 节点数: {num_nodes}, 边数: {num_edges}")
        print(f"  - 连通分量数: {num_components}")
        print(f"  - 最大子图占比: {largest_ratio:.2%}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"❌ 找不到文件: {file_path}")


if __name__ == "__main__":
    evaluate_graph("基准方法 (Baseline)", "evaluate/graph_baseline.json")
    evaluate_graph("本文方法 (Ours)", "evaluate/graph_ours.json")