import json
import random
import networkx as nx
from collections import defaultdict, deque


def load_graph_from_json(json_file_path):
    """
    从JSON文件加载图谱数据

    Args:
        json_file_path: 图谱JSON文件路径

    Returns:
        NetworkX MultiDiGraph对象
    """
    # 创建有向多重图
    graph = nx.MultiDiGraph()

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # 构建图结构
    for item in graph_data:
        start_node = item['start_node']
        end_node = item['end_node']
        relation = item['relation']

        # 提取节点信息
        start_name = start_node['properties']['name']
        start_label = start_node['label']
        start_props = start_node['properties']

        end_name = end_node['properties']['name']
        end_label = end_node['label']
        end_props = end_node['properties']

        # 添加节点
        if start_name not in graph.nodes:
            graph.add_node(start_name, label=start_label, properties=start_props)

        if end_name not in graph.nodes:
            graph.add_node(end_name, label=end_label, properties=end_props)

        # 添加边
        graph.add_edge(start_name, end_name, relation=relation)

    return graph


def random_walk_subgraph(graph, start_node=None, hops=2):
    """
    从指定节点开始进行随机游走，生成子图

    Args:
        graph: NetworkX图对象
        start_node: 起始节点名称，如果为None则随机选择
        hops: 游走步数（hop数）

    Returns:
        tuple: (子图节点集合, 子图边集合, 起始节点)
    """
    if start_node is None:
        # 随机选择一个节点作为起始点
        start_node = random.choice(list(graph.nodes()))
        print(f"随机选择起始节点: {start_node}")
    else:
        if start_node not in graph.nodes:
            raise ValueError(f"节点 '{start_node}' 不存在于图中")

    # 使用BFS进行hops步的遍历
    visited_nodes = set()
    visited_edges = set()
    queue = deque([(start_node, 0)])  # (节点, 当前步数)
    visited_nodes.add(start_node)

    while queue:
        current_node, current_hop = queue.popleft()

        # 如果达到指定hop数，停止扩展
        if current_hop >= hops:
            continue

        # 获取当前节点的所有邻居（包括入边和出边）
        neighbors = set()

        # 出边邻居
        for neighbor in graph.successors(current_node):
            neighbors.add(neighbor)
            edge_key = (current_node, neighbor)
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)

        # 入边邻居
        for neighbor in graph.predecessors(current_node):
            neighbors.add(neighbor)
            edge_key = (neighbor, current_node)
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)

        # 随机打乱邻居节点顺序
        neighbor_list = list(neighbors)
        random.shuffle(neighbor_list)

        # 将未访问的邻居加入队列
        for neighbor in neighbor_list:
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, current_hop + 1))

    return visited_nodes, visited_edges, start_node


def extract_subgraph_triples(graph, nodes, edges):
    """
    从子图中提取所有三元组

    Args:
        graph: 原始图对象
        nodes: 子图节点集合
        edges: 子图边集合

    Returns:
        list: 三元组列表，格式为 [(主体, 关系, 客体), ...]
    """
    triples = []

    # 遍历所有边，提取三元组
    for u, v in edges:
        # 获取节点间的所有边（可能有多重边）
        edge_data = graph.get_edge_data(u, v)
        if edge_data:
            for key, data in edge_data.items():
                relation = data.get('relation', 'related_to')
                triples.append((u, relation, v))

    return triples


def format_triples_for_display(triples):
    """
    格式化三元组用于显示

    Args:
        triples: 三元组列表

    Returns:
        str: 格式化的三元组字符串
    """
    formatted_triples = []
    for i, (subject, relation, object_) in enumerate(triples, 1):
        formatted_triples.append(f"{i}. [{subject}, {relation}, {object_}]")

    return "\n".join(formatted_triples)


def main():
    """
    主函数：演示随机游走子图生成功能
    """
    # 配置参数
    json_file_path = r"output/graphs/aviation_new.json"
    hops = 2  # 2-hop游走

    print("=" * 60)
    print("航空发动机故障诊断知识图谱 - 随机游走子图生成")
    print("=" * 60)

    try:
        # 1. 加载图谱
        print("正在加载图谱...")
        graph = load_graph_from_json(json_file_path)
        print(f"图谱加载完成！节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")

        # 2. 随机游走生成子图
        print(f"\n正在进行 {hops}-hop 随机游走...")
        subgraph_nodes, subgraph_edges, start_node = random_walk_subgraph(graph, hops=hops)

        print(f"起始节点: {start_node}")
        print(f"子图节点数: {len(subgraph_nodes)}")
        print(f"子图边数: {len(subgraph_edges)}")

        # 3. 提取三元组
        print("\n提取子图中的三元组...")
        triples = extract_subgraph_triples(graph, subgraph_nodes, subgraph_edges)

        # 4. 显示结果
        print(f"\n共提取到 {len(triples)} 个三元组:")
        print("-" * 40)
        print(format_triples_for_display(triples))

        # 5. 保存结果（可选）
        result_data = {
            "start_node": start_node,
            "subgraph_nodes": list(subgraph_nodes),
            "subgraph_edges": list(subgraph_edges),
            "triples": triples,
            "statistics": {
                "total_nodes": len(subgraph_nodes),
                "total_edges": len(subgraph_edges),
                "total_triples": len(triples)
            }
        }

        output_file = "random_walk_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_file}")

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


# 高级功能：多次随机游走比较
def multiple_random_walks(graph, num_walks=5, hops=2):
    """
    执行多次随机游走并比较结果

    Args:
        graph: 图对象
        num_walks: 游走次数
        hops: 每次游走的hop数
    """
    print(f"\n执行 {num_walks} 次随机游走比较:")
    print("=" * 50)

    results = []
    for i in range(num_walks):
        print(f"\n第 {i + 1} 次游走:")
        nodes, edges, start_node = random_walk_subgraph(graph, hops=hops)
        triples = extract_subgraph_triples(graph, nodes, edges)

        result = {
            "walk_id": i + 1,
            "start_node": start_node,
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "triples_count": len(triples),
            "triples": triples
        }
        results.append(result)

        print(f"  起始节点: {start_node}")
        print(f"  节点数: {len(nodes)}, 边数: {len(edges)}, 三元组数: {len(triples)}")

    return results


if __name__ == "__main__":
    # 运行基本功能
    main()

    # 如果想要比较多次游走结果，取消下面的注释
    # print("\n" + "="*60)
    # print("多次随机游走比较")
    # print("="*60)
    #
    # graph = load_graph_from_json(r"D:\桌面文件\Program\youtu-graphrag\youtu-graphrag\output\graphs\aviation_new.json")
    # multiple_results = multiple_random_walks(graph, num_walks=3, hops=2)
