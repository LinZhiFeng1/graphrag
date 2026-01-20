import json
import networkx as nx


def load_edges(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    edges = set()
    if isinstance(data, list):
        for item in data:
            u = item.get("start_node", {}).get("properties", {}).get("name")
            v = item.get("end_node", {}).get("properties", {}).get("name")
            rel = item.get("relation", "related")
            if u and v:
                # 使用排序元组保证无向对比 (u, v) == (v, u)
                # 如果是有向图，去掉 sorted
                edge_key = tuple(sorted([u, v]))
                edges.add(edge_key)
    return edges


def analyze_diff():
    print("🚀 正在对比 Baseline 与 Ours 的图谱差异...")

    # 1. 加载边集合
    base_edges = load_edges("evaluate/graph_baseline.json")  # 请确保文件名正确
    our_edges = load_edges("evaluate/graph_ours.json")  # 请确保文件名正确

    print(f"📊 Baseline 边数: {len(base_edges)}")
    print(f"📊 Ours     边数: {len(our_edges)}")


    # 2. 计算 Ours 独有的边 (Baseline 没发现，但 Ours 发现了)
    unique_edges = our_edges - base_edges
    print(f"\n✨ Ours 挖掘出的【独有新关系】: {len(unique_edges)} 条")

    # 3. 打印前 10 条看看 (这可是论文里的黄金案例！)
    print("\n🔎 独有关系示例 (Top 10):")
    for i, (u, v) in enumerate(list(unique_edges)[:10]):
        print(f"  {i + 1}. {u} <---> {v}")

    # 4. 计算密度提升
    # 假设节点数近似，直接比边数
    improvement = (len(our_edges) - len(base_edges)) / len(base_edges) * 100
    print(f"\n📈 关系丰富度提升: {improvement:.2f}%")


if __name__ == "__main__":
    analyze_diff()