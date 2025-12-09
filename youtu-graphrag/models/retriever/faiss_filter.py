import json
import os
import time
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

import faiss
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from utils.logger import logger

class DualFAISSRetriever:
    """
    基于 FAISS（Facebook AI Similarity Search）向量搜索引擎的知识图谱检索系统，它采用双路径检索策略来提高检索效果
    """
    # def __init__(self, dataset, graph: nx.MultiDiGraph, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "retriever/faiss_cache_new", device: str = None):
    def __init__(self, dataset, graph: nx.MultiDiGraph, model_name: str = "BAAI/bge-m3", cache_dir: str = "retriever/faiss_cache_new", device: str = None):
        """
        初始化 DualFAISSRetriever 实例

        :param dataset: 数据集名称，用于创建数据集特定的缓存目录
        :param graph: NetworkX MultiDiGraph 图对象，表示知识图谱
        :param model_name: 用于文本嵌入的 SentenceTransformer 模型名称，默认为 "all-MiniLM-L6-v2"
        :param cache_dir: FAISS 索引的缓存目录路径
        :param device: 计算设备 ("cuda" 或 "cpu")，默认为 None 表示自动检测
        """
        # 存储传入的图对象和数据集名称
        self.graph = graph
        # 初始化 SentenceTransformer 模型用于文本嵌入
        self.model = SentenceTransformer(model_name)
        # 设置缓存目录并确保其存在
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.dataset = dataset
        
        # 为特定数据集创建独立的缓存目录
        dataset_cache_dir = f"{self.cache_dir}/{self.dataset}"
        os.makedirs(dataset_cache_dir, exist_ok=True)

        # 初始化 FAISS 索引为 None，将在后续构建
        self.triple_index = None
        self.comm_index = None

        # 设备配置逻辑：根据传入参数和实际硬件情况确定计算设备
        if device is not None:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("Warning: CUDA requested but not available in DualFAISSRetriever, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"DualFAISSRetriever using device: {self.device}")

        # 初始化嵌入存储属性
        self.node_embeddings = None  # 节点嵌入
        self.relation_embeddings = None  # 关系嵌入
        self.node_id_to_embedding = {}  # 节点 ID 到嵌入的映射
        self.relation_to_embedding = {}  # 关系到嵌入的映射

        # 初始化索引映射属性，防止访问时出现 AttributeError
        self.node_map = {}  # 节点索引映射
        self.relation_map = {}  # 关系索引映射
        self.triple_map = {}  # 三元组索引映射
        self.comm_map = {}  # 社区索引映射

        # FAISS 搜索优化相关属性
        self.faiss_search_cache = {}  # FAISS 搜索结果缓存
        self.index_loaded = False  # 索引是否已加载标志
        self.gpu_resources = None  # GPU 资源

        # 节点嵌入缓存，用于存储已计算的节点嵌入向量
        self.node_embedding_cache = {}
        
        # 获取模型输出维度并处理维度转换
        self.model_dim = self.model.get_sentence_embedding_dimension()
        self.dim_transform = None
        # 如果模型输出维度不是 384，则创建线性变换层将其转换为 384 维
        # if self.model_dim != 1024:
        #     self.dim_transform = torch.nn.Linear(self.model_dim, 1024)
        #     if self.device.type == "cuda" and torch.cuda.is_available():
        #         self.dim_transform = self.dim_transform.to(self.device)
        #     else:
        #         self.dim_transform = self.dim_transform.to("cpu")

        # 构建节点名称到节点 ID 的映射，方便后续查找
        self.name_to_id = {}
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            # 处理节点属性中的 name 字段
            if 'properties' in node_data and isinstance(node_data['properties'], dict):
                name = node_data['properties'].get('name', '')
                if name:
                    # 处理 name 是列表或其他类型的情况，统一转换为字符串
                    if isinstance(name, list):
                        name = ", ".join(str(item) for item in name)
                    elif not isinstance(name, str):
                        name = str(name)
                    self.name_to_id[name] = node_id
            else:
                # 处理节点直接属性中的 name 字段
                name = node_data.get('name', '')
                if name:
                    # 处理 name 是列表或其他类型的情况，统一转换为字符串
                    if isinstance(name, list):
                        name = ", ".join(str(item) for item in name)
                    elif not isinstance(name, str):
                        name = str(name)
                    self.name_to_id[name] = node_id
        
        
    def _preload_faiss_indices(self):
        """
            将 FAISS 索引预加载到 GPU 内存中（如果可用）以提高检索性能
            """
        if self.index_loaded:
            return
        
        # 如果 CUDA 可用，则初始化 GPU 资源
        if torch.cuda.is_available():
            try:
                # 创建标准 GPU 资源对象，用于管理 GPU 内存和其他资源
                self.gpu_resources = faiss.StandardGpuResources()
            except Exception as e:
                self.gpu_resources = None
        
        # 如果有 GPU 资源且存在节点索引，则尝试将节点索引移动到 GPU
        if self.gpu_resources and self.node_index:
            try:
                # 将 CPU 上的节点索引移动到 GPU
                self.node_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.node_index)
                logger.info("Node index moved to GPU")
            except Exception as e:
                logger.warning(f"Warning: Failed to move node index to GPU: {e}")

        # 如果有 GPU 资源且存在关系索引，则尝试将关系索引移动到 GPU
        if self.gpu_resources and self.relation_index:
            try:
                # 将 CPU 上的关系索引移动到 GPU
                self.relation_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.relation_index)
                logger.info("Relation index moved to GPU")
            except Exception as e:
                logger.warning(f"Warning: Failed to move relation index to GPU: {e}")

        # 如果有 GPU 资源且存在三元组索引，则尝试将三元组索引移动到 GPU
        if self.gpu_resources and self.triple_index:
            try:
                # 将 CPU 上的三元组索引移动到 GPU
                self.triple_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.triple_index)
                logger.info("Triple index moved to GPU")
            except Exception as e:
                logger.warning(f"Warning: Failed to move triple index to GPU: {e}")

        # 如果有 GPU 资源且存在社区索引，则尝试将社区索引移动到 GPU
        if self.gpu_resources and self.comm_index:
            try:
                # 将 CPU 上的社区索引移动到 GPU
                self.comm_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.comm_index)
                logger.info("Community index moved to GPU")
            except Exception as e:
                logger.warning(f"Warning: Failed to move community index to GPU: {e}")

        # 标记索引已加载完成
        self.index_loaded = True
        logger.info("FAISS indices preloaded successfully")

    def _cached_faiss_search(self, index, query_embed, top_k: int, cache_key: str):
        """
            执行带缓存的 FAISS 相似性搜索

            Args:
                index: FAISS 索引对象
                query_embed: 查询嵌入向量
                top_k: 返回最相似的前 k 个结果
                cache_key: 用于缓存的键值

            Returns:
                tuple: (D, I) 其中 D 是距离，I 是索引
            """
        # 检查缓存中是否已有该查询的结果，如果有则直接返回缓存结果
        if cache_key in self.faiss_search_cache:
            return self.faiss_search_cache[cache_key]

        # 将查询嵌入转换为 numpy 数组并调整形状以适应 FAISS 要求
        query_embed_np = query_embed.cpu().detach().numpy().reshape(1, -1)
        # 使用 FAISS 索引执行相似性搜索，获取前 top_k 个最相似的结果
        D, I = index.search(query_embed_np, top_k)

        # 将搜索结果存储在结果变量中
        result = (D, I)
        # 将结果缓存起来，以便下次相同查询可以直接使用
        self.faiss_search_cache[cache_key] = result

        # 如果缓存大小超过 1000 条记录，则删除最旧的缓存条目以控制内存使用
        if len(self.faiss_search_cache) > 1000:
            # 获取迭代器中的第一个键（最旧的条目）
            oldest_key = next(iter(self.faiss_search_cache))
            # 删除最旧的缓存条目
            del self.faiss_search_cache[oldest_key]

        # 返回搜索结果
        return result

    def dual_path_retrieval(self, query_emb: str, top_k: int = 10) -> Dict:
        """
        完成双路径检索过程
        :return: {
            "triple_nodes": 通过三元组找到的实体及其邻居,
            "comm_nodes": 通过社区找到的节点,
            "scores": 节点相关性分数,
            "scored_triples": 三元组检索得到的带分数三元组
        }
        """
        
        start_time = time.time()
        # 路径1：通过三元组进行检索，获取相关的带分数三元组
        logger.info("开始检索三元组")
        scored_triples = self.retrieve_via_triples(query_emb, top_k)

        # 从带分数的三元组中提取所有涉及的节点（头实体和尾实体）
        triple_nodes = set()
        for h, r, t, score in scored_triples:
            triple_nodes.add(h)
            triple_nodes.add(t)
        
        # 过滤掉图中不存在的节点
        triple_nodes = [node for node in triple_nodes if node in self.graph.nodes]
                    
        end_time = time.time()
        logger.info(f"Time taken to get triple nodes: {end_time - start_time} seconds")
        
        start_time = time.time()
        # 路径2：通过社区进行检索，获取相关的节点
        logger.info("开始检索社区")
        comm_nodes = self.retrieve_via_communities(query_emb, top_k)
        # 过滤掉图中不存在的节点
        comm_nodes = [node for node in comm_nodes if node in self.graph.nodes]
                            
        end_time = time.time()

        # 合并来自两个路径的节点并去重
        merged_nodes = list(set(triple_nodes + comm_nodes))
        start_time = time.time()

        # 计算合并后节点与查询的相关性分数
        logger.info("开始计算节点相关性分数")
        node_scores = self._calculate_node_scores_optimized(query_emb, merged_nodes)
        end_time = time.time()
        logger.info(f"Time taken to calculate node scores: {end_time - start_time} seconds")

        # 构造并返回最终结果
        result = {
            "triple_nodes": triple_nodes,  # 三元组路径检索到的节点
            "comm_nodes": comm_nodes,  # 社区路径检索到的节点
            "scores": node_scores,  # 所有节点的相关性分数
            "scored_triples": scored_triples  # 带分数的三元组
        }
        
        return result

    def _collect_neighbor_triples(self, node: str) -> List[Tuple[str, str, str]]:
        """
        收集给定节点3跳邻居范围内涉及的所有三元组

        Args:
            node (str): 中心节点ID

        Returns:
            List[Tuple[str, str, str]]: 包含(头实体, 尾实体, 关系)的三元组列表
        """
        # 检查节点是否存在于嵌入映射中，不存在则返回空列表
        if node not in self.node_id_to_embedding:
            return []

        # 存储收集到的三元组
        neighbor_triples = []
        # 获取节点的3跳邻居集合
        neighbors = self._get_3hop_neighbors(node)

        # 遍历所有邻居节点
        for neighbor in neighbors:
            # 获取邻居节点的所有出边（从邻居指向其他节点的边）
            for _, target, edge_data in self.graph.out_edges(neighbor, data=True):
                # 只保含有关系属性且目标节点有嵌入表示的边
                if 'relation' in edge_data and target in self.node_id_to_embedding:
                    neighbor_triples.append((neighbor, target, edge_data['relation']))
            
            # 获取指向邻居节点的所有入边（从其他节点指向邻居的边）
            for source, _, edge_data in self.graph.in_edges(neighbor, data=True):
                # 只保含有关系属性且源节点有嵌入表示的边
                if 'relation' in edge_data and source in self.node_id_to_embedding:
                    neighbor_triples.append((source, neighbor, edge_data['relation']))
                    
        return neighbor_triples
    
    def _process_triple_index(self, idx: int) -> List[Tuple[str, str, str]]:
        """
    处理单个三元组索引，返回该三元组及其相关邻居三元组

    Args:
        idx (int): 三元组在索引中的位置

    Returns:
        List[Tuple[str, str, str]]: 包含原始三元组及邻居三元组的列表，
                                   格式为(头实体, 关系, 尾实体)
    """
        try:
            # 从三元组映射中获取指定索引对应的三元组(头实体, 关系, 尾实体)
            h, r, t = self.triple_map[str(idx)]
            triples = [(h, r, t)]  # Original triple
            
            # 扩展收集：添加头实体的3跳邻居三元组
            triples.extend(self._collect_neighbor_triples(h))
            # 扩展收集：添加尾实体的3跳邻居三元组
            triples.extend(self._collect_neighbor_triples(t))
            
            return triples
            
        except (KeyError, ValueError) as e:
            logger.error(f"Warning: Error processing triple index {idx}: {str(e)}")
            return []
    
    def _deduplicate_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """去除重复三元组的功能，同时保持原有的顺序"""
        # 初始化一个空列表用于存储唯一的三元组
        unique_triples = []
        # 初始化一个集合用于跟踪已经见过的三元组（利用集合的O(1)查找特性）
        seen = set()

        # 遍历输入的三元组列表
        for triple in triples:
            # 如果当前三元组尚未出现过
            if triple not in seen:
                # 将其添加到唯一三元组列表中
                unique_triples.append(triple)
                # 在已见集合中标记该三元组
                seen.add(triple)
                
        return unique_triples

    def retrieve_via_triples(self, query_embed, top_k: int = 5) -> List[Tuple[str, str, str, float]]:
        """
    Path 1: 通过三元组索引检索相关的三元组及其3跳邻居。
    返回具有高于阈值相关性得分的带分数三元组。

    参数:
        query_embed: 查询嵌入向量
        top_k: 返回最相关的三元组数量，默认为5

    返回:
        List[Tuple[str, str, str, float]]: 包含(头实体, 关系, 尾实体, 分数)的四元组列表
    """
        if not self.triple_index:
            raise ValueError("Please build triple index first!")
        
        # 确保查询嵌入在正确的设备上并应用维度变换
        logger.info("确保查询嵌入在正确的设备上并应用维度变换")
        if isinstance(query_embed, torch.Tensor):
            # 如果已经是张量，则移到指定设备
            logger.info("将查询嵌入移到指定设备")
            query_embed = query_embed.to(self.device)
        else:
            # 如果不是张量，则转换为FloatTensor并移到指定设备
            logger.info("将查询嵌入转换为FloatTensor并移到指定设备")
            query_embed = torch.FloatTensor(query_embed).to(self.device)

        # 对查询嵌入进行维度变换（如果需要的话）
        logger.info("对查询嵌入进行维度变换（如果需要的话）")
        query_embed = self.transform_vector(query_embed)
        
        # 创建缓存键并执行FAISS搜索
        logger.info(f"创建缓存键并执行FAISS搜索")
        cache_key = f"triple_search_{hash(query_embed.cpu().numpy().tobytes())}_{top_k}"
        # 使用缓存的FAISS搜索，避免重复计算相同的查询
        logger.info(f"使用缓存的FAISS搜索，避免重复计算相同的查询")
        D, I = self._cached_faiss_search(self.triple_index, query_embed, top_k, cache_key)
        
        # 从匹配的索引中收集所有三元组，包括原始三元组和它们的邻居三元组
        all_triples = []
        for idx in I[0]: # I[0]包含搜索返回的索引
            # 处理每个索引，获取该三元组及其3跳邻居三元组
            all_triples.extend(self._process_triple_index(idx))

        # 去除重复的三元组，保持原有顺序
        unique_triples = self._deduplicate_triples(all_triples)
        
        logger.info(f"Calling _calculate_triple_relevance_scores with {len(unique_triples)} unique triples")
        # 计算三元组与查询的相关性得分，过滤低相关性的三元组
        logger.info(f"计算三元组与查询的相关性得分，过滤低相关性的三元组")
        scored_triples = self._calculate_triple_relevance_scores(query_embed, unique_triples, threshold=0.1, top_k=top_k)

        logger.info(f"_calculate_triple_relevance_scores returned {len(scored_triples)} scored triples")
        # 返回最终的带分数三元组列表
        return scored_triples

    def retrieve_via_communities(self, query_embed, top_k: int = 3) -> List[str]:
        """
        Path 2: 通过社区索引检索相关节点。
        只返回具有有效缓存嵌入的节点。

        参数:
            query_embed: 查询嵌入向量
            top_k: 返回最相关的社区数量，默认为3

        返回:
            List[str]: 相关节点ID列表
        """
        # 检查是否已构建社区索引，如果没有则抛出错误
        if not self.comm_index:
            raise ValueError("Please build community index first!")

        # 确保查询嵌入在正确的设备上
        if isinstance(query_embed, torch.Tensor):
            # 如果已经是张量，则移到指定设备
            query_embed = query_embed.to(self.device)
        else:
            # 如果不是张量，则转换为FloatTensor并移到指定设备
            query_embed = torch.FloatTensor(query_embed).to(self.device)
            
        # 对查询嵌入进行维度变换（如果需要的话）
        query_embed = self.transform_vector(query_embed)
        
        # 创建缓存键
        cache_key = f"comm_search_{hash(query_embed.cpu().numpy().tobytes())}_{top_k}"
        
        # 使用缓存的FAISS搜索，避免重复计算相同的查询
        D, I = self._cached_faiss_search(self.comm_index, query_embed, top_k, cache_key)

        # 存储从社区中获取的节点
        nodes = []
        # 遍历搜索返回的索引
        for idx in I[0]:
            # 检查索引是否有效（非负数）
            if idx >= 0:
                try:
                    # 从社区映射中获取社区ID
                    community = self.comm_map[str(idx)]
                    # 获取该社区中的所有节点
                    community_nodes = self._get_community_nodes(community)
                    # 将这些节点添加到节点列表中
                    nodes.extend(community_nodes)
                except (KeyError, ValueError) as e:
                    logger.error(f"Warning: Error processing community index {idx}: {str(e)}")
                    continue
        
        # 去除重复节点并保持原有顺序
        unique_nodes = []
        seen = set()
        for node in nodes:
            if node not in seen and node in self.node_id_to_embedding:
                unique_nodes.append(node)
                seen.add(node)

        # 返回去重后的节点列表
        return unique_nodes

    def _get_3hop_neighbors(self, center: str) -> Set[str]:
        """
    使用带缓存的BFS优化3跳邻居搜索

    参数:
        center (str): 中心节点ID

    返回:
        Set[str]: 3跳范围内所有邻居节点的集合
    """
        # 检查中心节点是否同时存在于嵌入映射和图中
        if center not in self.node_id_to_embedding:
            logger.warning(f"Warning: Node {center} not found in embedding map")
            return set()
        
        if center not in self.graph.nodes:
            logger.warning(f"Warning: Node {center} not found in graph")
            return set()
        
        # 首先检查缓存，避免重复计算
        cache_key = f"3hop_{center}"
        if hasattr(self, '_3hop_cache') and cache_key in self._3hop_cache:
            return self._3hop_cache[cache_key]

        # 初始化邻居集合和已访问集合，都包含中心节点
        neighbors = {center}
        visited = {center}
        
        try:
            # 使用BFS进行更高效的遍历
            queue = [(center, 0)]  # 队列元素为(节点, 深度)
            
            while queue:
                current_node, depth = queue.pop(0)

                # 如果达到3跳深度限制，则停止继续深入
                if depth >= 3:
                    continue
                
                # 在获取邻居之前检查当前节点是否存在于图中
                if current_node not in self.graph.nodes:
                    logger.warning(f"Current node {current_node} not found in graph during BFS")
                    continue

                # 遍历当前节点的所有邻居
                for neighbor in self.graph.neighbors(current_node):
                    # 只包含同时存在于图和嵌入映射中的未访问邻居节点
                    if neighbor in self.node_id_to_embedding and neighbor not in visited:
                        visited.add(neighbor)
                        neighbors.add(neighbor)
                        # 只有当我们还能继续深入时才添加到队列
                        if depth < 2:
                            queue.append((neighbor, depth + 1))
                    elif neighbor not in self.node_id_to_embedding:
                        logger.warning(f"Warning: Neighbor {neighbor} of {current_node} not found in embedding map")
                            
        except Exception as e:
            logger.error(f"Error getting neighbors for node {center}: {str(e)}")
        
        # 缓存结果以供后续使用
        if not hasattr(self, '_3hop_cache'):
            self._3hop_cache = {}
        self._3hop_cache[cache_key] = neighbors
        
        # 限制缓存大小，防止内存占用过大
        if len(self._3hop_cache) > 10000:
            # 简单的LRU策略: 移除最老的条目
            oldest_keys = list(self._3hop_cache.keys())[:1000]
            for key in oldest_keys:
                del self._3hop_cache[key]
        
        return neighbors

    def _get_community_nodes(self, community: str) -> List[str]:
        """
            获取属于特定社区的所有节点。
            社区是标签为'community'且具有成员属性的节点。

            参数:
                community (str): 社区节点ID

            返回:
                List[str]: 属于该社区的节点ID列表
            """
        # 检查社区节点是否存在于图中
        if community not in self.graph.nodes:
            return []

        # 检查该节点是否为社区节点（标签必须是'community'）
        if self.graph.nodes[community].get('label') != 'community':
            return []
            
        # 从社区的属性中获取成员信息
        if 'properties' in self.graph.nodes[community]:
            # 获取成员名称列表
            member_names = self.graph.nodes[community]['properties'].get('members', [])
            # 将成员名称转换为节点ID
            member_ids = []
            for name in member_names:
                # 处理名称为列表或其他类型的情况，统一转换为字符串
                if isinstance(name, list):
                    name = ", ".join(str(item) for item in name)
                elif not isinstance(name, str):
                    name = str(name)

                # 通过名称查找对应的节点ID
                if name in self.name_to_id:
                    member_ids.append(self.name_to_id[name])
                else:
                    logger.warning(f"Warning: Member name '{name}' not found in graph nodes")
            return member_ids
        return []

    def _calculate_node_scores(self, query_embed, nodes: List[str]) -> Dict[str, float]:
        """
            计算查询嵌入与一组节点之间的相关性得分（余弦相似度）

            参数:
                query_embed: 查询嵌入向量
                nodes: 节点ID列表

            返回:
                Dict[str, float]: 节点ID到相似度得分的映射字典
            """
        scores = {}
        
        if not nodes:
            return scores

        # 将查询嵌入转换为numpy数组并在设备间移动
        query_embed = query_embed.cpu().detach().numpy()
        query_tensor = torch.FloatTensor(query_embed).to(self.device)
        # 如果需要，对查询向量进行维度变换
        query_tensor = self.transform_vector(query_tensor)

        # 分类节点：已有嵌入、缓存中存在、需要编码的节点
        nodes_with_embedding = []  # 图中已存在的节点嵌入
        nodes_without_embedding = []  # 需要编码的新节点
        nodes_to_encode = []  # 需要编码的节点列表

        # 遍历所有节点，分类处理
        for node in nodes:
            if 'embedding' in self.graph.nodes[node]:
                # 节点在图中有预存的嵌入
                nodes_with_embedding.append(node)
            elif node in self.node_embedding_cache:
                # 节点在缓存中有嵌入，直接计算相似度
                scores[node] = F.cosine_similarity(query_tensor, self.node_embedding_cache[node], dim=0).item()
            else:
                # 节点既不在图中也不在缓存中，需要重新编码
                nodes_without_embedding.append(node)
                nodes_to_encode.append(node)

        # 处理图中已有的节点嵌入
        if nodes_with_embedding:
            embeddings = []
            # 提取这些节点的嵌入
            for node in nodes_with_embedding:
                node_embed = torch.FloatTensor(self.graph.nodes[node]['embedding']).to(self.device)
                embeddings.append(node_embed)
            
            if embeddings:
                # 将嵌入堆叠成张量
                embeddings_tensor = torch.stack(embeddings)
                # 计算查询与所有节点嵌入的余弦相似度
                similarities = F.cosine_similarity(query_tensor.unsqueeze(0), embeddings_tensor, dim=1)

                # 将相似度得分存入结果字典
                for i, node in enumerate(nodes_with_embedding):
                    scores[node] = similarities[i].item()

        # 处理需要编码的节点
        if nodes_to_encode:
            texts = []
            # 获取需要编码节点的文本表示
            for node in nodes_to_encode:
                text = self._get_node_text(node)
                texts.append(text)
            
            if texts:
                # 使用模型对文本进行编码
                node_embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)

                # 如果需要，对嵌入进行维度变换
                if self.dim_transform is not None:
                    node_embeddings = self.dim_transform(node_embeddings)

                # 计算查询与新编码节点的余弦相似度
                similarities = F.cosine_similarity(query_tensor.unsqueeze(0), node_embeddings, dim=1)

                # 将相似度得分存入结果字典，并将新嵌入缓存起来
                for i, node in enumerate(nodes_to_encode):
                    scores[node] = similarities[i].item()
                    self.node_embedding_cache[node] = node_embeddings[i].detach()
        
        return scores

    def _calculate_node_scores_optimized(self, query_embed, nodes: List[str]) -> Dict[str, float]:
        """
            优化版的节点得分计算方法，计算查询嵌入与一组节点之间的相关性得分（余弦相似度）

            参数:
                query_embed: 查询嵌入向量
                nodes: 节点ID列表

            返回:
                Dict[str, float]: 节点ID到相似度得分的映射字典
            """
        if not nodes:
            return {}

        # 将查询嵌入转换为numpy数组并在设备间移动
        query_embed = query_embed.cpu().detach().numpy()
        query_tensor = torch.FloatTensor(query_embed).to(self.device)
        # 如果需要，对查询向量进行维度变换
        query_tensor = self.transform_vector(query_tensor)

        # 初始化存储节点嵌入和节点名称的列表
        node_embeddings = []
        node_names = []

        # 遍历所有节点，收集已有嵌入的节点
        for node in nodes:
            if 'embedding' in self.graph.nodes[node]:
                # 节点在图中有预存的嵌入
                embed = torch.FloatTensor(self.graph.nodes[node]['embedding']).to(self.device)
                node_embeddings.append(embed)
                node_names.append(node)
            elif node in self.node_embedding_cache:
                # 节点在缓存中有嵌入
                node_embeddings.append(self.node_embedding_cache[node])
                node_names.append(node)
            else:
                continue

        # 初始化得分字典
        scores = {}
        if node_embeddings:
            # 将嵌入堆叠成张量
            embeddings_tensor = torch.stack(node_embeddings)
            # 计算查询与所有节点嵌入的余弦相似度
            similarities = F.cosine_similarity(query_tensor.unsqueeze(0), embeddings_tensor, dim=1)

            # 将相似度得分存入结果字典
            for i, node in enumerate(node_names):
                scores[node] = similarities[i].item()

        # 找出还没有得分的节点（即需要编码的节点）
        nodes_to_encode = [node for node in nodes if node not in scores]
        if nodes_to_encode:
            # 获取需要编码节点的文本表示
            texts = [self._get_node_text(node) for node in nodes_to_encode]
            if texts:
                try:
                    # 使用模型对文本进行编码
                    embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
                    
                    if self.dim_transform is not None:
                        embeddings = self.dim_transform(embeddings)

                    # 计算查询与新编码节点的余弦相似度
                    similarities = F.cosine_similarity(query_tensor.unsqueeze(0), embeddings, dim=1)

                    # 将相似度得分存入结果字典，并将新嵌入缓存起来
                    for i, node in enumerate(nodes_to_encode):
                        scores[node] = similarities[i].item()
                        self.node_embedding_cache[node] = embeddings[i].detach()
                        
                except Exception as e:
                    logger.warning(f"Error encoding nodes: {e}")
                    for node in nodes_to_encode:
                        if node not in scores:
                            scores[node] = 0.0
        
        return scores

    def clear_embedding_cache(self, max_cache_size: int = 10000):
        """
            清理嵌入缓存，当缓存大小超过指定限制时，移除最旧的缓存项

            参数:
                max_cache_size (int): 缓存最大容量，默认为10000个条目
            """
        if len(self.node_embedding_cache) > max_cache_size:
            items_to_remove = len(self.node_embedding_cache) - max_cache_size
            oldest_keys = list(self.node_embedding_cache.keys())[:items_to_remove]
            for key in oldest_keys:
                del self.node_embedding_cache[key]

    def save_embedding_cache(self):
        """
            将嵌入缓存保存到磁盘，使用numpy格式避免pickle相关问题

            返回:
                bool: 保存成功返回True，否则返回False
            """
        cache_path = f"{self.cache_dir}/{self.dataset}/node_embedding_cache.pt"
        try:
            if not self.node_embedding_cache:
                return False
                
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Convert tensors to numpy arrays for safe serialization
            numpy_cache = {}
            for node, embed in self.node_embedding_cache.items():
                if embed is not None:
                    try:
                        # Convert to numpy array for safer serialization
                        if hasattr(embed, 'detach'):
                            numpy_cache[node] = embed.detach().cpu().numpy()
                        elif isinstance(embed, np.ndarray):
                            numpy_cache[node] = embed
                        else:
                            numpy_cache[node] = np.array(embed)
                    except Exception as e:
                        continue
            
            if not numpy_cache:
                return False
            
            # Save using torch.save with tensor format for better compatibility
            try:
                tensor_cache = {}
                for node, embed_array in numpy_cache.items():
                    if isinstance(embed_array, np.ndarray):
                        tensor_cache[node] = torch.from_numpy(embed_array).float()
                    else:
                        tensor_cache[node] = embed_array
                
                torch.save(tensor_cache, cache_path)
            except Exception as torch_error:
                # Fallback to numpy format to avoid pickle tensor issues
                cache_path_npz = cache_path.replace('.pt', '.npz')
                np.savez_compressed(cache_path_npz, **numpy_cache)
                cache_path = cache_path_npz
            
            file_size = os.path.getsize(cache_path)
            logger.info(f"已将包含 {len(numpy_cache)} 个条目的嵌入缓存保存到 {cache_path} (大小: {file_size} 字节)")
            return True
                
        except Exception as e:
            return False

    def load_embedding_cache(self):
        """从磁盘加载嵌入缓存"""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_embedding_cache.pt"
        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:  
                    logger.warning(f"Warning: Cache file too small ({file_size} bytes), likely empty or corrupted")
                    return False
                
                # 兼容PyTorch 2.6+的weights_only参数
                try:
                    cpu_cache = torch.load(cache_path, map_location='cpu', weights_only=False)
                except TypeError:
                    cpu_cache = torch.load(cache_path, map_location='cpu')
                except Exception as e:
                    if "numpy.core.multiarray._reconstruct" in str(e):
                        try:
                            import importlib
                            torch_serialization = importlib.import_module('torch.serialization')
                            torch_serialization.add_safe_globals(["numpy.core.multiarray._reconstruct"])
                            cpu_cache = torch.load(cache_path, map_location='cpu')
                        except:
                            raise e
                    else:
                        raise e
                
                if not cpu_cache:
                    logger.warning("Warning: Loaded cache is empty")
                    return False
                

                self.node_embedding_cache.clear()
                
                for node, embed in cpu_cache.items():
                    if embed is not None:
                        try:
                            # 安全地移动到目标设备，兼容CPU环境
                            if isinstance(embed, np.ndarray):
                                embed_tensor = torch.from_numpy(embed).float()
                            else:
                                embed_tensor = embed.cpu() if hasattr(embed, 'cpu') else embed
                            
                            # 只在CUDA可用时移动到CUDA设备
                            if self.device.type == "cuda" and torch.cuda.is_available():
                                embed_tensor = embed_tensor.to(self.device)
                            else:
                                embed_tensor = embed_tensor.to("cpu")
                            
                            self.node_embedding_cache[node] = embed_tensor
                        except Exception as e:
                            logger.warning(f"Warning: Failed to load embedding for node {node}: {e}")
                            continue

                logger.info(f"Loaded embedding cache with {len(self.node_embedding_cache)} entries from {cache_path} (file size: {file_size} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Error loading embedding cache: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted cache file: {cache_path}")
                except Exception as e2:
                    logger.warning(f"Failed to remove corrupted cache file {cache_path}: {type(e2).__name__}: {e2}")
        else:
            logger.info(f"Cache file not found: {cache_path}")
        return False

    def _is_valid_node_text(self, text: str) -> bool:
        """
    检查节点文本是否有效用于嵌入计算

    参数:
        text (str): 要检查的节点文本

    返回:
        bool: 如果文本有效返回True，否则返回False
    """
        return text and not text.startswith('[Error') and not text.startswith('[Unknown')
    
    def _prepare_batch_data(self, batch_nodes: list) -> tuple[list, list]:
        """
        从一批节点中准备批量文本和有效节点

        参数:
            batch_nodes: 节点ID列表

        返回:
            tuple: (有效的节点文本列表, 有效的节点ID列表)
        """
        batch_texts = []
        valid_nodes = []
        
        for node in batch_nodes:
            try:
                text = self._get_node_text(node)
                if self._is_valid_node_text(text):
                    batch_texts.append(text)
                    valid_nodes.append(node)
                else:
                    logger.warning(f"Warning: Invalid text for node {node}: {text}")
            except Exception as e:
                logger.error(f"Error getting text for node {node}: {e}")
                continue
                
        return batch_texts, valid_nodes
    
    def _compute_and_transform_embeddings(self, texts: list) -> torch.Tensor:
        """
        计算文本嵌入并向量维度变换（如果需要）

        参数:
            texts (list): 需要编码的文本列表

        返回:
            torch.Tensor: 编码后的嵌入向量张量
        """
        # 使用预训练模型对文本进行编码，转换为张量形式并在指定设备上运行
        embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)

        # 如果存在维度变换器，则对嵌入向量进行维度变换
        if hasattr(self, 'dim_transform') and self.dim_transform is not None:
            embeddings = self.dim_transform(embeddings)
            
        return embeddings
    
    def _process_single_node_fallback(self, node: str) -> bool:
        """
        当批量处理失败时，作为后备方案处理单个节点

        参数:
            node (str): 需要处理的节点ID

        返回:
            bool: 处理成功返回True，否则返回False
        """
        try:
            text = self._get_node_text(node)
            if not self._is_valid_node_text(text):
                return False

            # 对单个文本进行编码，注意这里使用列表包装[text]以符合encode方法要求
            embedding = self.model.encode([text], convert_to_tensor=True, device=self.device)[0]
            
            if hasattr(self, 'dim_transform') and self.dim_transform is not None:
                embedding = self.dim_transform(embedding.unsqueeze(0)).squeeze(0)
                
            self.node_embedding_cache[node] = embedding.detach()
            return True
            
        except Exception as e:
            logger.error(f"Error encoding individual node {node}: {e}")
            return False
    
    def _process_batch(self, batch_nodes: list, batch_num: int, total_batches: int) -> int:
        """
    处理单个节点批次并返回成功处理的节点数

    参数:
        batch_nodes: 需要处理的节点ID列表
        batch_num: 当前批次编号
        total_batches: 总批次数

    返回:
        int: 成功处理的节点数量
    """
        # 准备批次数据，获取有效的节点文本和对应的节点ID
        batch_texts, valid_nodes = self._prepare_batch_data(batch_nodes)
        
        if not batch_texts:
            logger.info(f"Warning: No valid texts in batch {batch_num}")
            return 0
        
        try:
            # 首先尝试批量处理
            embeddings = self._compute_and_transform_embeddings(batch_texts)

            # 将生成的嵌入向量存入节点嵌入缓存中
            for j, node in enumerate(valid_nodes):
                self.node_embedding_cache[node] = embeddings[j].detach()
            
            logger.info(f"Encoded batch {batch_num}/{total_batches} ({len(valid_nodes)} nodes)")
            return len(valid_nodes)
            
        except Exception as e:
            logger.error(f"Error encoding batch {batch_num}: {e}")
            logger.info("Falling back to individual node processing...")
            
            # 回退到单个节点处理
            success_count = 0
            for node in valid_nodes:
                if self._process_single_node_fallback(node):
                    success_count += 1
                    
            return success_count

    def _precompute_node_embeddings(self, batch_size: int = 100, force_recompute: bool = False):
        """
        使用优化的批处理预计算所有图节点的嵌入

        参数:
            batch_size: 批处理大小，默认为100
            force_recompute: 是否强制重新计算，即使缓存存在也重新计算
        """
        # 如果不是强制重新计算，尝试从磁盘缓存加载节点嵌入
        if not force_recompute:
            logger.info("正在尝试从磁盘缓存加载节点嵌入...")
            if self.load_embedding_cache():
                logger.info("成功从磁盘缓存加载节点嵌入")
                return

        logger.info("正在预计算节点嵌入...")

        # 清空当前的节点嵌入缓存
        self.node_embedding_cache.clear()
        
        # Prepare batch processing
        all_nodes = list(self.graph.nodes())
        total_nodes = len(all_nodes)
        total_batches = (total_nodes + batch_size - 1) // batch_size

        logger.info(f"总共需要处理的节点数: {total_nodes}")
        logger.info(f"分 {total_batches} 批处理，每批大小为 {batch_size}")

        # 分批处理节点
        total_processed = 0
        for i in range(0, total_nodes, batch_size):
            batch_nodes = all_nodes[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            processed_count = self._process_batch(batch_nodes, batch_num, total_batches)
            total_processed += processed_count

        logger.info(f"成功为 {len(self.node_embedding_cache)} 个节点预计算嵌入")
        logger.info(f"处理成功率: {len(self.node_embedding_cache)}/{total_nodes} ({len(self.node_embedding_cache) / total_nodes * 100:.1f}%)")

        # 如果有节点嵌入被计算出来，则保存到磁盘缓存
        if self.node_embedding_cache:
            self.save_embedding_cache()

    def build_indices(self):
        """
            构建 FAISS 索引，仅在索引不存在或与当前图不一致时进行构建
            """
        # 定义各个索引和嵌入文件的路径
        node_path = f"{self.cache_dir}/{self.dataset}/node.index"
        relation_path = f"{self.cache_dir}/{self.dataset}/relation.index"
        triple_path = f"{self.cache_dir}/{self.dataset}/triple.index"
        comm_path = f"{self.cache_dir}/{self.dataset}/comm.index"
        node_embed_path = f"{self.cache_dir}/{self.dataset}/node_embeddings.pt"
        relation_embed_path = f"{self.cache_dir}/{self.dataset}/relation_embeddings.pt"
        node_map_path = f"{self.cache_dir}/{self.dataset}/node_map.json"

        # 检查所有索引和嵌入文件是否都已存在
        all_exist = (os.path.exists(node_path) and 
                    os.path.exists(relation_path) and 
                    os.path.exists(triple_path) and 
                    os.path.exists(comm_path) and
                    os.path.exists(node_embed_path) and
                    os.path.exists(relation_embed_path) and
                    os.path.exists(node_map_path))
        
        indices_consistent = False
        # 如果所有文件都存在，检查索引与当前图的一致性
        if all_exist:
            try:
                # 加载节点映射文件
                with open(node_map_path, 'r') as f:
                    cached_node_map = json.load(f)
                # 获取当前图中的节点和缓存中的节点
                current_nodes = set(self.graph.nodes())
                cached_nodes = set(cached_node_map.values())

                # 比较当前节点和缓存节点是否一致
                if current_nodes == cached_nodes:
                    indices_consistent = True
                    logger.info("Cached FAISS indices are consistent with current graph")
                else:
                    logger.info(f"Graph inconsistency detected: current nodes {len(current_nodes)}, cached nodes {len(cached_nodes)}")
                    logger.info(f"Missing in cache: {current_nodes - cached_nodes}")
                    logger.info(f"Extra in cache: {cached_nodes - current_nodes}")
            except Exception as e:
                logger.error(f"Error checking index consistency: {e}")

        # 如果所有文件都存在且一致，则直接加载缓存
        if all_exist and indices_consistent:
            logger.info("All FAISS indices and embeddings already exist, loading from cache...")
            # 如果节点索引未加载，则加载所有索引
            if not hasattr(self, 'node_index') or self.node_index is None:
                self._load_indices()
            
            logger.info("Attempting to load node embedding cache from disk...")
            # 尝试加载节点嵌入缓存
            if not self.load_embedding_cache():
                logger.info("Disk cache not available, rebuilding node embedding cache...")
                self._precompute_node_embeddings(force_recompute=True)
            else:
                logger.info("Successfully loaded node embedding cache from disk")
        else:
            # 如果文件不存在或不一致，则重新构建索引和嵌入
            logger.info("正在构建FAISS索引和嵌入...")
            # 如果文件存在但不一致，清除不一致的缓存文件
            if all_exist and not indices_consistent:
                logger.info("正在清除不一致的缓存文件...")
                for path in [node_path, relation_path, triple_path, comm_path, node_embed_path, relation_embed_path, node_map_path]:
                    if os.path.exists(path):
                        os.remove(path)

            # 构建各种索引
            self._build_node_index()
            self._build_relation_index()
            self._build_triple_index()
            self._build_community_index()
            logger.info("FAISS索引和嵌入构建成功!")
            # 填充嵌入映射
            self._populate_embedding_maps()

            # 强制重新计算节点嵌入
            self._precompute_node_embeddings(force_recompute=True)

        # 预加载 FAISS 索引到 GPU（如果可用
        self._preload_faiss_indices()

    def _build_node_index(self):
        """
        为所有节点构建 FAISS 索引并缓存嵌入向量
        """
        # 获取图中所有节点列表
        nodes = list(self.graph.nodes())
        # 为每个节点生成文本表示
        texts = [self._get_node_text(n) for n in nodes]
        # 使用模型对节点文本进行编码生成嵌入向量
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # 将嵌入向量存储在CPU上以节省GPU内存
        self.node_embeddings = embeddings.cpu()
        # 保存为numpy格式以避免pickle相关问题
        embeddings_numpy = self.node_embeddings.numpy()
        np.save(f"{self.cache_dir}/{self.dataset}/node_embeddings.npy", embeddings_numpy)
        
        # 构建 FAISS 索引
        embeddings_np = embeddings.cpu().numpy()
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)

        # 创建节点ID到索引位置的映射
        faiss.write_index(index, f"{self.cache_dir}/{self.dataset}/node.index")
        # 创建节点ID到索引位置的映射
        self.node_map = {str(i): n for i, n in enumerate(nodes)}
        # 保存节点映射到JSON文件
        with open(f"{self.cache_dir}/{self.dataset}/node_map.json", 'w') as f:
            json.dump(self.node_map, f)

        # 保存索引到实例变量
        self.node_index = index
        
    def _build_relation_index(self):
        """Build FAISS index for all relations and cache embeddings"""
        relations = sorted(list({
            data['relation'] for _, _, data in self.graph.edges(data=True) if 'relation' in data
        }))
                
        embeddings = self.model.encode(relations, convert_to_tensor=True)

        # Store embeddings on CPU
        self.relation_embeddings = embeddings.cpu()
        # Save as numpy to avoid pickle issues
        embeddings_numpy = self.relation_embeddings.numpy()
        np.save(f"{self.cache_dir}/{self.dataset}/relation_embeddings.npy", embeddings_numpy)

        # Build FAISS index
        embeddings_np = embeddings.cpu().numpy()
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        faiss.write_index(index, f"{self.cache_dir}/{self.dataset}/relation.index")
        self.relation_map = {str(i): r for i, r in enumerate(relations)}
        with open(f"{self.cache_dir}/{self.dataset}/relation_map.json", 'w') as f:
            json.dump(self.relation_map, f)
            
        self.relation_index = index

    def _build_triple_index(self):
        """Build FAISS Triple Index"""
        triples = []
        for u, v, data in self.graph.edges(data=True):
            if 'relation' in data:
                triples.append((u, data['relation'],v))
        
        texts = [f"{self._get_node_text(h)},{r},{self._get_node_text(t)}" for h, r, t in triples]
        embeddings = self.model.encode(texts)
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        faiss.write_index(index, f"{self.cache_dir}/{self.dataset}/triple.index")
        with open(f"{self.cache_dir}/{self.dataset}/triple_map.json", 'w') as f:
            json.dump({i: n for i, n in enumerate(triples)}, f)
        
        self.triple_index = index
        self.triple_map = {str(i): n for i, n in enumerate(triples)}

    def _build_community_index(self):
        """Build FAISS Community Index"""
        communities = {
            n for n, d in self.graph.nodes(data=True) 
            if d.get('label') == 'community'
        }
        
        texts = []
        valid_communities = []
        for comm in communities:
            # Get community text representation
            data = self.graph.nodes[comm]
            if 'properties' in data:
                name = data['properties'].get('name', '')
                description = data['properties'].get('description', '')
                if name or description:  # Only include if it has name or description
                    texts.append(f"{name},{description}".strip())
                    valid_communities.append(comm)
        
        if not valid_communities:
            return
            
        embeddings = self.model.encode(texts)
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        faiss.write_index(index, f"{self.cache_dir}/{self.dataset}/comm.index")
        with open(f"{self.cache_dir}/{self.dataset}/comm_map.json", 'w') as f:
            json.dump({i: n for i, n in enumerate(valid_communities)}, f)
        
        self.comm_index = index
        self.comm_map = {str(i): n for i, n in enumerate(valid_communities)}

    def _load_indices(self):
        logger.info("Starting _load_indices...")
        triple_path = f"{self.cache_dir}/{self.dataset}/triple.index"
        comm_path = f"{self.cache_dir}/{self.dataset}/comm.index"
        node_path = f"{self.cache_dir}/{self.dataset}/node.index"
        relation_path = f"{self.cache_dir}/{self.dataset}/relation.index"
        node_embed_path = f"{self.cache_dir}/{self.dataset}/node_embeddings.pt"
        relation_embed_path = f"{self.cache_dir}/{self.dataset}/relation_embeddings.pt"
        
        logger.debug(f"Checking cache files...")
        logger.debug(f"node_path exists: {os.path.exists(node_path)}")
        logger.debug(f"relation_path exists: {os.path.exists(relation_path)}")
        logger.debug(f"triple_path exists: {os.path.exists(triple_path)}")
        logger.debug(f"comm_path exists: {os.path.exists(comm_path)}")
        logger.debug(f"node_embed_path exists: {os.path.exists(node_embed_path)}")
        logger.debug(f"relation_embed_path exists: {os.path.exists(relation_embed_path)}")
        
        if os.path.exists(node_path):
            logger.debug("Loading node index...")
            self.node_index = faiss.read_index(node_path)
            with open(f"{self.cache_dir}/{self.dataset}/node_map.json", 'r') as f:
                self.node_map = json.load(f)
                
        if os.path.exists(relation_path):
            self.relation_index = faiss.read_index(relation_path)
            with open(f"{self.cache_dir}/{self.dataset}/relation_map.json", 'r') as f:
                self.relation_map = json.load(f)
        
        if os.path.exists(triple_path):
            self.triple_index = faiss.read_index(triple_path)
            with open(f"{self.cache_dir}/{self.dataset}/triple_map.json", 'r') as f:
                self.triple_map = json.load(f)
                
        if os.path.exists(comm_path):
            self.comm_index = faiss.read_index(comm_path)
            with open(f"{self.cache_dir}/{self.dataset}/comm_map.json", 'r') as f:
                self.comm_map = json.load(f)

        if os.path.exists(node_embed_path):
            try:
                # 兼容PyTorch 2.6+的weights_only参数
                try:
                    self.node_embeddings = torch.load(node_embed_path, weights_only=False)
                except TypeError:
                    self.node_embeddings = torch.load(node_embed_path)
            except Exception as e:
                logger.warning(f"Warning: Failed to load node embeddings: {e}")
                
        if os.path.exists(relation_embed_path):
            try:
                # 兼容PyTorch 2.6+的weights_only参数
                try:
                    self.relation_embeddings = torch.load(relation_embed_path, weights_only=False)
                except TypeError:
                    self.relation_embeddings = torch.load(relation_embed_path)
            except Exception as e:
                logger.warning(f"Warning: Failed to load relation embeddings: {e}")

        # Populate maps if all necessary data is loaded
        if self.node_map and self.node_embeddings is not None:
            self._populate_embedding_maps()
        else:
            logger.debug("Cannot populate embedding maps - missing node_map or node_embeddings")
            logger.debug(f"node_map exists: {self.node_map is not None}")
            logger.debug(f"node_embeddings exists: {self.node_embeddings is not None}")

    def _populate_embedding_maps(self):
        """
        填充节点ID和关系到嵌入向量的映射
        """
        # 如果节点映射和节点嵌入都存在，则填充节点ID到嵌入的映射
        if self.node_map and self.node_embeddings is not None:
            for i_str, node_id in self.node_map.items():
                # 将字符串索引转换为整数，从node_embeddings中获取对应嵌入向量
                self.node_id_to_embedding[node_id] = self.node_embeddings[int(i_str)]

        # 如果关系映射和关系嵌入都存在，则填充关系到嵌入的映射
        if self.relation_map and self.relation_embeddings is not None:
            for i_str, rel in self.relation_map.items():
                # 将字符串索引转换为整数，从relation_embeddings中获取对应嵌入向量
                self.relation_to_embedding[rel] = self.relation_embeddings[int(i_str)]
        
        # 验证数据一致性
        self._verify_data_consistency()

    def _verify_data_consistency(self):
        """
    验证图节点和嵌入映射之间的一致性
    """
        logger.debug("Verifying data consistency...")

        # 获取图中所有节点的集合
        graph_nodes = set(self.graph.nodes())
        # 获取嵌入映射中所有节点ID的集合
        embedding_nodes = set(self.node_id_to_embedding.keys())

        # 找出在图中存在但在嵌入映射中缺失的节点
        missing_in_embeddings = graph_nodes - embedding_nodes
        # 找出在嵌入映射中存在但在图中不存在的节点
        extra_in_embeddings = embedding_nodes - graph_nodes

        if missing_in_embeddings:
            logger.warning(f"警告: 图中有 {len(missing_in_embeddings)} 个节点在嵌入中缺失: {list(missing_in_embeddings)[:5]}...")

        if extra_in_embeddings:
            logger.warning(f"警告: 嵌入中有 {len(extra_in_embeddings)} 个节点在图中不存在: {list(extra_in_embeddings)[:5]}...")

        if not missing_in_embeddings and not extra_in_embeddings:
            logger.info("✓ 数据一致性验证通过: 所有图节点都有嵌入")
        else:
            logger.info(f"✗ 检测到数据不一致: 缺失 {len(missing_in_embeddings)} 个, 多余 {len(extra_in_embeddings)} 个")

    def _get_node_text(self, node: str) -> str:
        """
            从节点数据中提取文本表示，用于生成节点嵌入

            参数:
                node (str): 节点ID

            返回:
                str: 节点的文本表示，格式为"name,description"
            """
        # 获取指定节点的数据
        data = self.graph.nodes[node]
        # 首先尝试从properties属性中获取name和description
        if 'properties' in data and isinstance(data['properties'], dict):
            name = data['properties'].get('name') or 'none'
            description = data['properties'].get('description') or 'none'
            name = str(name).strip()
            description = str(description).strip()
        else:
            name = data.get('name') or 'none'
            description = data.get('description') or 'none'
            name = str(name).strip()
            description = str(description).strip()
        
        if isinstance(name, list):
            name = ", ".join(str(item) for item in name)
        elif not isinstance(name, str):
            name = str(name)
            
        if isinstance(description, list):
            description = ", ".join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        return f"{name},{description}".strip()

    def _subgraph_to_text(self, subgraph: nx.MultiDiGraph) -> str:
        """
            将子图转换为可读的文本格式

            参数:
                subgraph (nx.MultiDiGraph): 需要转换的NetworkX子图

            返回:
                str: 格式化的文本表示
            """
        # 存储文本部分的列表
        text_parts = []
        
        # 添加节点信息
        for node, data in subgraph.nodes(data=True):
            # 获取节点名称，如果不存在则使用节点ID
            node_text = f"Node: {data.get('name', node)}\n"
            # 如果节点有描述信息，则添加描述
            if 'description' in data:
                node_text += f"Description: {data['description']}\n"
            # 如果节点有属性信息，则添加属性
            if 'properties' in data:
                node_text += f"Properties: {data['properties']}\n"
            # 将节点文本添加到文本部分列表中
            text_parts.append(node_text)
        
        # 添加边信息
        for u, v, data in subgraph.edges(data=True):
            edge_text = f"Relation: {data.get('relation', '')} between {subgraph.nodes[u].get('name', u)} and {subgraph.nodes[v].get('name', v)}\n"
            text_parts.append(edge_text)

        # 将所有文本部分用换行符连接并返回
        return "\n".join(text_parts)

    def _extract_node_info(self, node_data: dict) -> tuple[str, str]:
        """
        从节点数据中提取并标准化名称和描述信息

        参数:
            node_data (dict): 包含节点信息的字典

        返回:
            tuple[str, str]: 包含标准化后的名称和描述的元组
        """
        def normalize_field(field) -> str:
            """
            将各种类型的字段转换为干净的字符串

            参数:
                field: 需要标准化的字段值

            返回:
                str: 标准化后的字符串
            """
            if not field:
                return ''
            if isinstance(field, list):
                return ", ".join(str(item) for item in field)
            return str(field).strip()

        # 优先尝试从properties属性中获取name和description
        if 'properties' in node_data and isinstance(node_data['properties'], dict):
            name = normalize_field(node_data['properties'].get('name'))
            description = normalize_field(node_data['properties'].get('description'))
        else:
            name = normalize_field(node_data.get('name'))
            description = normalize_field(node_data.get('description'))
        
        return name, description
    
    def _format_node_text(self, name: str, description: str) -> str:
        """
        将节点名称和描述格式化为显示文本

        参数:
            name (str): 节点名称
            description (str): 节点描述

        返回:
            str: 格式化后的显示文本
        """
        if not name:
            return ''
        return f"{name} - {description}" if description else name
    
    def _get_community_members(self, community_node: str) -> tuple[list[str], list[str]]:
        """
        获取属于某个社区的实体和关键词节点

        参数:
            community_node (str): 社区节点ID

        返回:
            tuple[list[str], list[str]]: 包含实体列表和关键词列表的元组
        """
        # 初始化实体和关键词列表
        entities, keywords = [], []

        # 遍历图中所有节点，查找属于指定社区的节点
        for node_id, node_data in self.graph.nodes(data=True):
            # 检查节点是否属于指定的社区（通过community_l4属性判断）
            if node_data.get('community_l4') == community_node:
                # 提取节点的名称和描述信息
                name, description = self._extract_node_info(node_data)
                if not name:
                    continue

                # 格式化节点文本显示
                formatted_text = self._format_node_text(name, description)
                # 获取节点类型（level属性）
                node_type = node_data.get('level')

                # 根据节点类型分类添加到相应列表
                if node_type == 2:  # Entity
                    entities.append(formatted_text)
                elif node_type == 1:  # Keyword
                    keywords.append(formatted_text)
        
        return entities, keywords
    
    def _format_community_content(self, base_text: str, entities: list[str], keywords: list[str]) -> str:
        """
    格式化社区内容，包括其成员实体和关键词

    参数:
        base_text (str): 社区的基本信息文本
        entities (list[str]): 社区包含的实体列表
        keywords (list[str]): 社区包含的关键词列表

    返回:
        str: 格式化后的社区内容文本
    """
        if not entities and not keywords:
            return base_text

        # 初始化内容部分列表，包含基本文本和"Contains:"标题
        content_parts = [base_text, "\n  Contains:"]

        # 如果有实体，则添加实体信息
        if entities:
            # 只显示前3个实体
            shown = entities[:3]
            entities_text = f"\n    Entities: {', '.join(shown)}"
            # 如果实体数量超过3个，添加"and X more"提示
            if len(entities) > 3:
                entities_text += f" and {len(entities) - 3} more"
            content_parts.append(entities_text)

        # 如果有关键词，则添加关键词信息
        if keywords:
            # 只显示前3个关键词
            shown = keywords[:3]
            keywords_text = f"\n    Keywords: {', '.join(shown)}"
            # 如果关键词数量超过3个，添加"and X more"提示
            if len(keywords) > 3:
                keywords_text += f" and {len(keywords) - 3} more"
            content_parts.append(keywords_text)

        # 将所有内容部分连接成一个字符串并返回
        return "".join(content_parts)
    
    def _nodes_to_text(self, nodes: List[str]) -> str:
        """
        将节点列表转换为带有节点信息的可读文本格式

        参数:
            nodes (List[str]): 节点ID列表

        返回:
            str: 格式化的文本表示
        """
        # 节点类型映射，使代码更清晰
        NODE_TYPES = {1: 'keywords', 2: 'entities', 4: 'communities'}
        
        # 按类型收集节点
        collected = {node_type: [] for node_type in NODE_TYPES.values()}
        
        for node in nodes:
            if node not in self.graph.nodes:
                continue

            # 获取节点数据和类型
            node_data = self.graph.nodes[node]
            node_type = node_data.get('level')
            # 提取节点名称和描述
            name, description = self._extract_node_info(node_data)
            
            if not name:  # Skip nodes without meaningful names
                continue
                
            if node_type == 2:  # Entity
                formatted = self._format_node_text(name, description)
                collected['entities'].append(formatted)
                
            elif node_type == 1:  # Keyword
                formatted = self._format_node_text(name, description)
                collected['keywords'].append(formatted)
                
            elif node_type == 4:  # Community
                base_text = self._format_node_text(name, description)
                entities, keywords = self._get_community_members(node)
                community_text = self._format_community_content(base_text, entities, keywords)
                collected['communities'].append(community_text)
        
        # 构建输出部分
        text_parts = ["=== Retrieved Information ==="]

        # 定义各部分的配置
        section_configs = [
            ('entities', '=== Entity Information ==='),
            ('keywords', '=== Keyword Information ==='),
            ('communities', '=== Community Information ===')
        ]

        # 为每个部分添加内容
        for section_key, section_header in section_configs:
            items = collected[section_key]
            if items:
                text_parts.extend([f"\n{section_header}"] + [f"• {item}" for item in items])
        
        # 返回结果或回退消息
        if len(text_parts) == 1:  # Only header present
            return "No relevant information found."
        
        return "\n".join(text_parts)

    def transform_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """
        在需要时对向量进行维度变换

        参数:
            vector (torch.Tensor): 输入的向量张量

        返回:
            torch.Tensor: 经过维度变换后的向量张量，如果不需要变换则返回原向量
        """
        if self.dim_transform is not None:
            return self.dim_transform(vector)
        return vector

    def _calculate_triple_relevance_scores(self, query_embed: torch.Tensor, triples: List[Tuple[str, str, str]], threshold: float = 0.3, top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """
    使用FAISS计算三元组的相关性得分，并过滤掉低相关性的三元组

    参数:
        query_embed: 查询嵌入张量
        triples: 三元组列表，格式为(头实体, 尾实体, 关系)
        threshold: 最小相关性得分阈值
        top_k: 返回的最大三元组数量

    返回:
        包含(头实体, 尾实体, 关系, 得分)的四元组列表，得分高于阈值，数量不超过top_k
    """
        
        scored_triples = []
        
        if not triples:
            logger.debug("No triples to process")
            return []

        # 对查询嵌入进行维度变换（如果需要）
        query_embed = self.transform_vector(query_embed)
        # 将查询嵌入转换为numpy数组并调整形状以适应FAISS要求
        query_embed_np = query_embed.cpu().detach().numpy().reshape(1, -1)
        
        # 对查询嵌入进行L2归一化，以便在FAISS中进行余弦相似度计算
        faiss.normalize_L2(query_embed_np)
        
        # Create a set of input triples for fast lookup
        input_triples_set = set(triples)
        logger.debug(f"Input triples set size: {len(input_triples_set)}")
        logger.debug(f"First few input triples: {list(input_triples_set)[:3]}")

        # 检查三元组索引是否存在且有效
        if not hasattr(self, 'triple_index') or self.triple_index is None:
            logger.debug("triple_index is None or doesn't exist")
            # 回退方案：为所有三元组分配默认得分
            for h, r, t in triples:
                scored_triples.append((h, r, t, 0.5))  # Default score
            logger.debug(f"Using fallback method, returning {len(scored_triples)} triples")
            return scored_triples[:top_k]
        logger.debug(f"triple_index exists, size: {self.triple_index.ntotal}")
        # 使用FAISS在索引中搜索相似的三元组
        try:
            # 搜索比需要的数量更多的相似三元组，以获得更好的匹配
            search_k = min(len(triples) * 2, 50)  # Search more than needed to get good matches
            logger.debug(f"Searching for {search_k} similar triples")
            # 执行FAISS搜索
            D, I = self.triple_index.search(query_embed_np, search_k)

            # 处理FAISS搜索结果
            for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                if idx >= 0:  # Valid index
                    try:
                        # 从索引中获取三元组
                        indexed_triple = self.triple_map[str(idx)]
                        h, r, t = indexed_triple  # This is (head, tail, relation) format

                        # 检查这个三元组是否在我们的输入三元组中
                        if (h, r, t) in input_triples_set:
                            # 将距离转换为相似度得分（FAISS返回距离，我们需要相似度）
                            # 对于归一化向量，相似度 = 1 - distance^2 / 2
                            similarity_score = 1.0 - (distance ** 2) / 2.0

                            # 只保留得分高于阈值的三元组
                            if similarity_score >= threshold:
                                scored_triples.append((h, r, t, similarity_score))  # Return as (head, tail, relation, score)
                            else:
                                logger.debug(f"Triple ({h}, {t}, {r}) below threshold {threshold}")
                                
                    except (KeyError, ValueError) as e:
                        logger.error(f"Warning: Error processing indexed triple {idx}: {str(e)}")
                        continue
        except Exception as e:
            for h, r, t in triples:
                scored_triples.append((h, r, t, 0.5))  # Default score
        
        logger.debug(f"Found {len(scored_triples)} triples above threshold")
        
        # 按得分降序排序
        scored_triples.sort(key=lambda x: x[3], reverse=True)
        
        # 只返回前top_k个三元组
        result = scored_triples[:top_k]
        return result

    def __del__(self):
        try:
            if hasattr(self, 'node_embedding_cache') and self.node_embedding_cache:
                self.save_embedding_cache()
        except Exception as e:
            logger.warning(f"Error during __del__ saving embedding cache: {type(e).__name__}: {e}")

