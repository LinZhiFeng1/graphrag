import os
import pickle
import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import spacy
import torch
import torch.nn.functional as F
import concurrent.futures
from sentence_transformers import SentenceTransformer

from models.retriever.faiss_filter import DualFAISSRetriever
from utils import graph_processor
from utils import call_llm_api
from utils.logger import logger

import networkx as nx
from sklearn.preprocessing import MinMaxScaler

try:
    from config import get_config
except ImportError:
    get_config = None


class KTRetriever:
    """
    增强型的知识三元组检索器，支持多种检索路径和优化机制。
    主要功能：
        双路径检索（节点/关系检索和三元组检索）
        缓存机制（节点嵌入、查询嵌入等）
        FAISS 向量索引加速检索
        支持基于类型的过滤检索
        支持多种数据集配置
    """

    def __init__(
            self,
            dataset: str,
            json_path: str = None,
            qa_encoder: Optional[SentenceTransformer] = None,
            device: str = "",
            cache_dir: str = "retriever/faiss_cache_new",
            top_k: int = 5,
            recall_paths: int = 2,
            schema_path: str = None,
            mode: str = "agent",
            config=None
    ):
        # 尝试获取全局配置
        if config is None and get_config is not None:
            try:
                config = get_config()
            except:
                config = None

        self.config = config

        # 如果有配置，使用配置中的默认值覆盖传入参数
        if config:
            json_path = json_path or config.get_dataset_config(dataset).graph_output
            device = config.embeddings.device
            cache_dir = cache_dir if cache_dir != "retriever/faiss_cache_new" else config.retrieval.cache_dir
            top_k = top_k if top_k != 5 else config.retrieval.top_k
            recall_paths = recall_paths if recall_paths != 2 else config.retrieval.recall_paths
            schema_path = schema_path or config.get_dataset_config(dataset).schema_path
            mode = mode if mode != "agent" else config.triggers.mode
            qa_encoder = qa_encoder or SentenceTransformer(config.embeddings.model_name, device=device)

        # 加载图谱数据和编码器
        logger.info(f"加载图谱数据")
        self.graph = graph_processor.load_graph_from_json(json_path)
        # self.qa_encoder = qa_encoder or SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"加载编码器,device:{device}")
        self.qa_encoder = qa_encoder or SentenceTransformer('BAAI/bge-m3', device=device)

        # 初始化LLM客户端用于生成答案
        self.llm_client = call_llm_api.LLMCompletionCall()

        # 设备设置（GPU/CPU）
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Warning: CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device
        logger.info(f"KTRetriever Using device: {self.device}")

        # 设置基本参数
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.dataset = dataset
        self.schema_path = schema_path
        self.recall_paths = recall_paths
        self.mode = mode
        os.makedirs(cache_dir, exist_ok=True)
        self.debug_mode = True

        # 加载spaCy英语模型用于自然语言处理
        # self.nlp = spacy.load("en_core_web_lg")
        # 加载spaCy中文模型用于自然语言处理
        self.nlp = spacy.load("zh_core_web_lg")
        logger.info(f"加载spaCy模型")

        # 初始化FAISS检索器用于向量相似度搜索
        self.faiss_retriever = DualFAISSRetriever(dataset, self.graph, cache_dir=cache_dir, device=self.device)

        # 各种缓存字典，用于提高检索性能
        self.node_embedding_cache = {}  # 节点嵌入缓存
        self.triple_embedding_cache = {}  # 三元组嵌入缓存
        self.query_embedding_cache = {}  # 查询嵌入缓存
        self.faiss_search_cache = {}  # FAISS搜索结果缓存
        self.chunk_embedding_cache = {}  # 文本块嵌入缓存

        # 文本块检索相关组件
        self.chunk_faiss_index = None  # 文本块FAISS索引
        self.chunk_id_to_index = {}  # 文本块ID到索引的映射
        self.index_to_chunk_id = {}  # 索引到文本块ID的映射
        self.chunk_embeddings_precomputed = False  # 文本块嵌入是否已预计算

        # 缓存锁，确保多线程安全
        self.cache_locks = {
            'node_embedding': threading.RLock(),
            'triple_embedding': threading.RLock(),
            'query_embedding': threading.RLock(),
            'chunk_embedding': threading.RLock()
        }

        # 节点嵌入预计算状态和锁
        self.node_embeddings_precomputed = False
        self.precompute_lock = threading.Lock()

        # 加载与图谱关联的原始文本块，用于提供上下文信息
        self.chunk2id = {}
        chunk_file = f"output/chunks/{self.dataset}.txt"
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "\t" in line:
                            parts = line.split("\t", 1)
                            if len(parts) == 2 and parts[0].startswith("id: ") and parts[1].startswith("Chunk: "):
                                chunk_id = parts[0][4:]
                                chunk_text = parts[1][7:]
                                self.chunk2id[chunk_id] = chunk_text
                logger.info(f"从 {chunk_file} 加载了 {len(self.chunk2id)} 个文本块")
            except Exception as e:
                logger.error(f"Error loading chunks from {chunk_file}: {e}")
                self.chunk2id = {}

        # 初始化性能优化相关组件
        self._node_text_index = None  # 节点文本索引
        self.use_exact_keyword_matching = True  # 使用精确关键词匹配
        self.enable_performance_optimizations = True  # 启用性能优化
        self._node_text_cache = {}  # 节点文本缓存

        # 如果启用性能优化，执行预计算和缓存加载
        if self.enable_performance_optimizations:
            try:
                # 尝试加载节点嵌入缓存
                cache_loaded = self._load_node_embedding_cache()
                # 构建FAISS索引
                self.faiss_retriever.build_indices()
                # 预计算节点文本
                self._precompute_node_texts()
                # 构建节点文本索引
                self._build_node_text_index()
                # 预计算文本块嵌入
                self._precompute_chunk_embeddings()

                # 如果没有加载到缓存，则预计算节点嵌入
                if not cache_loaded:
                    self._precompute_node_embeddings()
                else:
                    self.node_embeddings_precomputed = True

                    # 确保FAISS检索器也有节点嵌入缓存
                    if not hasattr(self.faiss_retriever,
                                   'node_embedding_cache') or not self.faiss_retriever.node_embedding_cache:
                        self.faiss_retriever.node_embedding_cache = {}
                        for node, embed in self.node_embedding_cache.items():
                            self.faiss_retriever.node_embedding_cache[node] = embed.clone().detach()

            except Exception as e:
                # 如果优化初始化失败，禁用性能优化
                self.enable_performance_optimizations = False

    def build_indices(self):
        """构建所有必要的FAISS索引和预计算节点嵌入，以实现高效的向量检索"""
        # 构建节点和关系的FAISS向量索引，这些索引用于快速相似性搜索，提高检索效率
        self.faiss_retriever.build_indices()
        # 预计算图中所有节点的向量嵌入表示，将计算结果缓存起来，避免在检索过程中重复计算
        self._precompute_node_embeddings()

    def _get_query_embedding(self, query: str) -> torch.Tensor:
        """
        用于将自然语言查询转换为向量表示（嵌入），以便进行向量相似度计算（最耗时的操作）
         """
        # 使用qa_encoder对查询语句进行编码，生成向量表示
        # 然后将结果转换为PyTorch张量，设置为浮点型，并移动到指定设备（CPU或GPU）
        query_embed = torch.tensor(
            self.qa_encoder.encode(query)
        ).float().to(self.device)
        return query_embed

    # ================= [Ch4 新增: 拓扑感知核心组件] =================

    def _extract_local_subgraph(self, candidate_nodes: List[str], hop: int = 1) -> nx.Graph:
        """构建局部诱导子图：包含候选节点及其 N-hop 邻居"""
        if not candidate_nodes: return nx.Graph()

        relevant_nodes = set(candidate_nodes)
        if hop > 0:
            current = list(relevant_nodes)
            for _ in range(hop):
                next_layer = []
                for n in current:
                    if self.graph.has_node(n):
                        next_layer.extend(list(self.graph.neighbors(n)))
                relevant_nodes.update(next_layer)
                current = next_layer

        # 使用 subgraph 创建子图视图，copy 确保它是独立的图对象
        return self.graph.subgraph(list(relevant_nodes)).copy()

    def _calculate_topology_scores(self, subgraph: nx.Graph, nodes: List[str]) -> Dict[str, float]:
        """计算拓扑重要性 (PageRank)"""
        if subgraph.number_of_nodes() == 0: return {n: 0.0 for n in nodes}
        try:
            # PageRank 衡量节点在局部引用网络中的核心地位
            return {n: s for n, s in nx.pagerank(subgraph, alpha=0.85, max_iter=50).items() if n in nodes}
        except:
            # 回退策略
            return {n: s for n, s in nx.degree_centrality(subgraph).items() if n in nodes}

    def _hybrid_scoring(self, node_scores: Dict[str, float], alpha: float, beta: float) -> List[str]:
        """执行混合评分：Score = alpha * Norm(Vec) + beta * Norm(Topo)"""
        nodes = list(node_scores.keys())
        if not nodes: return []

        # 1. 准备向量分
        vec_arr = np.array([node_scores[n] for n in nodes]).reshape(-1, 1)

        # 2. 计算拓扑分
        subgraph = self._extract_local_subgraph(nodes, hop=1)
        topo_map = self._calculate_topology_scores(subgraph, nodes)
        topo_arr = np.array([topo_map.get(n, 0.0) for n in nodes]).reshape(-1, 1)

        # 3. 归一化 (Min-Max)
        scaler = MinMaxScaler()
        norm_vec = scaler.fit_transform(vec_arr).flatten() if np.ptp(vec_arr) > 0 else np.ones(len(nodes))
        norm_topo = scaler.fit_transform(topo_arr).flatten() if np.ptp(topo_arr) > 0 else np.zeros(len(nodes))

        # 4. 加权融合
        final_scores = []
        for i, node in enumerate(nodes):
            score = alpha * norm_vec[i] + beta * norm_topo[i]
            final_scores.append((node, score))

        # 降序排列返回节点ID
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in final_scores]

    def _precompute_node_texts(self):
        """
        预计算所有节点的文本表示，避免在检索过程中重复提取文本。
        在初始化期间调用以构建文本缓存。
        """
        # 首先尝试从磁盘加载已缓存的节点文本
        # 如果成功加载，则直接返回，无需重新计算
        if self._load_node_text_cache():
            return

        start_time = time.time()

        # 获取图中所有节点的列表
        all_nodes = list(self.graph.nodes())
        processed_nodes = 0

        # 遍历图中的每个节点
        for node in all_nodes:
            try:
                # 获取节点的文本表示
                node_text = self._get_node_text(node)
                # 如果文本有效且不以'[Error'开头（表示获取成功）
                if node_text and not node_text.startswith('[Error'):
                    # 将节点文本存储到缓存中
                    self._node_text_cache[node] = node_text
                processed_nodes += 1

            except Exception as e:
                continue

        end_time = time.time()
        logger.info(
            f"已完成预计算 {len(self._node_text_cache)} 个节点的文本，耗时 {end_time - start_time:.2f} 秒")

        # 尝试将计算得到的节点文本缓存保存到磁盘
        try:
            self._save_node_text_cache()
        except Exception as e:
            logger.warning(f"Failed to save node text cache: {type(e).__name__}: {e}")

    def _save_node_text_cache(self):
        """将节点文本缓存保存到磁盘"""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_cache.pkl"
        try:
            # 检查缓存是否为空，如果为空则不保存
            if not self._node_text_cache:
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # 以二进制写入模式打开文件，使用pickle序列化保存缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(self._node_text_cache, f)

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"已保存包含 {len(self._node_text_cache)} 个条目的节点文本缓存到 {cache_path} (大小: {file_size} 字节)")
            return True

        except Exception as e:
            return False

    def _load_node_text_cache(self):
        """从磁盘加载节点文本缓存"""
        # 构建缓存文件路径
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_cache.pkl"
        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                # 如果文件小于1000字节，认为文件可能为空或已损坏
                if file_size < 1000:  # Less than 1KB likely empty or corrupted
                    logger.warning(f"警告: 缓存文件太小 ({file_size} 字节)，可能为空或已损坏")
                    return False

                # 以二进制读取模式打开文件，使用pickle反序列化加载缓存
                with open(cache_path, 'rb') as f:
                    self._node_text_cache = pickle.load(f)

                # 检查加载的缓存是否为空
                if not self._node_text_cache:
                    logger.warning("警告: 加载的缓存为空")
                    return False

                # 检查加载的缓存与当前图是否一致
                if not self._check_text_cache_consistency():
                    logger.warning("文本缓存与当前图不一致，将重新构建")
                    return False

                logger.info(
                    f"从 {cache_path} 加载了包含 {len(self._node_text_cache)} 个条目的节点文本缓存 (文件大小: {file_size} 字节)")
                return True

            except Exception as e:
                logger.error(f"加载节点文本缓存时出错: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"已移除损坏的缓存文件: {cache_path}")
                except Exception as e2:
                    logger.warning(f"无法移除损坏的缓存文件 {cache_path}: {type(e2).__name__}: {e2}")
        else:
            logger.warning(f"缓存文件不存在: {cache_path}")
        return False

    def _check_text_cache_consistency(self):
        """检查加载的文本缓存是否与当前图保持一致"""
        try:
            # 获取当前图中所有节点的集合
            current_nodes = set(self.graph.nodes())

            # 获取缓存中所有节点的集合
            cached_nodes = set(self._node_text_cache.keys())

            # 计算图中存在但缓存中缺失的节点
            missing_nodes = current_nodes - cached_nodes
            # 如果有缺失节点，记录日志并返回False
            if missing_nodes:
                logger.info(f"Text cache missing {len(missing_nodes)} nodes from current graph")
                return False

            # 计算缓存中存在但图中不存在的节点（多余的节点）
            extra_nodes = cached_nodes - current_nodes
            # 如果多余节点数量超过当前图节点数量的10%，记录警告并返回False
            if len(extra_nodes) > len(current_nodes) * 0.1:
                logger.warning(
                    f"Text cache has too many extra nodes: {len(extra_nodes)} extra vs {len(current_nodes)} current")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking text cache consistency: {e}")
            return False

    def _precompute_node_embeddings(self):
        """
        预计算所有节点的嵌入向量，避免在检索过程中重复编码
        """
        # 使用预计算锁确保线程安全，防止多个线程同时执行预计算
        with self.precompute_lock:
            # 如果节点嵌入已经预计算完成，直接返回
            if self.node_embeddings_precomputed:
                return

            # 尝试从磁盘加载节点嵌入缓存
            if self._load_node_embedding_cache():
                self.node_embeddings_precomputed = True
                return

            # 检查FAISS检索器是否已有节点嵌入缓存
            if hasattr(self.faiss_retriever, 'node_embedding_cache') and self.faiss_retriever.node_embedding_cache:
                # 如果有，则将FAISS检索器的缓存复制到当前对象的缓存中
                for node, embed in self.faiss_retriever.node_embedding_cache.items():
                    self.node_embedding_cache[node] = embed.clone().detach()
                self.node_embeddings_precomputed = True

                # 记录日志并保存缓存到磁盘
                logger.info(
                    f"Successfully loaded {len(self.node_embedding_cache)} node embeddings from faiss_retriever cache")
                self._save_node_embedding_cache()
                return

            logger.warning("No cache found, computing embeddings from scratch...")

            # 获取图中所有节点列表
            all_nodes = list(self.graph.nodes())
            batch_size = 100
            if self.config:
                batch_size = self.config.embeddings.batch_size * 3

            total_processed = 0
            for i in range(0, len(all_nodes), batch_size):
                # 获取当前批次的节点
                batch_nodes = all_nodes[i:i + batch_size]
                # 存储当前批次的文本
                batch_texts = []
                # 存储有效的节点（文本获取成功的节点）
                valid_nodes = []

                # 遍历当前批次的节点，获取节点文本
                for node in batch_nodes:
                    try:
                        node_text = self._get_node_text(node)
                        if node_text and not node_text.startswith('[Error'):
                            batch_texts.append(node_text)
                            valid_nodes.append(node)
                    except Exception as e:
                        logger.error(f"Error getting text for node {node}: {str(e)}")
                        continue

                # 如果当前批次有有效文本，则进行批处理编码
                if batch_texts:
                    try:
                        # 使用qa_encoder对整个批次的文本进行编码
                        batch_embeddings = self.qa_encoder.encode(batch_texts, convert_to_tensor=True)

                        # 将编码结果存储到节点嵌入缓存中
                        for j, node in enumerate(valid_nodes):
                            self.node_embedding_cache[node] = batch_embeddings[j]
                            total_processed += 1

                    except Exception as e:
                        # 如果批处理编码失败，记录错误并回退到逐个节点编码
                        logger.error(f"Error encoding batch {i // batch_size}: {str(e)}")
                        for node in valid_nodes:
                            try:
                                node_text = self._get_node_text(node)
                                # 再次检查节点文本有效性
                                if node_text and not node_text.startswith('[Error'):
                                    # 对单个节点进行编码并存储到缓存中
                                    embedding = torch.tensor(self.qa_encoder.encode(node_text)).float().to(self.device)
                                    self.node_embedding_cache[node] = embedding
                                    total_processed += 1
                            except Exception as e2:
                                logger.error(f"Error encoding node {node}: {str(e2)}")
                                continue

            # 标记节点嵌入预计算完成
            self.node_embeddings_precomputed = True
            logger.info(
                f"Node embeddings precomputed for {total_processed} nodes (cache size: {len(self.node_embedding_cache)})")

            # 尝试将节点嵌入缓存保存到磁盘
            try:
                self._save_node_embedding_cache()
            except Exception as e:
                logger.warning(f"Failed to save node embedding cache: {e}")
                logger.info("Continuing without saving cache...")

            # 清理节点缓存，如果缓存过大则只保留最近的条目
            self._cleanup_node_cache()

    def _save_node_embedding_cache(self):
        """将节点嵌入缓存保存到磁盘"""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_embedding_cache.pt"
        try:
            if not self.node_embedding_cache:
                logger.warning("Warning: No node embeddings to save!")
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # 创建用于保存到磁盘的CPU缓存字典
            cpu_cache = {}
            # 遍历节点嵌入缓存中的每个节点和对应的嵌入向量
            for node, embed in self.node_embedding_cache.items():
                if embed is not None:
                    try:
                        # 将嵌入向量转换为可以在CPU上处理的numpy数组格式
                        if hasattr(embed, 'detach'):
                            cpu_cache[node] = embed.detach().cpu().numpy()
                        elif isinstance(embed, np.ndarray):
                            cpu_cache[node] = embed
                        else:
                            cpu_cache[node] = np.array(embed)
                    except Exception as e:
                        logger.warning(f"Warning: Failed to convert embedding for node {node}: {e}")
                        continue

            if not cpu_cache:
                logger.warning("Warning: No valid embeddings to save!")
                return False

            try:
                # 创建用于PyTorch保存的张量缓存字典
                tensor_cache = {}
                # 将numpy数组转换回PyTorch张量
                for node, embed_array in cpu_cache.items():
                    if isinstance(embed_array, np.ndarray):
                        tensor_cache[node] = torch.from_numpy(embed_array).float()
                    else:
                        tensor_cache[node] = embed_array

                # 使用PyTorch的save方法保存缓存
                torch.save(tensor_cache, cache_path)
                logger.info(f"Saved node embedding cache using torch.save with tensor format")
            except Exception as torch_error:
                # 如果PyTorch保存失败，回退到使用numpy保存
                logger.error(f"torch.save failed: {torch_error}, using numpy.save")
                cache_path_npz = cache_path.replace('.pt', '.npz')
                # 使用numpy的压缩格式保存
                np.savez_compressed(cache_path_npz, **cpu_cache)
                cache_path = cache_path_npz
                logger.error(f"Saved using numpy.savez_compressed format")

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"Saved node embedding cache with {len(cpu_cache)} entries to {cache_path} (size: {file_size} bytes)")
            return True

        except Exception as e:
            logger.error(f"Error saving node embedding cache: {e}")
            return False

    def _load_node_embedding_cache(self):
        """从磁盘加载节点嵌入缓存"""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_embedding_cache.pt"
        cache_path_npz = cache_path.replace('.pt', '.npz')

        # 首先尝试加载.npyz格式的缓存文件
        if os.path.exists(cache_path_npz):
            try:
                file_size = os.path.getsize(cache_path_npz)
                logger.info(f"Loading node embedding cache from {cache_path_npz} (file size: {file_size} bytes)")

                # 使用numpy加载压缩的.npyz文件
                numpy_cache = np.load(cache_path_npz)

                if len(numpy_cache.files) == 0:
                    logger.warning("Warning: Loaded cache is empty")
                    return False

                # 清空当前的节点嵌入缓存
                self.node_embedding_cache.clear()

                # 遍历缓存中的所有节点
                for node in numpy_cache.files:
                    try:
                        # 加载节点的嵌入数组
                        embed_array = numpy_cache[node]
                        # 将numpy数组转换为PyTorch张量并移到指定设备
                        embed_tensor = torch.from_numpy(embed_array).float().to(self.device)
                        # 存储到节点嵌入缓存中
                        self.node_embedding_cache[node] = embed_tensor
                    except Exception as e:
                        logger.warning(f"Warning: Failed to load embedding for node {node}: {e}")
                        continue

                numpy_cache.close()

                # 检查加载的缓存与当前图是否一致
                if not self._check_embedding_cache_consistency():
                    logger.info("Embedding cache inconsistent with current graph, will rebuild")
                    return False

                logger.info(
                    f"Loaded node embedding cache with {len(self.node_embedding_cache)} entries from {cache_path_npz}")
                return True

            except Exception as e:
                logger.error(f"Error loading numpy cache: {e}")

        # 如果.npyz格式不存在或加载失败，回退到.pt格式
        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:
                    logger.warning(f"警告: 缓存文件太小 ({file_size} 字节)，可能为空或已损坏")
                    return False

                try:
                    # 尝试使用PyTorch加载.pt文件
                    cpu_cache = torch.load(cache_path, map_location='cpu', weights_only=False)
                except TypeError:
                    # 处理旧版本PyTorch可能不支持weights_only参数的情况
                    cpu_cache = torch.load(cache_path, map_location='cpu')
                except Exception as e:
                    # 处理numpy序列化相关的问题
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

                # 清空当前节点嵌入缓存
                self.node_embedding_cache.clear()

                # 遍历加载的缓存项
                for node, embed in cpu_cache.items():
                    if embed is not None:
                        try:
                            if isinstance(embed, np.ndarray):
                                embed_tensor = torch.from_numpy(embed).float()
                            else:
                                embed_tensor = embed.cpu() if hasattr(embed, 'cpu') else embed

                            if self.device == "cuda" and torch.cuda.is_available():
                                embed_tensor = embed_tensor.to(self.device)
                            else:
                                embed_tensor = embed_tensor.to("cpu")

                            self.node_embedding_cache[node] = embed_tensor
                        except Exception as e:
                            logger.error(f"Warning: Failed to load embedding for node {node}: {e}")
                            continue

                # 检查加载的缓存与当前图是否一致
                if not self._check_embedding_cache_consistency():
                    logger.info("Embedding cache inconsistent with current graph, will rebuild")
                    return False
                logger.info(
                    f"从 {cache_path} 加载了包含 {len(self.node_embedding_cache)} 个条目的节点嵌入缓存 (文件大小: {file_size / 1024:.1f} KB)")
                return True

            except Exception as e:
                logger.error(f"Error loading node embedding cache: {e}")
                try:
                    # 尝试删除损坏的缓存文件
                    os.remove(cache_path)
                    logger.info(f"移除损坏的缓存文件: {cache_path}")
                except Exception as e3:
                    logger.warning(f"Failed to remove corrupted cache file {cache_path}: {type(e3).__name__}: {e3}")
        else:
            logger.info(f"缓存文件不存在: {cache_path}")
        return False

    def _check_embedding_cache_consistency(self):
        """检查加载的嵌入缓存是否与当前图保持一致"""
        try:
            # 获取当前图中所有节点的集合
            current_nodes = set(self.graph.nodes())

            # 获取缓存中所有节点的集合
            cached_nodes = set(self.node_embedding_cache.keys())

            # 计算图中存在但缓存中缺失的节点
            missing_nodes = current_nodes - cached_nodes
            if missing_nodes:
                logger.info(f"Embedding cache missing {len(missing_nodes)} nodes from current graph")
                return False

            # 计算缓存中存在但图中不存在的节点（多余的节点）
            extra_nodes = cached_nodes - current_nodes
            if len(extra_nodes) > len(current_nodes) * 0.1:  # Allow 10% tolerance
                logger.info(
                    f"Embedding cache has too many extra nodes: {len(extra_nodes)} extra vs {len(current_nodes)} current")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking embedding cache consistency: {e}")
            return False

    def _cleanup_node_cache(self):
        """
        清理节点嵌入缓存以节省内存
        """
        # 使用节点嵌入缓存锁确保线程安全
        with self.cache_locks['node_embedding']:
            # 检查节点嵌入缓存的大小是否超过5000个条目
            if len(self.node_embedding_cache) > 5000:
                # 只保留最近的5000个条目
                recent_nodes = list(self.node_embedding_cache.keys())[-5000:]
                self.node_embedding_cache = {k: self.node_embedding_cache[k] for k in recent_nodes}

    def retrieve(self, question: str, alpha: float = 1.0, beta: float = 0.0) -> Dict:
        """
        执行增强的双路径检索过程，包含查询理解和缓存机制。
        
        参数:
        question: 查询问题

        返回:
            包含以下内容的字典：
            - path1_results: 节点/关系嵌入检索结果及一跳三元组
            - path2_results: 仅三元组检索的结果
            - chunk_ids: 所有检索到节点的文本块ID集合
        """
        # 记录开始时间，用于计算查询编码时间
        start_time = time.time()

        # 将自然语言问题转换为向量表示（嵌入）
        question_embed = self._get_query_embedding(question)
        query_time = time.time() - start_time

        # 初始化存储所有文本块ID的集合
        all_chunk_ids = set()

        # 根据recall_paths参数决定使用单路径还是双路径检索
        # 目前主要修改了 path1 (节点关系检索)
        # 如果涉及 path2 (triples only)，暂未修改，保持原样
        if self.recall_paths == 1:
            # 单路径检索模式
            path_start = time.time()
            # 执行节点/关系嵌入检索
            path1_results = self._node_relation_retrieval(question_embed, question, alpha, beta)
            path1_time = time.time() - path_start
            # 记录查询编码和路径1检索的时间日志
            logger.info(f"查询编码耗时: {query_time:.3f}秒, 路径1检索耗时: {path1_time:.3f}秒")

            # 从路径1结果中提取节点相关的文本块ID
            path1_chunk_ids = self._extract_chunk_ids_from_nodes(path1_results['top_nodes'])
            # 将路径1的文本块ID添加到总集合中
            all_chunk_ids.update(path1_chunk_ids)

            # 如果路径1结果中包含文本块检索结果，则也提取其中的文本块ID
            if 'chunk_results' in path1_results and path1_results['chunk_results']:
                chunk_chunk_ids = set(path1_results['chunk_results'].get('chunk_ids', []))
                all_chunk_ids.update(chunk_chunk_ids)

            # 将所有文本块ID转换为列表形式
            limited_chunk_ids = list(all_chunk_ids)

            # 构造单路径结果
            result = {
                "path1_results": path1_results,
                "chunk_ids": limited_chunk_ids
            }
        else:
            # 多路径模式暂不支持混合评分，或需要去修改 _parallel_dual_path_retrieval 内部
            # 建议实验时使用 recall_paths=1
            # 双路径并行检索模式
            parallel_start = time.time()
            # 并行执行双路径检索
            result = self._parallel_dual_path_retrieval(question_embed, question)
            parallel_time = time.time() - parallel_start
            logger.info(f"Query encoding: {query_time:.3f}s, Parallel retrieval: {parallel_time:.3f}s")

        return question_embed, result

    def retrieve_with_type_filtering(self, question: str, involved_types: dict = None, alpha: float = 1.0,
                                     beta: float = 0.0) -> Dict:
        """
        增强检索，先基于类型过滤再进行相似度搜索。

        参数:
            question: 查询问题
            involved_types: 包含相关schema类型的字典

        返回:
            包含过滤后检索结果的字典
        """
        start_time = time.time()

        # 将自然语言问题转换为向量表示（嵌入）
        logger.info("开始将自然语言问题转换为向量表示（嵌入）")
        question_embed = self._get_query_embedding(question)
        query_time = time.time() - start_time

        # 检查是否提供了类型信息且类型信息不为空
        if involved_types and any(involved_types.get(k, []) for k in ['nodes', 'relations', 'attributes']):
            # 使用基于类型的过滤路径
            type_start = time.time()
            # 执行类型基础的检索
            logger.info("开始使用基于类型的过滤路径")
            type_filtered_results = self._type_based_retrieval(question_embed, question, involved_types, alpha, beta)
            type_filtering_time = time.time() - type_start
            logger.info(f"查询编码耗时: {query_time:.3f}秒, 基于类型的检索耗时: {type_filtering_time:.3f}秒")

            return question_embed, type_filtered_results
        else:
            # 如果没有提供类型信息或类型信息为空，则回退到原始检索方法
            original_results = self.retrieve(question)
            logger.info(f"Query encoding: {query_time:.3f}s, Fallback to original retrieval")
            return original_results

    def _type_based_retrieval(self, question_embed: torch.Tensor, question: str, involved_types: dict,
                              alpha: float = 1.0, beta: float = 0.0) -> Dict:
        """
        执行混合检索：基于类型的节点/关系路径过滤 + 原始其他路径。

        参数:
            question_embed: 问题嵌入张量
            question: 原始问题文本
            involved_types: 包含节点类型、关系和属性的字典

        返回:
            包含混合检索结果的字典
        """
        # 检查recall_paths参数以决定使用单路径还是多路径检索
        if self.recall_paths == 1:
            # 单路径模式：仅对节点/关系路径进行类型过滤
            logger.info("开始使用单路径模式")
            filtered_results = self._type_filtered_node_relation_retrieval(question_embed, question, involved_types,
                                                                           alpha, beta)
            return filtered_results
        else:
            # 多路径模式：仅对节点/关系路径进行类型过滤，保持其他路径原始状态
            logger.info("开始使用多路径模式")
            hybrid_results = self._hybrid_type_filtered_retrieval(question_embed, question, involved_types)
            return hybrid_results

    def _type_filtered_node_relation_retrieval(self, question_embed: torch.Tensor, question: str,
                                               involved_types: dict, alpha: float = 1.0, beta: float = 0.0) -> Dict:
        """
        单路径检索，仅在节点/关系路径上进行类型过滤。

        参数:
            question_embed: 问题嵌入张量
            question: 原始问题文本
            involved_types: 包含涉及的schema类型的字典

        返回:
            包含类型过滤后检索结果的字典
        """
        # 从involved_types中提取目标节点类型列表
        target_node_types = involved_types.get('nodes', [])

        # 根据目标节点类型过滤图谱中的节点
        type_filtered_nodes = self._filter_nodes_by_schema_type(target_node_types)

        # 如果有类型过滤后的节点，则在这些节点上执行相似度搜索
        if type_filtered_nodes:
            # ================= [Ch4 修改: 混合评分支持] =================
            if beta > 0:
                logger.info("类型过滤关系检索开始使用混合评分")
                # 1. 计算所有过滤后节点的向量分数
                # 注意：如果节点太多(>500)，可能需要先向量初筛再算分，这里假设过滤后数量可控
                all_scores = self._batch_calculate_entity_similarities(question_embed, type_filtered_nodes)

                # 从 all_scores 中提取 type_filtered_nodes 对应的分数，构建 node_scores 字典
                node_scores = {}
                for node in type_filtered_nodes:
                    if node in all_scores:
                        node_scores[node] = float(all_scores[node])

                # 2. 混合重排
                logger.info("混合重排开始")
                top_nodes = self._hybrid_scoring(node_scores, alpha, beta)[:self.top_k]

                # 3. 获取上下文 (复用 helper 方法)
                one_hop_triples = self._get_one_hop_triples_from_nodes(top_nodes)
                chunk_ids = self._extract_chunk_ids_from_nodes(top_nodes)

                # 构造结果结构 (模拟 filtered_node_results 的输出)
                result = {
                    "path1_results": {
                        "top_nodes": top_nodes,
                        "one_hop_triples": one_hop_triples
                    },
                    "chunk_ids": list(chunk_ids)
                }
            else:
                # [原逻辑] 使用原有搜索方法
                logger.info("类型过滤关系检索开始使用原始评分")
                filtered_node_results = self._similarity_search_on_filtered_nodes(question_embed, type_filtered_nodes)
                one_hop_triples = self._get_one_hop_triples_from_nodes(filtered_node_results['top_nodes'])
                chunk_ids = self._extract_chunk_ids_from_nodes(filtered_node_results['top_nodes'])
                result = {
                    "path1_results": {
                        "top_nodes": filtered_node_results['top_nodes'],
                        "one_hop_triples": one_hop_triples
                    },
                    "chunk_ids": list(chunk_ids)
                }
            # ==========================================================
        else:
            # 如果没有类型过滤后的节点，则回退到标准的节点/关系检索
            result = self._node_relation_retrieval(question_embed, question)

        return result

    def _hybrid_type_filtered_retrieval(self, question_embed: torch.Tensor, question: str,
                                        involved_types: dict) -> Dict:
        """
        多路径检索：类型过滤的节点/关系路径 + 原始其他路径。

        参数:
            question_embed: 问题嵌入张量
            question: 原始问题文本
            involved_types: 包含涉及的schema类型的字典

        返回:
            包含混合类型过滤检索结果的字典
        """
        # 从involved_types中提取目标节点类型列表
        target_node_types = involved_types.get('nodes', [])

        # Path 1: 类型过滤的节点/关系检索
        # 如果提供了目标节点类型
        if target_node_types:
            # 根据目标节点类型过滤图谱中的节点
            type_filtered_nodes = self._filter_nodes_by_schema_type(target_node_types)
            # 如果有类型过滤后的节点
            if type_filtered_nodes:
                # 在过滤后的节点上执行类型过滤的节点/关系路径检索
                logger.info("开始使用类型过滤的节点/关系检索")
                path1_results = self._type_filtered_node_relation_path(question_embed, type_filtered_nodes)
            else:
                # 如果没有类型过滤后的节点，则回退到标准的节点/关系检索
                logger.info("开始使用原始节点/关系检索")
                path1_results = self._node_relation_retrieval(question_embed, question)
        else:
            # 如果没有提供目标节点类型，则执行标准的节点/关系检索
            logger.info("开始使用原始节点/关系检索")
            path1_results = self._node_relation_retrieval(question_embed, question)

        # Path 2: 仅三元组检索
        # 执行三元组-only检索路径
        logger.info("开始使用三元组-only检索")
        path2_results = self._triple_only_retrieval(question_embed)

        # 收集所有文本块ID
        all_chunk_ids = set()
        # 从路径1结果中提取节点相关的文本块ID
        path1_chunk_ids = self._extract_chunk_ids_from_nodes(path1_results['top_nodes'])
        # 将路径1的文本块ID添加到总集合中
        all_chunk_ids.update(path1_chunk_ids)

        # 如果路径2结果中包含文本块检索结果，则也提取其中的文本块ID
        if 'chunk_results' in path2_results and path2_results['chunk_results']:
            chunk_chunk_ids = set(path2_results['chunk_results'].get('chunk_ids', []))
            all_chunk_ids.update(chunk_chunk_ids)

        # 构造最终结果字典
        result = {
            "path1_results": path1_results,
            "path2_results": path2_results,
            "chunk_ids": list(all_chunk_ids)
        }

        return result

    def _type_filtered_node_relation_path(self, question_embed: torch.Tensor, filtered_nodes: list) -> Dict:
        """
        执行类型过滤的节点/关系检索路径。

        参数:
            question_embed: 问题嵌入张量，用于计算相似度
            filtered_nodes: 已经过类型过滤的节点列表

        返回:
            包含顶级节点和一跳三元组的字典
        """
        # 在类型过滤后的节点上执行相似度搜索
        # 这会返回与问题最相关的节点列表
        logger.info("开始使用类型过滤的节点相似度搜索")
        filtered_node_results = self._similarity_search_on_filtered_nodes(question_embed, filtered_nodes)
        logger.info("完成类型过滤的节点相似度搜索")

        # 基于检索到的顶级节点，获取它们的一跳三元组（邻接关系）
        # 这些三元组提供了节点之间的直接关系信息
        one_hop_triples = self._get_one_hop_triples_from_nodes(filtered_node_results['top_nodes'])

        # 返回结构化的结果，包含：
        # 1. top_nodes: 与问题最相关的顶级节点
        # 2. one_hop_triples: 这些节点的一跳关系三元组
        return {
            "top_nodes": filtered_node_results['top_nodes'],
            "one_hop_triples": one_hop_triples
        }

    def _similarity_search_on_filtered_nodes(self, question_embed: torch.Tensor, filtered_nodes: list) -> Dict:
        """
        仅在过滤后的节点上执行相似度搜索。

        参数:
            question_embed: 问题的嵌入向量（张量格式）
            filtered_nodes: 经过类型过滤后的节点ID列表

        返回:
            包含最相似节点列表的字典，键为"top_nodes"
        """
        if not filtered_nodes:
            return {"top_nodes": []}

        # 初始化存储过滤节点嵌入和映射关系的变量
        filtered_node_embeddings = []  # 存储过滤节点的嵌入向量
        filtered_node_map = {}  # 映射索引到节点ID的关系

        # 遍历所有过滤后的节点
        for idx, node_id in enumerate(filtered_nodes):
            # 检查节点是否存在于FAISS检索器的节点映射中
            if node_id in self.faiss_retriever.node_map.values():
                # 查找该节点在FAISS索引中的原始索引
                original_idx = None
                for orig_idx, orig_node_id in self.faiss_retriever.node_map.items():
                    if orig_node_id == node_id:
                        original_idx = orig_idx
                        break

                # 如果找到了原始索引
                if original_idx is not None:
                    # 从FAISS索引中重建该节点的嵌入向量
                    node_embedding = self.faiss_retriever.node_index.reconstruct(int(original_idx))
                    # 将嵌入向量添加到列表中
                    filtered_node_embeddings.append(node_embedding)
                    # 建立新的索引到节点ID的映射关系
                    filtered_node_map[len(filtered_node_embeddings) - 1] = node_id

        # 如果成功获取了过滤节点的嵌入向量
        if filtered_node_embeddings:
            logger.info("成功获取了过滤节点的嵌入向量")
            filtered_embeddings_array = np.array(filtered_node_embeddings).astype('float32')

            # 创建临时的FAISS索引用于相似度搜索
            temp_index = faiss.IndexFlatIP(filtered_embeddings_array.shape[1])  # 使用内积相似度
            temp_index.add(filtered_embeddings_array)  # 将嵌入向量添加到索引中

            # 确定搜索的节点数量（不超过top_k和实际节点数的最小值）
            search_k = min(self.top_k, len(filtered_node_embeddings))
            # 执行相似度搜索，找到与问题最相似的节点
            logger.info("开始执行相似度搜索,，找到与问题最相似的节点")
            _, indices = temp_index.search(question_embed.cpu().reshape(1, -1), search_k)

            # 根据索引映射回实际的节点ID
            top_filtered_nodes = [filtered_node_map[idx] for idx in indices[0] if idx in filtered_node_map]
        else:
            # 如果没有获取到嵌入向量，则直接取前top_k个节点
            top_filtered_nodes = filtered_nodes[:self.top_k]

        # 返回最相似的节点列表
        return {"top_nodes": top_filtered_nodes}

    def _get_one_hop_triples_from_nodes(self, node_list: list) -> list:

        """
           从给定的节点列表中获取一跳邻居三元组

           Args:
               node_list: 节点ID列表，用于查找与其相连的三元组

           Returns:
               包含一跳三元组的列表，每个三元组格式为(头节点名, 关系, 尾节点名)
               最多返回self.top_k个三元组
           """
        one_hop_triples = []
        node_set = set(node_list)

        # 遍历图中所有边，data=True表示同时获取边的属性信息
        for u, v, data in self.graph.edges(data=True):
            # 检查边的头节点或尾节点是否在给定的节点集合中
            if u in node_set or v in node_set:
                # 从边的属性中获取关系信息，默认为空字符串
                relation = data.get('relation', '')
                # 获取头节点和尾节点的名称
                u_name = self._get_node_name(u)
                v_name = self._get_node_name(v)
                # 将三元组添加到结果列表中
                one_hop_triples.append((u_name, relation, v_name))

        # 返回前top_k个三元组，限制结果数量
        return one_hop_triples[:self.top_k]

    def _filter_nodes_by_schema_type(self, target_types: list) -> list:
        """
        根据schema_type属性过滤节点

        Args:
            target_types: 目标schema类型列表

        Returns:
            过滤后的节点ID列表
    """
        if not target_types:
            return list(self.graph.nodes())

        # 初始化存储过滤后节点的列表
        filtered_nodes = []
        # 遍历图中所有节点及其数据
        for node_id, node_data in self.graph.nodes(data=True):
            # 获取节点的属性信息，默认为空字典
            node_properties = node_data.get('properties', {})
            # 从属性中获取schema_type，默认为空字符串
            node_schema_type = node_properties.get('schema_type', '')

            # 如果节点的schema_type在目标类型列表中，则添加到结果列表
            if node_schema_type in target_types:
                filtered_nodes.append(node_id)
            # 向后兼容：如果节点没有schema_type但标签为'entity'，也添加到结果列表
            # 这确保了旧版本数据的兼容性
            elif not node_schema_type and node_data.get('label') == 'entity':
                filtered_nodes.append(node_id)

        return filtered_nodes

    def _get_node_name(self, node_id: str) -> str:
        """
        获取节点的名称属性

        Args:
            node_id: 节点ID

        Returns:
            节点的名称属性值，如果不存在则返回节点ID
        """
        # 从图谱中获取节点数据，如果节点不存在则返回空字典
        node_data = self.graph.nodes.get(node_id, {})
        # 从节点数据中获取properties属性，如果不存在则返回空字典
        properties = node_data.get('properties', {})
        # 从properties中获取name属性，如果不存在则返回节点ID作为默认值
        return properties.get('name', node_id)

    def _parallel_dual_path_retrieval(self, question_embed: torch.Tensor, question: str) -> Dict:
        """
        并行执行双路径检索并整合结果

        Args:
            question_embed: 问题的嵌入向量
            question: 原始问题文本

        Returns:
            包含两个路径检索结果和文本块ID的字典
        """
        all_chunk_ids = set()
        start_time = time.time()

        # 设置并行执行的最大工作线程数
        max_workers = 4
        if self.config:
            max_workers = self.config.retrieval.faiss.max_workers
        # 使用线程池并发执行两个检索路径
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交路径1任务：节点/关系嵌入检索
            path1_future = executor.submit(self._node_relation_retrieval, question_embed, question)
            # 提交路径2任务：仅三元组检索
            path2_future = executor.submit(self._triple_only_retrieval, question_embed)

            # 等待并获取两个路径的检索结果
            path1_results = path1_future.result()
            path2_results = path2_future.result()

        start_time = time.time()

        # 从路径1的结果中提取节点相关的文本块ID
        path1_chunk_ids = self._extract_chunk_ids_from_nodes(path1_results['top_nodes'])
        # 从路径2的结果中提取三元组相关的文本块ID
        path2_chunk_ids = self._extract_chunk_ids_from_triple_nodes(path2_results['scored_triples'])

        # 从路径1的文本块检索结果中提取文本块ID（如果存在）
        path3_chunk_ids = set()
        if 'chunk_results' in path1_results and path1_results['chunk_results']:
            path3_chunk_ids = set(path1_results['chunk_results'].get('chunk_ids', []))

        # 将所有来源的文本块ID合并到集合中
        all_chunk_ids.update(path1_chunk_ids)
        all_chunk_ids.update(path2_chunk_ids)
        all_chunk_ids.update(path3_chunk_ids)

        # 限制返回的文本块ID数量为top_k
        limited_chunk_ids = list(all_chunk_ids)[:self.top_k]

        end_time = time.time()
        logger.info(f"Time taken to extract chunk IDs: {end_time - start_time} seconds")

        # 返回整合后的检索结果
        return {
            "path1_results": path1_results,  # 路径1的检索结果
            "path2_results": path2_results,  # 路径2的检索结果
            "chunk_ids": limited_chunk_ids  # 限制数量的文本块ID列表
        }

    def _execute_retrieval_strategies_parallel(self, question_embed: torch.Tensor, question: str, q_embed) -> Dict:
        """
        并行执行多种检索策略以获得最佳性能

        参数:
            question_embed: 编码后的问题张量
            question: 原始问题文本
            q_embed: 为FAISS转换后的查询嵌入向量

        返回:
            包含所有策略结果的字典
        """
        # 初始化结果字典，用于存储各种检索策略的结果
        results = {
            'faiss_nodes': [],  # FAISS节点搜索结果
            'faiss_relations': [],  # FAISS关系搜索结果
            'keyword_nodes': [],  # 关键词策略找到的节点
            'path_triples': [],  # 路径策略找到的三元组
            'keywords': []  # 提取的关键词
        }

        # 设置并行执行的最大工作线程数，默认为4
        max_workers = 4
        if self.config:
            max_workers = self.config.retrieval.faiss.max_workers

        # 使用线程池并发执行多个检索策略
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            # 提交FAISS节点搜索任务: 搜索最多min(top_k*3, 50)个节点
            faiss_node_future = executor.submit(
                self._faiss_node_search, q_embed, min(self.top_k * 3, 50)
            )

            # 提交FAISS关系搜索任务: 搜索top_k个关系
            faiss_relation_future = executor.submit(
                self._faiss_relation_search, q_embed, self.top_k
            )

            # 如果有问题文本，则提交关键词策略任务
            if question:
                keyword_future = executor.submit(self._keyword_strategy, question, question_embed)
            else:
                keyword_future = None

            # 如果有问题文本，则提交路径策略任务
            if question:
                path_future = executor.submit(self._path_strategy, question, question_embed)
            else:
                path_future = None

            # 收集FAISS节点搜索结果
            try:
                results['faiss_nodes'] = faiss_node_future.result()
            except Exception as e:
                logger.error(f"FAISS node search failed: {e}")

            # 收集FAISS关系搜索结果
            try:
                results['faiss_relations'] = faiss_relation_future.result()
            except Exception as e:
                logger.error(f"FAISS relation search failed: {e}")

            # 收集关键词策略结果
            if keyword_future:
                try:
                    keyword_results = keyword_future.result()
                    results['keyword_nodes'] = keyword_results.get('nodes', [])
                    results['keywords'] = keyword_results.get('keywords', [])
                except Exception as e:
                    logger.error(f"Keyword strategy failed: {e}")

            # 收集路径策略结果
            if path_future:
                try:
                    results['path_triples'] = path_future.result()
                except Exception as e:
                    logger.error(f"Path strategy failed: {e}")

        return results

    def _faiss_node_search(self, q_embed, search_k: int) -> List[str]:
        """执行带缓存的FAISS节点搜索"""
        # 创建搜索缓存键，基于查询向量的哈希值和搜索数量
        search_key = f"node_search_{hash(q_embed.tobytes())}_{search_k}"

        # 检查是否有缓存的结果，如果有则直接使用缓存
        if hasattr(self, 'faiss_search_cache') and search_key in self.faiss_search_cache:
            D_nodes, I_nodes = self.faiss_search_cache[search_key]
        else:
            # 没有缓存则执行实际的FAISS搜索
            D_nodes, I_nodes = self.faiss_retriever.node_index.search(
                q_embed.reshape(1, -1), search_k
            )
            # 初始化缓存字典（如果没有的话）
            if not hasattr(self, 'faiss_search_cache'):
                self.faiss_search_cache = {}
            # 将搜索结果存入缓存
            self.faiss_search_cache[search_key] = (D_nodes, I_nodes)

        # 处理搜索结果，将索引转换为实际的节点ID
        candidate_nodes = []
        for idx in I_nodes[0]:  # I_nodes[0]包含返回的最相似向量的索引
            if idx == -1:
                continue
            try:
                # 通过node_map将FAISS索引转换为实际的节点ID
                node_id = self.faiss_retriever.node_map[str(idx)]
                # 检查节点是否在图中存在
                if node_id in self.graph.nodes:
                    candidate_nodes.append(node_id)
            except KeyError:
                continue

        return candidate_nodes

    def _faiss_relation_search(self, q_embed, top_k: int) -> List[str]:
        """执行带缓存的FAISS关系搜索"""
        # 创建关系搜索的缓存键，基于查询向量的哈希值和返回结果数量
        search_key = f"relation_search_{hash(q_embed.tobytes())}_{top_k}"

        # 检查是否有缓存的结果，如果有则直接使用缓存
        if hasattr(self, 'faiss_search_cache') and search_key in self.faiss_search_cache:
            D_relations, I_relations = self.faiss_search_cache[search_key]
        else:
            # 没有缓存则执行实际的FAISS关系搜索
            D_relations, I_relations = self.faiss_retriever.relation_index.search(
                q_embed.reshape(1, -1), top_k
            )
            # 初始化缓存字典（如果没有的话）
            if not hasattr(self, 'faiss_search_cache'):
                self.faiss_search_cache = {}
            # 将搜索结果存入缓存
            self.faiss_search_cache[search_key] = (D_relations, I_relations)

        # 处理搜索结果，将索引转换为实际的关系名称
        relations = []
        for idx in I_relations[0]:  # I_relations[0]包含返回的最相似向量的索引
            if idx == -1:
                continue
            try:
                # 通过relation_map将FAISS索引转换为实际的关系名称
                relation = self.faiss_retriever.relation_map[str(idx)]
                relations.append(relation)
            except KeyError:
                continue

        # 返回找到的关系列表
        return relations

    def _keyword_strategy(self, question: str, question_embed: torch.Tensor) -> Dict:
        """执行关键词提取和搜索策略"""
        # 从问题文本中提取关键词
        # 使用自然语言处理技术识别问题中的重要词汇
        keywords = self._extract_query_keywords(question)

        # 基于提取的关键词在图谱中搜索相关的节点
        # 这一步会返回与关键词匹配的节点列表
        keyword_nodes = self._keyword_based_node_search(keywords)

        # 返回关键词和匹配节点的结果字典
        return {
            'keywords': keywords,
            'nodes': keyword_nodes
        }

    def _path_strategy(self, question: str):
        """执行基于路径的搜索策略"""
        # 从问题文本中提取关键词
        # 这个方法调用会识别问题中的重要实体和词汇
        self._extract_query_keywords(question)

        # 方法直接返回，没有进一步的路径搜索逻辑
        return

    def _node_relation_retrieval(self, question_embed: torch.Tensor, question: str = "", alpha: float = 1.0,
                                 beta: float = 0.0) -> Dict:
        """执行节点/关系联合检索，结合多种检索策略获取最相关的信息"""
        # 设置并行执行的最大工作线程数
        max_workers = 4
        if self.config:
            max_workers = self.config.retrieval.faiss.max_workers

        # 使用线程池并发执行多个检索任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 转换查询向量以适应FAISS索引
            q_embed = self.faiss_retriever.transform_vector(question_embed)
            # [修改 1] 动态调整初筛数量：开启拓扑感知时扩大范围，否则保持原逻辑
            current_k = self.top_k
            search_k = min(self.top_k * 5, 200) if beta > 0 else min(self.top_k * 3, 50)

            # 提交FAISS节点搜索任务
            future_faiss_nodes = executor.submit(self._execute_faiss_node_search, q_embed.cpu().numpy(), search_k)

            # 初始化关键词相关任务变量
            future_keywords = future_keyword_nodes = None
            # 如果提供了问题文本，则执行关键词策略
            if question:
                # 提交关键词提取任务
                future_keywords = executor.submit(
                    self._extract_query_keywords,
                    question
                )
                # 提交基于关键词的节点搜索任务
                future_keyword_nodes = executor.submit(
                    self._get_keyword_based_nodes,
                    future_keywords
                )

            # 提交FAISS关系搜索任务
            future_faiss_relations = executor.submit(
                self._execute_faiss_relation_search,
                q_embed.cpu().numpy()
            )

            # 提交文本块检索任务
            future_chunk_retrieval = executor.submit(
                self._chunk_embedding_retrieval,
                question_embed,
                self.top_k
            )

            # 获取FAISS节点搜索结果
            faiss_candidate_nodes = future_faiss_nodes.result()

            # 提交FAISS候选节点相似度计算任务
            future_faiss_sim = executor.submit(
                self._batch_calculate_entity_similarities,
                question_embed,
                faiss_candidate_nodes
            )

            # 处理关键词搜索结果
            keyword_candidate_nodes = []
            if future_keyword_nodes:
                # 获取关键词节点搜索结果
                keyword_nodes = future_keyword_nodes.result()
                # 获取FAISS节点集合，用于去重
                existing_faiss_nodes = set(faiss_candidate_nodes)
                # 过滤掉已经在FAISS结果中的节点
                keyword_candidate_nodes = [
                    n for n in keyword_nodes
                    if n not in existing_faiss_nodes
                ]

            # 如果有关键词候选节点，提交相似度计算任务
            future_keyword_sim = executor.submit(
                self._batch_calculate_entity_similarities,
                question_embed,
                keyword_candidate_nodes
            ) if keyword_candidate_nodes else None

            # --- 收集所有分数 ---
            node_scores = {}  # {node_id: vector_score}

            faiss_sims = future_faiss_sim.result()
            for n, s in faiss_sims.items(): node_scores[n] = float(s)

            if future_keyword_sim:
                kw_sims = future_keyword_sim.result()
                for n, s in kw_sims.items():
                    if s > 0.05: node_scores[n] = float(s)

            # ================= [修改 2: 混合重排序逻辑] =================
            if beta > 0 and node_scores:
                logger.info(f"触发拓扑重排序 (Candidates={len(node_scores)}, α={alpha}, β={beta})")
                # 使用混合评分重新排序所有候选节点
                sorted_all_nodes = self._hybrid_scoring(node_scores, alpha, beta)
                # 截取最终的 Top-K
                top_nodes = sorted_all_nodes[:current_k]
            else:
                # [原逻辑] 仅根据向量相似度排序
                sorted_candidates = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
                top_nodes = [n for n, s in sorted_candidates[:current_k] if s > 0.05]
            # ==========================================================

            # 获取FAISS关系搜索结果
            all_relations = future_faiss_relations.result()

            # 提交路径搜索任务
            future_path_triples = executor.submit(
                self._path_based_search,
                top_nodes,
                future_keywords.result() if future_keywords else [],
                max_depth=2
            ) if question else None

            # 提交邻居扩展任务
            future_neighbor_triples = executor.submit(
                self._optimized_neighbor_expansion,
                top_nodes,
                question_embed
            )

            # 获取各种三元组结果
            one_hop_triples = future_neighbor_triples.result()
            path_triples = future_path_triples.result() if future_path_triples else []
            # 获取关系匹配的三元组
            relation_triples = self._get_relation_matched_triples(
                top_nodes,
                all_relations
            )

            # 合并所有三元组并去重
            all_triples = list({
                triple for triple in
                one_hop_triples + path_triples + relation_triples
            })

            chunk_results = future_chunk_retrieval.result()

        # 返回检索结果
        return {
            "top_nodes": top_nodes,  # 顶级相关节点
            "top_relations": all_relations,  # 顶级相关关系
            "one_hop_triples": all_triples,  # 所有相关三元组
            "chunk_results": chunk_results  # 文本块检索结果
        }

    def _execute_faiss_node_search(self, q_embed, search_k: int) -> List[str]:
        """执行FAISS节点搜索并返回对应的节点ID列表"""

        # 使用FAISS节点索引进行相似性搜索
        # q_embed.reshape(1, -1): 将查询向量重塑为二维数组(1, embedding_dim)
        # search_k: 搜索返回的最相似向量数量
        # 返回值: D_nodes(距离), I_nodes(索引)
        _, I_nodes = self.faiss_retriever.node_index.search(
            q_embed.reshape(1, -1), search_k
        )

        # 将FAISS索引转换为实际的节点ID并返回
        return [
            # 通过node_map将FAISS索引转换为节点ID
            self.faiss_retriever.node_map[str(idx)]
            # 遍历返回的索引列表中的每个索引
            for idx in I_nodes[0]
            # 过滤掉无效索引(-1表示无效)并确保索引在node_map中存在
            if idx != -1 and str(idx) in self.faiss_retriever.node_map
        ]

    def _execute_faiss_relation_search(self, q_embed) -> List[str]:
        """执行FAISS关系搜索并返回对应的关系名称列表"""

        # 使用FAISS关系索引进行相似性搜索
        # q_embed.reshape(1, -1): 将查询向量重塑为二维数组(1, embedding_dim)
        # self.top_k: 搜索返回的最相似关系数量
        # 返回值: D_relations(距离), I_relations(索引)
        _, I_relations = self.faiss_retriever.relation_index.search(
            q_embed.reshape(1, -1), self.top_k
        )

        # 将FAISS索引转换为实际的关系名称并返回
        return [
            self.faiss_retriever.relation_map[str(idx)]
            for idx in I_relations[0]
            if idx != -1 and str(idx) in self.faiss_retriever.relation_map
        ]

    def _get_keyword_based_nodes(self, future_keywords) -> List[str]:
        """根据关键词Future对象获取匹配的节点列表"""
        # 从Future对象中获取关键词提取的结果
        # future_keywords.result()会阻塞直到关键词提取任务完成
        keywords = future_keywords.result()

        # 使用提取到的关键词在图谱中搜索相关的节点
        # 调用_keyword_based_node_search方法执行实际的节点搜索
        return self._keyword_based_node_search(keywords)

    @lru_cache(maxsize=1000)
    def _get_cached_neighbors(self, node_id: str) -> List[str]:
        """获取节点的邻居列表并缓存结果以提高性能"""

        # 使用networkx的neighbors方法获取指定节点的所有邻居节点
        # 并将结果转换为列表格式返回
        return list(self.graph.neighbors(node_id))

    def _optimized_neighbor_expansion(self, top_nodes: List[str], question_embed: torch.Tensor) -> List[Tuple]:
        """执行优化的邻居节点扩展，获取顶级节点的邻居关系三元组"""

        # 初始化存储所有邻居节点的集合
        all_neighbors = set()
        # 初始化边查询集合，用于存储需要查询的节点对
        edge_queries = set()

        # 遍历所有顶级相关节点
        for node in top_nodes:
            # 使用缓存获取当前节点的邻居列表
            neighbors = self._get_cached_neighbors(node)

            # 将所有邻居节点添加到集合中
            all_neighbors.update(n for n in neighbors)

            # 添加正向边查询 (node -> neighbor)
            edge_queries.update((node, n) for n in all_neighbors)

            # 添加反向边查询 (neighbor -> node)
            edge_queries.update((n, node) for n in all_neighbors)

        # 初始化存储三元组的列表
        triples = []
        # 遍历所有边查询
        for u, v in edge_queries:
            # 获取图中指定节点对的边数据
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                # 获取边的关系属性，默认为空字符串
                relation = list(edge_data.values())[0].get('relation', '')

                # 如果关系不为空，则添加到三元组列表中
                if relation:
                    triples.append((u, relation, v))
        # 返回收集到的三元组列表
        return triples

    def _get_relation_matched_triples(self, top_nodes: List[str], relations: List[str]) -> List[Tuple]:
        """根据顶级节点和关系列表获取匹配的三元组

            Args:
                top_nodes: 顶级相关节点ID列表
                relations: 相关关系类型列表

            Returns:
                匹配的三元组列表，格式为(头节点, 关系, 尾节点)
            """
        top_node_set = set(top_nodes)
        relation_set = set(relations)

        # 遍历图中所有边，查找匹配的三元组
        return [
            # 构造三元组(头节点, 关系, 尾节点)
            (u, data.get('relation'), v)
            # 遍历图中所有边，u为头节点，v为尾节点，data为边的属性数据
            for u, v, data in self.graph.edges(data=True)
            # 筛选条件：边的关系在关系集合中，且头节点或尾节点在顶级节点集合中
            if data.get('relation') in relation_set and
               (u in top_node_set or v in top_node_set)
        ]

    def _triple_only_retrieval(self, question_embed: torch.Tensor) -> Dict:
        """
        第二条检索路径：直接从FAISS中检索与问题最相关的三元组

        Args:
            question_embed: 编码后的问题张量，用于向量相似度计算

        Returns:
            包含以下内容的字典：
            - scored_triples: 带评分的三元组列表，格式为(头节点, 关系, 尾节点, 分数)元组
        """
        try:
            # 调用FAISS检索器执行双路径检索，获取与问题嵌入最相似的三元组
            # question_embed: 查询问题的嵌入向量
            # top_k: 返回的三元组数量，使用类中设置的top_k参数
            logger.info("开始双路径检索")
            faiss_results = self.faiss_retriever.dual_path_retrieval(
                question_embed,
                top_k=self.top_k
            )
            logger.info("双路径检索完成")

            # 从FAISS检索结果中提取带评分的三元组列表
            # 如果没有找到相关三元组，则返回空列表
            scored_triples = faiss_results.get("scored_triples", [])

            return {
                "scored_triples": scored_triples
            }
        except Exception as e:
            logger.error(f"Error in _triple_only_retrieval: {str(e)}")
            return {
                "scored_triples": []
            }

    def _get_node_text(self, node: str) -> str:
        """
        通过组合节点的名称和描述来获取节点的文本表示
        经过优化，使用预计算缓存以获得更好的性能

        Args:
            node: 图谱中的节点ID

        Returns:
            节点的组合文本表示
        """
        # 如果可用，使用预计算的缓存
        if hasattr(self, '_node_text_cache') and node in self._node_text_cache:
            return self._node_text_cache[node]

        try:
            # 检查节点是否存在于图谱中
            if node not in self.graph.nodes:
                return f"[Unknown Node: {node}]"

            # 获取节点数据
            data = self.graph.nodes[node]
            # 从节点属性中提取名称和描述信息
            if 'properties' in data and isinstance(data['properties'], dict):
                name = data['properties'].get('name', '')
                description = data['properties'].get('description', '')
            else:
                name = data.get('name', '')
                description = data.get('description', '')

            # 处理名称为列表的情况
            if isinstance(name, list):
                name = ", ".join(str(item) for item in name)
            elif not isinstance(name, str):
                name = str(name)

            # 处理描述为列表的情况
            if isinstance(description, list):
                description = ", ".join(str(item) for item in description)
            elif not isinstance(description, str):
                description = str(description)

            # 组合名称和描述，去除首尾空格
            result = f"{name} {description}".strip()

            # 如果结果为空或只包含空格，则返回默认节点表示
            if not result or result.isspace():
                result = f"[Node: {node}]"

            # 如果有缓存，则将结果存入缓存
            if hasattr(self, '_node_text_cache'):
                self._node_text_cache[node] = result

            return result

        except Exception as e:
            logger.error(f"Error getting text for node {node}: {str(e)}")
            return f"[Error Node: {node}]"

    def _get_node_properties(self, node: str) -> str:
        """
            获取节点的格式化属性信息用于显示

            Args:
                node: 图谱中的节点ID

            Returns:
                格式化的节点属性字符串表示
        """
        # 检查节点是否存在于图谱中，不存在则返回空字符串
        if node not in self.graph.nodes:
            return ""

        # 获取节点数据
        data = self.graph.nodes[node]
        # 初始化存储属性的列表
        properties = []

        # 定义需要跳过的字段集合
        SKIP_FIELDS = {'name', 'description', 'properties', 'label', 'chunk id', 'level'}

        # 遍历数据源（属性字典和节点数据本身）
        for source in [data.get('properties', {}), data]:
            if not isinstance(source, dict):
                continue
            # 遍历源中的所有键值对
            for key, value in source.items():
                # 如果键在跳过字段集合中则跳过
                if key in SKIP_FIELDS:
                    continue
                value_str = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                properties.append(f"{key}: {value_str}")

        # 如果有属性则返回格式化的字符串，否则返回空字符串
        return f"[{', '.join(properties)}]" if properties else ""

    def _extract_triple_based_info(self, triples: List[Tuple[str, str, str]]) -> List[str]:
        """
        从三元组中提取可读信息，包含节点属性

        Args:
            triples: 三元组列表，每个三元组包含(头节点, 关系, 尾节点)

        Returns:
            包含节点属性的三元组文本描述列表
        """
        triple_texts = []

        # 遍历所有三元组
        for h, r, t in triples:
            try:
                # 获取头节点和尾节点的文本表示
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)
                # 获取头节点和尾节点的属性信息
                head_props = self._get_node_properties(h)
                tail_props = self._get_node_properties(t)

                # 检查节点文本是否有效（不为空且不以'[Error'开头）
                if head_text and tail_text and not head_text.startswith('[Error') and not tail_text.startswith(
                        '[Error'):
                    # 构造包含属性信息的三元组文本表示
                    triple_text = f"({head_text} {head_props}, {r}, {tail_text} {tail_props})"
                    triple_texts.append(triple_text)
                else:
                    # 记录跳过的无效三元组信息
                    logger.info(f"Skipping triple with invalid nodes: ({h}, {r}, {t})")
            except Exception as e:
                logger.error(f"Warning: Error processing triple ({h}, {r}, {t}): {str(e)}")
                continue

        # 返回处理后的三元组文本列表
        return triple_texts

    def _extract_scored_triple_info(self, scored_triples: List[Tuple[str, str, str, float]]) -> List[str]:
        """
            从带评分的三元组中提取可读信息，包含节点属性和评分

            Args:
                scored_triples: 带评分的三元组列表，每个元素为(头节点, 尾节点, 关系, 分数)的元组

            Returns:
                包含节点属性和评分的三元组文本描述列表
        """
        triples = []

        # 遍历所有带评分的三元组，同时获取索引和元组内容
        for i, (h, r, t, score) in enumerate(scored_triples):
            try:
                # 获取头节点和尾节点的文本表示
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)
                # 获取头节点和尾节点的属性信息
                head_props = self._get_node_properties(h)
                tail_props = self._get_node_properties(t)

                # 检查节点文本是否有效（不为空且不以'[Error'开头）
                if head_text and tail_text and not head_text.startswith('[Error') and not tail_text.startswith(
                        '[Error'):
                    triple_text = f"({head_text} {head_props}, {r}, {tail_text} {tail_props}) [score: {score:.3f}]"
                    triples.append(triple_text)
                    # 构造包含属性信息和评分的三元组文本表示
                else:
                    logger.info(f"Skipping scored triple with invalid nodes: ({h}, {r}, {t})")
            except Exception as e:
                logger.error(f"Warning: Error processing scored triple ({h}, {r}, {t}): {str(e)}")
                continue

        return triples

    def _parse_triple_string(self, triple: str) -> tuple[str, str, str, str]:
        """
        解析三元组字符串并提取头实体、关系、尾实体和评分部分

        Args:
            triple: 格式为"(head, relation, tail) [score: X]"的三元组字符串

        Returns:
            包含(头实体名称, 关系, 尾实体, 评分部分)的元组
        """
        if not (triple.startswith('(') and triple.endswith(')')):
            return None, None, None, ""

        # 移除首尾的括号，获取内容部分
        content = triple[1:-1]  # Remove parentheses

        # 初始化评分部分为空字符串
        score_part = ""
        # 如果内容中包含评分信息，则提取评分部分
        if ' [score:' in content:
            content, score_suffix = content.split(' [score:', 1)
            score_part = f" [score:{score_suffix}"

        # 按逗号分割内容，同时考虑方括号的嵌套情况
        parts = self._split_respecting_brackets(content)

        # 如果分割后的部分少于3个，说明格式不正确，返回空值
        if len(parts) < 3:
            return None, None, None, ""

        # 提取头实体、关系和尾实体部分
        head = parts[0].strip()
        relation = parts[1].strip()
        tail = parts[2].strip()

        # 从头实体中提取实体名称（去除属性部分）
        head_name = head.split(' [')[0] if ' [' in head else head

        return head_name, relation, tail, score_part

    def _split_respecting_brackets(self, content: str) -> List[str]:
        """
        按逗号分割内容，同时考虑方括号嵌套，确保不会在属性部分错误分割

        Args:
            content: 需要分割的字符串内容

        Returns:
            分割后的字符串列表
        """
        parts = []
        current_part = ""
        bracket_count = 0
        comma_count = 0

        for i, char in enumerate(content):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == ',' and bracket_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                comma_count += 1
                if comma_count == 2:  # After finding 2 commas, rest is tail
                    remaining = content[i + 1:].strip()
                    if remaining:
                        parts.append(remaining)
                    break
                continue
            current_part += char

        # Add final part if we haven't reached 3 parts yet
        if len(parts) < 3 and current_part.strip():
            parts.append(current_part.strip())

        return parts

    def _build_merged_triple(self, entity_name: str, relation: str, values: List[str]) -> str:
        """
        从实体、关系和值列表构建合并的三元组字符串

        Args:
            entity_name: 实体名称
            relation: 关系类型
            values: 值列表

        Returns:
            格式化的三元组字符串
        """
        # 如果只有一个值，则构建简单的三元组格式
        if len(values) == 1:
            return f"({entity_name}, {relation}, {values[0]})"
        else:
            # 如果有多个值，则将它们合并为列表格式
            merged_values = f"[{', '.join(values)}]"
            return f"({entity_name}, {relation}, {merged_values})"

    def _merge_entity_attributes(self, triples: List[str]) -> List[str]:
        """
        合并同一实体的多个属性为单个列表，通过辅助方法优化可读性和性能

        Args:
            triples: 包含属性的三元组字符串列表

        Returns:
            合并属性后的三元组列表
        """
        start_time = time.time()

        # Use defaultdict for cleaner nested dictionary handling
        from collections import defaultdict
        # 创建嵌套的defaultdict，结构为{实体名: {关系: [值列表]}}
        entity_attributes = defaultdict(lambda: defaultdict(list))

        # 遍历所有三元组字符串
        for triple in triples:
            try:
                # 解析三元组字符串，提取实体名、关系、值和评分部分
                head_name, relation, tail, score_part = self._parse_triple_string(triple)

                if head_name and relation and tail is not None:
                    # 将值（包括评分部分）添加到对应实体和关系的列表中
                    entity_attributes[head_name][relation].append(tail + score_part)

            except Exception as e:
                logger.error(f"Error processing triple {triple}: {str(e)}")
                continue

        # 使用辅助方法构建合并后的三元组
        merged_triples = [
            # 调用_build_merged_triple方法构建格式化的三元组字符串
            self._build_merged_triple(entity_name, relation, values)
            # 遍历所有实体
            for entity_name, relations in entity_attributes.items()
            # 遍历实体的所有关系
            for relation, values in relations.items()
        ]

        elapsed = time.time() - start_time
        logger.info(f"[StepTiming] step=_merge_entity_attributes time={elapsed:.4f}")
        return merged_triples

    def _process_chunk_results(self, chunk_results: Dict, question_embed: torch.Tensor, top_k: int) -> Tuple[
        List[str], set]:
        """
        处理文本块检索结果，返回格式化的结果和文本块ID集合

        Args:
            chunk_results: 原始文本块检索结果字典
            question_embed: 问题嵌入向量，用于重新排序
            top_k: 返回结果的数量限制

        Returns:
            元组(格式化结果列表, 文本块ID集合)
        """
        if not chunk_results:
            return [], set()

        # 根据与问题的相关性重新排序文本块结果
        reranked_results = self._rerank_chunks_by_relevance(chunk_results, question_embed, top_k)
        # 从重新排序的结果中提取文本块ID、评分和内容
        chunk_ids = reranked_results.get('chunk_ids', [])
        chunk_scores = reranked_results.get('scores', [])
        chunk_contents = reranked_results.get('chunk_contents', [])

        formatted_results = []
        chunk_id_set = set()

        # 遍历所有文本块信息，生成格式化的结果
        for chunk_id, score, content in zip(chunk_ids, chunk_scores, chunk_contents):
            # 创建格式化的文本块结果字符串，包含ID、内容摘要和相关性评分
            formatted_result = f"[Chunk {chunk_id}] {content[:200]}... [score: {score:.3f}]"
            formatted_results.append(formatted_result)
            # 将文本块ID添加到集合中
            chunk_id_set.add(chunk_id)

        return formatted_results, chunk_id_set

    def _collect_all_scored_triples(self, results: Dict, question_embed: torch.Tensor) -> List[
        Tuple[str, str, str, float]]:
        """
        收集并合并来自两个路径的所有带评分三元组

        Args:
            results: 检索结果字典，包含两个路径的结果
            question_embed: 问题嵌入向量，用于重新排序三元组

        Returns:
            按评分排序的带评分三元组列表，格式为(头节点, 关系, 尾节点, 评分)
        """
        all_scored_triples = []

        # 添加路径2的带评分三元组（如果存在）
        path2_results = results.get('path2_results', {})
        if path2_results:
            path2_scored = path2_results.get('scored_triples', [])
            if path2_scored:
                all_scored_triples.extend(path2_scored)

        # 添加路径1的重新排序三元组
        path1_triples = results['path1_results'].get('one_hop_triples', [])
        if path1_triples:
            # 使用问题嵌入向量重新计算路径1三元组的相关性评分
            path1_scored = self._rerank_triples_by_relevance(path1_triples, question_embed)
            all_scored_triples.extend(path1_scored)

        # 按评分降序排序并返回
        all_scored_triples.sort(key=lambda x: x[3], reverse=True)
        return all_scored_triples

    def _format_scored_triples(self, scored_triples: List[Tuple[str, str, str, float]]) -> List[str]:
        """
        将带评分的三元组格式化为包含节点属性的可读文本

        参数:
            scored_triples: 带评分的三元组列表，每个元素是一个四元组 (头节点, 关系, 尾节点, 评分)

        返回:
            格式化后的三元组文本列表
        """
        formatted_triples = []

        # 遍历所有带评分的三元组
        for h, r, t, score in scored_triples:
            # 获取头节点和尾节点的文本表示
            head_text = self._get_node_text(h)
            tail_text = self._get_node_text(t)

            # 检查节点文本是否有效（不为空且不以错误标记开头）
            # 如果任一节点文本无效，则跳过这个三元组
            if not head_text or not tail_text or head_text.startswith('[Error') or tail_text.startswith('[Error'):
                continue

            # 获取头节点和尾节点的属性信息
            head_props = self._get_node_properties(h)
            tail_props = self._get_node_properties(t)

            # 构造格式化的三元组文本，包含节点文本、属性和评分
            # 格式为: (头节点文本 [属性], 关系, 尾节点文本 [属性]) [score: 评分]
            triple_text = f"({head_text} {head_props}, {r}, {tail_text} {tail_props}) [score: {score:.3f}]"
            # 将格式化后的文本添加到结果列表中
            formatted_triples.append(triple_text)

        # 返回所有格式化后的三元组文本列表
        return formatted_triples

    def _extract_chunk_ids_from_triples(self, scored_triples: List[Tuple[str, str, str, float]]) -> set:
        """
        从带评分的三元组中提取节点关联的文本块ID

        参数:
            scored_triples: 带评分的三元组列表，每个元素是(头节点, 关系, 尾节点, 评分)的四元组

        返回:
            包含所有找到的文本块ID的集合
        """
        chunk_ids = set()

        # 遍历所有带评分的三元组
        for h, r, t, score in scored_triples:
            # 检查头节点是否在图中存在
            if h in self.graph.nodes:
                # 从头节点的数据中提取文本块ID
                chunk_id = self._get_node_chunk_id(self.graph.nodes[h])
                if chunk_id:
                    chunk_ids.add(str(chunk_id))

            # 检查尾节点是否在图中存在
            if t in self.graph.nodes:
                chunk_id = self._get_node_chunk_id(self.graph.nodes[t])
                if chunk_id:
                    chunk_ids.add(str(chunk_id))

        return chunk_ids

    def _get_node_chunk_id(self, node_data: dict) -> str:
        """
        从节点数据中提取文本块ID，兼容新旧两种数据结构格式

        参数:
            node_data: 节点数据字典，包含节点的各种属性信息

        返回:
            节点关联的文本块ID字符串，如果未找到则返回None
        """
        # 检查节点数据中是否存在'properties'字段，且该字段是一个字典
        # 这是新版本的数据结构格式，chunk id被嵌套在properties内部
        if isinstance(node_data.get('properties'), dict):
            return node_data['properties'].get('chunk id')
        # 如果没有properties字段或者不是字典，则使用旧版本的数据结构格式
        # 在旧格式中，'chunk id'字段直接位于节点数据的顶层
        return node_data.get('chunk id')

    def _get_matching_chunks(self, chunk_ids: set) -> List[str]:
        """
        根据给定的文本块ID集合获取对应的文本块内容

        参数:
            chunk_ids: 文本块ID的集合

        返回:
            包含对应文本块内容的列表
        """
        # 遍历所有给定的文本块ID，如果该ID在chunk2id映射中存在，则获取其对应的文本内容
        # chunk2id是一个字典，存储了文本块ID到文本内容的映射关系
        return [self.chunk2id[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunk2id]

    def process_retrieval_results(self, question: str, top_k: int = 20, involved_types: dict = None, alpha: float = 1.0,
                                  beta: float = 0.0) -> Tuple[Dict, float]:
        """
        处理检索结果，使用优化的结构和辅助方法

        参数:
            question: 查询问题
            top_k: 返回结果的数量限制，默认为20
            involved_types: 包含相关schema类型的字典，用于类型过滤

        返回:
            元组(检索结果字典, 检索时间)
        """
        start_time = time.time()

        # 根据是否提供类型信息选择不同的检索方法
        if involved_types:
            # 如果提供了类型信息，使用带类型过滤的检索方法
            logger.info("开始使用带类型过滤的检索方法")
            question_embed, results = self.retrieve_with_type_filtering(question, involved_types, alpha, beta)
        else:
            # 否则使用普通的检索方法
            question_embed, results = self.retrieve(question, alpha, beta)

        retrieval_time = time.time() - start_time
        logger.info(f"检索耗时: {retrieval_time:.4f}秒")

        # path1_triples = self._extract_triple_based_info(results['path1_results']['one_hop_triples'])

        # path2_triples = []
        # if results['path2_results'].get('scored_triples'):
        #     path2_triples = self._extract_scored_triple_info(results['path2_results']['scored_triples'])

        # Merge entity attributes for both paths
        # merged_path1 = self._merge_entity_attributes(path1_triples)
        # merged_path2 = self._merge_entity_attributes(path2_triples)
        # all_triples = merged_path1 + merged_path2

        # 处理路径1的文本块检索结果
        chunk_results = results['path1_results'].get('chunk_results')
        # 使用_process_chunk_results方法处理文本块结果，获取格式化结果和文本块ID集合
        chunk_retrieval_results, chunk_retrieval_ids = self._process_chunk_results(
            chunk_results, question_embed, top_k
        )

        # 收集所有带评分的三元组（来自两个检索路径）
        all_scored_triples = self._collect_all_scored_triples(results, question_embed)
        # 限制三元组数量为top_k
        limited_scored_triples = all_scored_triples[:top_k]

        # 格式化带评分的三元组，使其更易读
        formatted_triples = self._format_scored_triples(limited_scored_triples)
        # 从三元组中提取文本块ID
        triple_chunk_ids = self._extract_chunk_ids_from_triples(limited_scored_triples)

        # 合并来自文本块检索和三元组的文本块ID
        all_chunk_ids = chunk_retrieval_ids | triple_chunk_ids
        # 根据文本块ID获取对应的文本块内容
        matching_chunks = self._get_matching_chunks(all_chunk_ids)

        # 构造最终的检索结果字典
        retrieval_results = {
            'triples': formatted_triples,  # 格式化的三元组
            'chunk_ids': list(all_chunk_ids),  # 所有文本块ID列表
            'chunk_contents': matching_chunks,  # 文本块内容列表
            'chunk_retrieval_results': chunk_retrieval_results  # 文本块检索的格式化结果
        }
        logger.info(f"返回结果数量: {len(retrieval_results['triples'])}")

        # 返回检索结果和检索时间
        return retrieval_results, retrieval_time

    def process_subquestions_parallel(self, sub_questions: List[Dict], top_k: int = 10, involved_types: dict = None) -> \
            Tuple[Dict, float]:
        """
        并行处理多个子问题的检索任务

        参数:
            sub_questions: 子问题字典列表，每个字典包含子问题信息
            top_k: 每个子问题返回的顶部结果数量，默认为10
            involved_types: 包含相关schema类型的字典，用于类型过滤

        返回:
            元组(聚合结果字典, 总处理时间)
        """
        start_time = time.time()

        # 设置并行执行的最大工作线程数
        default_max_workers = 4
        if self.config:
            default_max_workers = self.config.retrieval.faiss.max_workers
        # 线程数不超过子问题数量和默认最大线程数的较小值
        max_workers = min(len(sub_questions), default_max_workers)
        # 使用线程池并发执行多个子问题处理任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有子问题处理任务到线程池
            future_to_subquestion = {
                executor.submit(self._process_single_subquestion, sub_q, top_k, involved_types): sub_q
                for sub_q in sub_questions
            }

            # 初始化存储聚合结果的变量
            all_triples = set()  # 所有三元组集合
            all_chunk_ids = set()  # 所有文本块ID集合
            all_chunk_contents = {}  # 所有文本块内容字典
            all_sub_question_results = []  # 所有子问题结果列表

            # 处理完成的任务结果
            for future in concurrent.futures.as_completed(future_to_subquestion):
                sub_q = future_to_subquestion[future]
                # try:
                # 获取子问题处理结果
                sub_result = future.result()

                # 使用线程锁确保线程安全地更新共享数据
                with threading.Lock():
                    # 更新所有三元组集合
                    all_triples.update(sub_result['triples'])
                    # 更新所有文本块ID集合
                    all_chunk_ids.update(sub_result['chunk_ids'])

                    # 更新所有文本块内容字典
                    for chunk_id, content in sub_result['chunk_contents'].items():
                        all_chunk_contents[chunk_id] = content

                    # 添加子问题结果到总结果列表
                    all_sub_question_results.append(sub_result['sub_result'])

                # except Exception as e:
                #     logger.error(f"Error processing sub-question: {str(e)}")
                #     with threading.Lock():
                #         all_sub_question_results.append({
                #             'sub_question': sub_q.get('sub-question', ''),
                #             'triples_count': 0,
                #             'chunk_ids_count': 0,
                #             'time_taken': 0.0
                #         })

        # 去重后的三元组列表
        dedup_triples = list(all_triples)
        # 去重后的文本块ID列表
        dedup_chunk_ids = list(all_chunk_ids)

        # 构建去重后的文本块内容字典，对于缺失内容提供默认值
        dedup_chunk_contents = {chunk_id: all_chunk_contents.get(chunk_id, f"[Missing content for chunk {chunk_id}]")
                                for chunk_id in dedup_chunk_ids}

        # 如果没有找到任何相关信息，提供默认提示信息
        if not dedup_triples and not dedup_chunk_contents:
            dedup_triples = ["No relevant information found"]
            dedup_chunk_contents = {"no_chunks": "No relevant chunks found"}

        total_time = time.time() - start_time

        # 返回聚合结果和总处理时间
        return {
            'triples': dedup_triples,  # 去重后的三元组列表
            'chunk_ids': dedup_chunk_ids,  # 去重后的文本块ID列表
            'chunk_contents': dedup_chunk_contents,  # 去重后的文本块内容字典
            'sub_question_results': all_sub_question_results  # 所有子问题的详细结果
        }, total_time

    def _process_single_subquestion(self, sub_question: Dict, top_k: int, involved_types: dict = None) -> Dict:
        """
        处理单个子问题的检索任务

        参数:
            sub_question: 包含子问题信息的字典
            top_k: 返回结果的数量限制
            involved_types: 包含相关schema类型的字典，用于类型过滤

        返回:
            包含子问题处理结果的字典
        """
        # 从子问题字典中提取子问题文本，如果没有则使用空字符串
        sub_question_text = sub_question.get('sub-question', '')
        # 调用process_retrieval_results方法处理子问题检索
        # 获取检索结果和处理时间
        retrieval_results, time_taken = self.process_retrieval_results(sub_question_text, top_k, involved_types)

        # 从检索结果中提取三元组信息，如果没有则使用空列表
        triples = retrieval_results.get('triples', []) or []
        # 从检索结果中提取文本块ID，如果没有则使用空列表
        chunk_ids = retrieval_results.get('chunk_ids', []) or []
        # 从检索结果中提取文本块内容，如果没有则使用空列表
        chunk_contents = retrieval_results.get('chunk_contents', []) or []

        # 处理文本块内容格式
        if isinstance(chunk_contents, dict):
            # 如果是字典格式，获取所有值组成的列表
            chunk_contents_list = list(chunk_contents.values())
        else:
            # 否则直接使用原内容
            chunk_contents_list = chunk_contents

        # 数据类型验证和规范化
        # 检查triples是否为列表或元组，如果不是则设为空列表并记录警告
        if not isinstance(triples, (list, tuple)):
            logger.warning(f"triples is not a list: {type(triples)}")
            triples = []
        # 检查chunk_ids是否为列表或元组，如果不是则设为空列表并记录警告
        if not isinstance(chunk_ids, (list, tuple)):
            logger.warning(f"chunk_ids is not a list: {type(chunk_ids)}")
            chunk_ids = []
        # 检查chunk_contents_list是否为列表或元组，如果不是则设为空列表并记录警告
        if not isinstance(chunk_contents_list, (list, tuple)):
            logger.warning(f"chunk_contents_list is not a list: {type(chunk_contents_list)}")
            chunk_contents_list = []

        # 构造子问题统计信息
        sub_result = {
            'sub_question': sub_question_text,  # 子问题文本
            'triples_count': len(triples),  # 三元组数量
            'chunk_ids_count': len(chunk_ids),  # 文本块ID数量
            'time_taken': time_taken  # 处理耗时
        }

        # 构造文本块内容字典，为每个文本块ID关联对应的内容
        chunk_contents_dict = {}
        for i, chunk_id in enumerate(chunk_ids):
            if i < len(chunk_contents_list):
                # 如果索引在有效范围内，使用对应的内容
                chunk_contents_dict[chunk_id] = chunk_contents_list[i]
            else:
                # 如果索引超出范围，使用默认的缺失内容提示
                chunk_contents_dict[chunk_id] = f"[Missing content for chunk {chunk_id}]"
        # 返回处理结果
        return {
            'triples': set(triples),  # 三元组集合（去重）
            'chunk_ids': set(chunk_ids),  # 文本块ID集合（去重）
            'chunk_contents': chunk_contents_dict,  # 文本块内容字典
            'sub_result': sub_result  # 子问题统计信息
        }

        # except Exception as e:
        #     logger.error(f"Error processing sub-question '{sub_question_text}': {str(e)}")
        #     return {
        #         'triples': set(),
        #         'chunk_ids': set(),
        #         'chunk_contents': {},
        #         'sub_result': {
        #             'sub_question': sub_question_text,
        #             'triples_count': 0,
        #             'chunk_ids_count': 0,
        #             'time_taken': 0.0
        #         }
        #     }

    def generate_prompt(self, question: str, context: str) -> str:
        """
        根据问题和上下文生成传递给LLM的提示(prompt)

        参数:
            question: 用户提出的问题
            context: 检索到的相关上下文信息

        返回:
            格式化的提示字符串
        """
        # 检查是否存在配置对象
        if self.config:
            # 根据数据集类型从配置中获取相应的格式化提示
            if self.dataset == 'novel':
                # 中文小说数据集使用专门的提示模板
                return self.config.get_prompt_formatted("retrieval", "novel_chs", question=question, context=context)
            elif self.dataset == 'novel_eng':
                # 英文小说数据集使用专门的提示模板
                return self.config.get_prompt_formatted("retrieval", "novel_eng", question=question, context=context)
            else:
                # 其他数据集使用通用提示模板
                return self.config.get_prompt_formatted("retrieval", "general", question=question, context=context)
        else:
            # 如果没有配置对象，则使用硬编码的提示模板
            if self.dataset == 'novel':
                # 中文小说数据集的硬编码提示模板
                prompt = f"""
                你是小说知识助手，你的任务是根据提供的小说知识库回答问题。
                1. 如果知识库中的信息不足以回答问题，请根据你的推理和知识回答。
                2. 回答要简洁明了。
                3. 对于事实性问题，提供具体的事实或人物名称。
                4. 对于时间性问题，提供具体的时间、年份或时间段。
                问题：{question}
                相关知识：{context}
                答案（简洁明了）：
                """
            elif self.dataset == 'novel_eng':
                # 英文小说数据集的硬编码提示模板
                prompt = f"""
                You are a novel knowledge assistant. Your task is to answer the question based on the provided novel knowledge context.
                1. If the knowledge is insufficient, answer the question based on your own knowledge.
                2. Be precise and concise in your answer.
                3. For factual questions, provide the specific fact or entity name
                4. For temporal questions, provide the specific date, year, or time period

                Question: {question}

                Knowledge Context:
                {context}   

                Answer (be specific and direct):
                """
            else:
                # 通用数据集的硬编码提示模板
                # prompt = f"""
                # You are an expert knowledge assistant. Your task is to answer the question based on the provided knowledge context.
                #
                # 1. Use ONLY the information from the provided knowledge context and try your best to answer the question.
                # 2. If the knowledge is insufficient, reject to answer the question.
                # 3. Be precise and concise in your answer
                # 4. For factual questions, provide the specific fact or entity name
                # 5. For temporal questions, provide the specific date, year, or time period
                #
                # Question: {question}
                #
                # Knowledge Context:
                # {context}
                #
                # Answer (be specific and direct):
                # """
                prompt = f"""
                你是知识问答助手，你的任务是根据提供的知识上下文回答问题。
                        
                        1. 仅使用提供的知识上下文中的信息来回答问题
                        2. 如果知识不足，请说明无法回答
                        3. 回答应精确简洁
                        4. 对于事实性问题，提供具体的事实或实体名称
                        5. 对于时间性问题，提供具体的日期、年份或时间段
                        
                        问题：{question}
                        
                        知识上下文：
                        {context}
                        
                        答案（精确直接）：
                """
            # 返回生成的提示
            return prompt

    def generate_answer(self, prompt: str) -> str:
        """
            调用LLM API生成基于给定提示的答案

            参数:
                prompt: 传递给LLM的提示字符串

            返回:
                LLM生成的答案字符串
            """
        # 调用LLM客户端的API方法生成答案
        # self.llm_client是在__init__方法中初始化的call_llm_api.LLMCompletionCall实例
        answer = self.llm_client.call_api(prompt)
        logger.info("检索到的上下文:")
        logger.info(prompt)
        logger.info(f"Answer: {answer}")
        # 返回LLM生成的答案
        return answer

    def _extract_chunk_ids_from_nodes(self, nodes: List[str]) -> set:
        """
        从节点ID列表中提取文本块ID

        参数:
            nodes: 节点ID列表

        返回:
            在这些节点中找到的文本块ID集合
        """
        chunk_ids = set()

        # 遍历所有给定的节点
        for node in nodes:
            try:
                # 检查节点是否存在于图谱中
                if node in self.graph.nodes:
                    # 获取节点数据
                    data = self.graph.nodes[node]

                    # 从节点数据中提取文本块ID，兼容新旧两种数据结构格式
                    # 新格式：chunk id嵌套在properties字典中
                    # 旧格式：chunk id直接在节点数据顶层
                    chunk_id = (
                        data.get('properties', {}).get('chunk id')
                        if isinstance(data.get('properties'), dict)
                        else data.get('chunk id')
                    )
                    # 如果找到了文本块ID，则添加到集合中
                    if chunk_id:
                        chunk_ids.add(str(chunk_id))
                    else:
                        logger.warning(f"Debug: No chunk ID found for node {node}")
                else:
                    logger.warning(f"Debug: Node {node} not found in graph")
            except Exception as e:
                logger.error(f"Debug: Error processing node {node}: {str(e)}")
                continue

        # 返回收集到的所有文本块ID集合
        return chunk_ids

    def _extract_chunk_ids_from_triple_nodes(self, scored_triples: List[Tuple[str, str, str, float]]) -> set:
        """
        从带评分的三元组中提取文本块ID

        参数:
            scored_triples: 带评分的三元组列表，每个元素是(头节点, 关系, 尾节点, 评分)的四元组

        返回:
            在这些三元组中找到的文本块ID集合
        """
        chunk_ids = set()

        # 遍历所有带评分的三元组
        for h, r, t, score in scored_triples:
            try:
                # 检查头节点是否在图谱中存在
                if h in self.graph.nodes:
                    # 获取头节点数据
                    data = self.graph.nodes[h]
                    chunk_id = (
                        data.get('properties', {}).get('chunk id')
                        if isinstance(data.get('properties'), dict)
                        else data.get('chunk id')
                    )
                    if chunk_id:
                        chunk_ids.add(str(chunk_id))
                # 检查尾节点是否在图谱中存在
                if t in self.graph.nodes:
                    data = self.graph.nodes[t]
                    chunk_id = (
                        data.get('properties', {}).get('chunk id')
                        if isinstance(data.get('properties'), dict)
                        else data.get('chunk id')
                    )
                    if chunk_id:
                        chunk_ids.add(str(chunk_id))
            except Exception as e:
                continue

        return chunk_ids

    def _enhance_query_with_entities(self, question: str) -> str:
        """
        通过使用spaCy NER和依存句法分析提取实体和关系来增强查询。
        使用缓存进行性能优化。

        参数:
            question: 原始问题

        返回:
            包含实体信息的增强查询
        """

        try:
            # 使用spaCy处理问题文本，生成文档对象
            doc = self.nlp(question)

            # 提取命名实体
            entities = []
            for ent in doc.ents:  # 遍历文档中的所有命名实体
                entities.append(ent.text)  # 将实体文本添加到列表中

            # 提取关键词短语
            key_phrases = []
            for token in doc:  # 遍历文档中的所有词元
                # 选择名词、专有名词、动词、形容词，且不是停用词的词元
                if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                    key_phrases.append(token.text)  # 将词元文本添加到列表中
                    if len(key_phrases) >= 5:  # 限制最多提取5个关键词
                        break

            # 构建增强查询的各个部分
            enhanced_parts = [question]  # 首先是原始问题
            # 如果提取到了实体
            if entities:
                # 添加实体信息
                enhanced_parts.append(f"Entities: {', '.join(entities)}")
            # 如果提取到了关键词
            if key_phrases:
                # 添加关键词信息
                enhanced_parts.append(f"Key terms: {', '.join(key_phrases)}")

            # 将所有部分用空格连接成增强后的查询
            enhanced_query = " ".join(enhanced_parts)

            # 返回增强后的查询
            return enhanced_query

        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return question

    def _calculate_entity_similarity(self, query_embed: torch.Tensor, node: str) -> float:
        """
        计算查询和节点之间的实体级相似度。
        使用缓存进行性能优化。

        参数:
            query_embed: 查询嵌入向量
            node: 节点ID

        返回:
            相似度分数
    """

        try:
            if node not in self.graph.nodes:
                return 0.0

            # 获取节点的文本表示
            node_text = self._get_node_text(node)
            if not node_text or node_text.startswith('[Error') or node_text.startswith('[Unknown'):
                return 0.0

            # 检查节点嵌入是否已在缓存中
            if node in self.node_embedding_cache:
                node_embed = self.node_embedding_cache[node]
                node_embed = node_embed.to(self.device)
            else:
                # 如果不在缓存中，则计算节点文本的嵌入
                node_embed = torch.tensor(self.qa_encoder.encode(node_text)).float().to(self.device)
                # 将计算得到的嵌入存储到缓存中
                self.node_embedding_cache[node] = node_embed

            # 计算查询嵌入和节点嵌入之间的余弦相似度
            similarity = F.cosine_similarity(query_embed, node_embed, dim=0).item()
            similarity = max(0.0, similarity)

            # 返回相似度分数
            return similarity

        except Exception as e:
            logger.error(f"Error calculating entity similarity for node {node}: {str(e)}")
            return 0.0

    def _batch_calculate_entity_similarities(self, query_embed: torch.Tensor, nodes: List[str]) -> Dict[str, float]:
        """
            批量计算查询嵌入与节点嵌入之间的相似度

            参数:
                query_embed: 查询嵌入张量
                nodes: 节点ID列表

            返回:
                字典，键为节点ID，值为对应的相似度分数
            """
        # 初始化相似度字典和相关变量
        similarities = {}  # 存储节点ID到相似度的映射
        node_embeddings = []  # 存储节点嵌入的列表
        valid_nodes = []  # 存储有缓存嵌入的有效节点ID列表

        # 使用节点嵌入缓存锁确保线程安全
        with self.cache_locks['node_embedding']:
            # 遍历所有节点，检查是否有缓存的嵌入
            for node in nodes:
                if node in self.node_embedding_cache:
                    # 如果节点嵌入在缓存中，添加到列表中
                    node_embeddings.append(self.node_embedding_cache[node])
                    valid_nodes.append(node)
        # 如果有缓存的节点嵌入
        if node_embeddings:
            try:
                # 将节点嵌入列表堆叠成张量
                node_embeddings_tensor = torch.stack(node_embeddings)

                # 批量计算余弦相似度
                batch_similarities = F.cosine_similarity(
                    query_embed.unsqueeze(0),  # 将查询嵌入增加一个维度以匹配批量计算
                    node_embeddings_tensor,  # 节点嵌入张量
                    dim=1  # 计算维度
                )

                # 遍历有效节点，将相似度结果存储到字典中
                for i, node in enumerate(valid_nodes):
                    similarity = max(0.0, batch_similarities[i].item())
                    similarities[node] = similarity

            except Exception as e:
                # 如果批量计算失败，回退到逐个计算
                for node in valid_nodes:
                    try:
                        # 调用单个相似度计算方法
                        similarity = self._calculate_entity_similarity(query_embed, node)
                        similarities[node] = similarity
                    except Exception as e2:
                        logger.error(f"Error calculating similarity for node {node}: {str(e2)}")
                        continue
        else:
            # 如果没有缓存的节点嵌入，逐个计算相似度
            for node in nodes:
                try:
                    similarity = self._calculate_entity_similarity(query_embed, node)
                    similarities[node] = similarity
                except Exception as e:
                    logger.error(f"Error calculating similarity for node {node}: {str(e)}")
                    continue

        return similarities

    def _smart_neighbor_expansion(self, center_node: str, query_embed: torch.Tensor, max_neighbors: int = 5) -> List[
        str]:
        """
        优化的智能邻居扩展，使用批量相似度计算

        参数:
            center_node: 中心节点ID
            query_embed: 查询嵌入张量
            max_neighbors: 最大邻居数量，默认为5

        返回:
            排序后的邻居节点ID列表
        """
        # 检查中心节点是否存在于图谱中，不存在则返回空列表
        if center_node not in self.graph.nodes:
            return []

        # 获取中心节点的所有邻居节点
        neighbors = list(self.graph.neighbors(center_node))
        if not neighbors:
            return []

        # 过滤出在图谱中存在的有效邻居节点
        valid_neighbors = [n for n in neighbors if n in self.graph.nodes]
        if not valid_neighbors:
            return []

        # 使用批量计算方法计算所有有效邻居节点与查询的相似度
        neighbor_similarities = self._batch_calculate_entity_similarities(query_embed, valid_neighbors)

        # 根据相似度对邻居节点进行降序排序
        sorted_neighbors = sorted(
            neighbor_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 返回前max_neighbors个相似度大于0.1的邻居节点
        return [node for node, score in sorted_neighbors[:max_neighbors] if score > 0.1]

    def _rerank_triples_by_relevance(self, triples: List[Tuple[str, str, str]], question_embed: torch.Tensor) -> List[
        Tuple[str, str, str, float]]:
        """
    优化的三元组重排序，使用批量编码和增强缓存

    参数:
        triples: 三元组列表，每个三元组包含(头节点, 关系, 尾节点)
        question_embed: 查询嵌入张量

    返回:
        带评分的三元组列表，每个元素是(头节点, 关系, 尾节点, 分数)的元组
    """
        start_time = time.time()
        if not triples:
            return []

        # 初始化相关变量
        scored_triples = []  # 存储带评分的三元组
        triple_texts = []  # 存储三元组的文本表示
        valid_triples = []  # 存储有效的三元组

        # 遍历所有三元组
        for h, r, t in triples:
            try:
                # 获取头节点和尾节点的文本表示
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)

                # 检查节点文本是否有效
                if not head_text or not tail_text or head_text.startswith('[Error') or tail_text.startswith('[Error'):
                    continue

                # 构造三元组的文本表示
                triple_text = f"{head_text} {r} {tail_text}"
                triple_texts.append(triple_text)
                valid_triples.append((h, r, t))

            except Exception as e:
                logger.error(f"Error processing triple ({h}, {r}, {t}): {str(e)}")
                continue

        if not valid_triples:
            return []

        try:
            # 批量编码三元组文本
            encode_start = time.time()
            triple_embeddings = self.qa_encoder.encode(triple_texts, convert_to_tensor=True).to(self.device)
            encode_elapsed = time.time() - encode_start
            logger.info(f"[StepTiming] 步骤：批量编码三元组文本 耗时={encode_elapsed:.4f}秒")

            # 批量计算相似度
            sim_calc_start = time.time()
            similarities = F.cosine_similarity(
                question_embed.unsqueeze(0),
                triple_embeddings,
                dim=1
            )
            sim_calc_elapsed = time.time() - sim_calc_start
            logger.info(f"[StepTiming] 步骤：批量计算相似度 耗时={sim_calc_elapsed:.4f}秒")

            # 为每个有效三元组计算最终评分
            for i, (h, r, t) in enumerate(valid_triples):
                similarity = similarities[i].item()

                # 关系加权：某些关系类型给予额外加分
                relation_bonus = 0.0
                if r.lower() in ['is', 'was', 'has', 'had', 'contains', 'located', 'born', 'died']:
                    relation_bonus = 0.1

                # 计算最终评分（相似度+关系加权）
                final_score = max(0.0, similarity + relation_bonus)

                if final_score > 0.05:
                    scored_triples.append((h, r, t, final_score))

        except Exception as e:
            logger.error(f"Error in batch triple encoding: {str(e)}")
            # 出错时回退到逐个处理的方法
            return self._rerank_triples_individual(triples, question_embed)

        # 按评分降序排序
        scored_triples.sort(key=lambda x: x[3], reverse=True)
        elapsed = time.time() - start_time
        logger.info(f"[StepTiming] 步骤：按评分降序排序三元组 耗时={elapsed:.4f}秒")
        return scored_triples

    def _rerank_triples_individual(self, triples: List[Tuple[str, str, str]], question_embed: torch.Tensor) -> List[
        Tuple[str, str, str, float]]:
        """
    当批量处理失败时，回退到单独处理每个三元组的方法

    参数:
        triples: 三元组列表，每个三元组包含(头节点, 关系, 尾节点)
        question_embed: 查询嵌入张量

    返回:
        带评分的三元组列表，每个元素是(头节点, 关系, 尾节点, 分数)的元组
    """
        scored_triples = []

        # 遍历所有三元组
        for h, r, t in triples:
            try:
                # 获取头节点和尾节点的文本表示
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)

                if not head_text or not tail_text or head_text.startswith('[Error') or tail_text.startswith('[Error'):
                    continue

                # 构造三元组的文本表示
                triple_text = f"{head_text} {r} {tail_text}"
                # 对三元组文本进行编码并转换为张量
                triple_embed = torch.tensor(self.qa_encoder.encode(triple_text)).float().to(self.device)
                # 计算查询嵌入和三元组嵌入之间的余弦相似度
                similarity = F.cosine_similarity(question_embed, triple_embed, dim=0).item()

                # 关系加权：某些关系类型给予额外加分
                relation_bonus = 0.0
                if r.lower() in ['is', 'was', 'has', 'had', 'contains', 'located', 'born', 'died']:
                    relation_bonus = 0.1

                # 计算最终评分（相似度+关系加权）
                final_score = max(0.0, similarity + relation_bonus)

                if final_score > 0.05:
                    scored_triples.append((h, r, t, final_score))

            except Exception as e:
                logger.error(f"Error reranking triple ({h}, {r}, {t}): {str(e)}")
                continue

        # 按评分降序排序
        scored_triples.sort(key=lambda x: x[3], reverse=True)
        return scored_triples

    def _extract_query_keywords(self, question: str) -> List[str]:
        """
        使用spaCy的命名实体识别(NER)和词性标注(POS tagging)自动从问题中提取关键词。
        针对单查询场景进行了优化。

        参数:
            question: 输入的问题文本

        返回:
            自动发现的关键词列表
        """
        try:
            # 使用spaCy处理问题文本，转换为小写以统一处理
            doc = self.nlp(question.lower())
            keywords = []

            # 基于词性标注提取关键词
            for token in doc:
                # 过滤条件：非停用词且长度大于2
                if (not token.is_stop and len(token.text) > 2):
                    # 如果是命名实体，添加到关键词列表
                    if token.ent_type_:
                        keywords.append(token.text.lower())
                    # 如果是名词、专有名词或形容词，添加到关键词列表
                    elif token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                        keywords.append(token.text.lower())
                    # 如果是动词，也添加到关键词列表
                    elif token.pos_ == 'VERB':
                        keywords.append(token.text.lower())
            # 基于命名实体识别提取关键词
            for ent in doc.ents:
                # 过滤条件：长度大于2
                if len(ent.text) > 2:
                    keywords.append(ent.text.lower())

            # 去重并转换为列表
            unique_keywords = list(set(keywords))
            return unique_keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def _keyword_based_node_search(self, keywords: List[str]) -> List[str]:
        """
        使用文本索引和提前终止优化的基于关键词的节点搜索。
        针对单查询场景进行了优化。
        """
        if not keywords:
            return []

        # 获取是否使用精确匹配的配置，默认为True
        use_exact_matching = getattr(self, 'use_exact_keyword_matching', True)

        # 如果启用精确匹配
        if use_exact_matching:
            # 检查节点文本索引是否存在，如果不存在则记录警告并返回空列表
            if not hasattr(self, '_node_text_index') or self._node_text_index is None:
                logger.warning("Node text index not found. This should be built during initialization.")
                return []

            # 创建存储相关节点的集合
            relevant_nodes = set()
            # 设置每个关键词最多返回的节点数
            max_nodes_per_keyword = 50

            # 遍历所有关键词
            for keyword in keywords:
                # 如果关键词在节点文本索引中存在
                if keyword in self._node_text_index:
                    # 获取与该关键词相关的节点列表
                    keyword_nodes = self._node_text_index[keyword]
                    if len(keyword_nodes) > max_nodes_per_keyword:
                        keyword_nodes = set(list(keyword_nodes)[:max_nodes_per_keyword])
                    relevant_nodes.update(keyword_nodes)
                else:
                    continue

                if len(relevant_nodes) > 200:
                    break
            return list(relevant_nodes)
        else:
            # 如果不使用精确匹配，则调用原始的关键词搜索方法
            result = self._keyword_based_node_search_original(keywords)
            return result

    def _keyword_based_node_search_original(self, keywords: List[str]) -> List[str]:
        """
        基于关键词的原始节点搜索方法，使用子字符串匹配

        Args:
            keywords: 关键词列表

        Returns:
            包含匹配节点ID的列表
        """
        # 初始化存储相关节点的列表
        relevant_nodes = []

        # 遍历图谱中的所有节点
        for node in self.graph.nodes():
            try:
                # 获取节点的文本表示并转换为小写以便进行不区分大小写的匹配
                node_text = self._get_node_text(node).lower()

                # 遍历所有关键词
                for keyword in keywords:
                    # 检查关键词是否在节点文本中作为子字符串存在
                    if keyword in node_text:
                        # 如果找到匹配，将节点添加到相关节点列表中
                        relevant_nodes.append(node)
                        break

            except Exception as e:
                continue

        return relevant_nodes

    def _build_node_text_index(self):
        """
        构建节点文本的倒排索引以加速关键词搜索。
        优化为使用预计算的节点文本和持久化缓存。
        """
        # 首先尝试从磁盘加载已缓存的节点文本索引
        # 如果成功加载，则直接返回，无需重新构建
        if self._load_node_text_index():
            logger.info("Loaded node text index from cache")
            return

        start_time = time.time()
        logger.info("正在为关键词搜索构建优化的节点文本索引...")
        # 初始化节点文本索引字典
        self._node_text_index = {}

        # 判断是否已经有预计算的节点文本缓存
        if hasattr(self, '_node_text_cache') and self._node_text_cache:
            node_texts = self._node_text_cache
        else:
            # 如果没有缓存，则需要重新获取所有节点的文本
            node_texts = {}
            for node in self.graph.nodes():
                node_texts[node] = self._get_node_text(node)

        # 获取总节点数和已处理节点计数器
        total_nodes = len(node_texts)
        processed_nodes = 0

        # 遍历所有节点及其文本
        for node, node_text in node_texts.items():
            try:
                # 将节点文本转换为小写以便进行不区分大小写的匹配
                node_text_lower = node_text.lower()
                # 将文本拆分为单词并去重
                words = set(node_text_lower.split())

                # 遍历所有单词
                for word in words:
                    if len(word) > 2:
                        # 如果单词不在索引中，则创建一个新的集合
                        if word not in self._node_text_index:
                            self._node_text_index[word] = set()
                        # 将节点添加到该单词的索引集合中
                        self._node_text_index[word].add(node)

                # 更新已处理节点计数器
                processed_nodes += 1
                if processed_nodes % 1000 == 0:
                    logger.info(f"Indexed {processed_nodes}/{total_nodes} nodes")

            except Exception as e:
                logger.error(f"Error indexing node {node}: {str(e)}")
                continue

        end_time = time.time()
        logger.info(f"构建节点文本索引耗时: {end_time - start_time} 秒")

        # 尝试将构建好的节点文本索引保存到磁盘缓存
        self._save_node_text_index()

    def _save_node_text_index(self):
        """将节点文本索引保存到磁盘缓存"""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_index.pkl"
        try:
            # 检查节点文本索引是否存在，如果不存在则记录警告并返回False
            if not self._node_text_index:
                logger.warning("No node text index to save!")
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # 创建可序列化的索引字典
            # 因为_pickle_无法直接序列化set类型，需要将set转换为list
            serializable_index = {}
            for word, nodes in self._node_text_index.items():
                # 将每个单词对应的节点集合转换为列表
                serializable_index[word] = list(nodes)

            # 以二进制写入模式打开文件，使用pickle序列化保存索引
            with open(cache_path, 'wb') as f:
                pickle.dump(serializable_index, f)

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"已保存包含 {len(serializable_index)} 个词的节点文本索引到 {cache_path} (大小: {file_size} 字节)")
            return True

        except Exception as e:
            logger.error(f"Error saving node text index: {e}")
            return False

    def _load_node_text_index(self):
        """从磁盘缓存加载节点文本索引"""
        # 构建缓存文件路径
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_index.pkl"
        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:
                    logger.warning(f"Cache file too small ({file_size} bytes), likely empty or corrupted")
                    return False

                # 以二进制读取模式打开文件，使用pickle反序列化加载索引
                with open(cache_path, 'rb') as f:
                    serializable_index = pickle.load(f)

                # 检查加载的索引是否为空
                if not serializable_index:
                    logger.warning("Loaded index is empty")
                    return False

                # 初始化节点文本索引字典
                self._node_text_index = {}
                # 将序列化索引中的列表转换回集合类型
                # 因为保存时将set转换为了list，现在需要转回set
                for word, nodes in serializable_index.items():
                    self._node_text_index[word] = set(nodes)

                # 检查加载的索引与当前图是否一致
                if not self._check_text_index_consistency():
                    logger.info("文本索引与当前图不一致，将重新构建")
                    return False

                logger.info(
                    f"从 {cache_path} 加载了包含 {len(self._node_text_index)} 个词汇的节点文本索引 (文件大小: {file_size} 字节)")
                return True

            except Exception as e:
                logger.error(f"Error loading node text index: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted cache file: {cache_path}")
                except Exception as e2:
                    logger.error(f"Failed to remove corrupted cache file {cache_path}: {type(e2).__name__}: {e2}")
        else:
            logger.info(f"缓存文件不存在: {cache_path}")
        return False

    def _check_text_index_consistency(self):
        """检查加载的文本索引是否与当前图保持一致"""
        try:
            # 收集索引中所有的节点
            indexed_nodes = set()
            for nodes in self._node_text_index.values():
                indexed_nodes.update(nodes)

            # 获取当前图中所有节点的集合
            current_nodes = set(self.graph.nodes())
            # 计算图中存在但索引中缺失的节点
            missing_nodes = current_nodes - indexed_nodes
            if missing_nodes:
                logger.warning(f"文本索引缺少图中 {len(missing_nodes)} 个节点")
                return False

            # 计算索引中存在但图中不存在的节点（多余的节点）
            extra_nodes = indexed_nodes - current_nodes
            # 如果多余节点数量超过当前图节点数量的10%，记录警告并返回False
            if len(extra_nodes) > len(current_nodes) * 0.1:  # Allow 10% tolerance
                logger.warning(
                    f"Text index has too many extra nodes: {len(extra_nodes)} extra vs {len(current_nodes)} current")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking text index consistency: {e}")
            return False

    def _path_based_search(self, start_nodes: List[str], target_keywords: List[str], max_depth: int = 2) -> List[
        Tuple[str, str, str]]:
        """
        从起始节点搜索到包含目标关键词的节点的路径。

        Args:
            start_nodes: 起始节点ID列表
            target_keywords: 要搜索的关键词列表
            max_depth: 最大路径深度

        Returns:
            沿路径找到的三元组列表
        """
        # 初始化存储找到的三元组列表和已访问节点集合
        found_triples = []
        visited = set()

        def dfs_search(node: str, depth: int, path: List[str]):
            """
            深度优先搜索函数
            Args:
                node: 当前节点
                depth: 当前深度
                path: 当前路径
            """
            # 如果超过最大深度或节点已被访问，则返回
            if depth > max_depth or node in visited:
                return

            # 标记当前节点为已访问
            visited.add(node)

            try:
                # 获取当前节点的文本并转换为小写
                node_text = self._get_node_text(node).lower()
                # 检查目标关键词是否在节点文本中
                for keyword in target_keywords:
                    if keyword in node_text:
                        # 如果找到关键词，在当前路径中构建三元组
                        for i in range(len(path) - 1):
                            # 获取路径中相邻节点
                            u, v = path[i], path[i + 1]
                            # 获取边的数据
                            edge_data = self.graph.get_edge_data(u, v)
                            # 如果边存在且包含关系信息，则添加到结果中
                            if edge_data and 'relation' in edge_data:
                                relation = list(edge_data.values())[0]['relation']
                                found_triples.append((u, relation, v))
                        break
            except Exception as e:
                logger.warning(
                    f"Error during DFS path search at node {start_node if 'start_node' in locals() else ''}: {type(e).__name__}: {e}")

            # 如果未达到最大深度，继续搜索邻居节点
            if depth < max_depth:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        # 递归搜索邻居节点
                        dfs_search(neighbor, depth + 1, path + [neighbor])

        # 从每个起始节点开始进行深度优先搜索
        for start_node in start_nodes:
            dfs_search(start_node, 0, [start_node])

        # 返回找到的三元组列表
        return found_triples

    def _precompute_chunk_embeddings(self):
        """
        预计算所有文本块的嵌入向量，以支持直接的文本块检索
        """
        # 使用预计算锁确保线程安全，防止多个线程同时执行预计算
        with self.precompute_lock:
            if self.chunk_embeddings_precomputed:
                return

            logger.info("正在预计算文本块嵌入以支持直接文本块检索...")
            # 尝试从磁盘加载文本块嵌入缓存
            if self._load_chunk_embedding_cache():
                logger.info("成功从磁盘缓存加载文本块嵌入")
                self.chunk_embeddings_precomputed = True
                return

            # 检查是否有可用的文本块
            if not self.chunk2id:
                logger.info("警告: 没有可用于嵌入计算的文本块")
                return

            logger.info("从头开始计算文本块嵌入...")

            # 获取所有文本块ID和对应的文本内容
            chunk_ids = list(self.chunk2id.keys())
            chunk_texts = list(self.chunk2id.values())
            # 设置批处理大小，默认为50
            batch_size = 50
            if self.config:
                batch_size = self.config.embeddings.batch_size

            # 初始化计数器和存储变量
            total_processed = 0
            embeddings_list = []  # 存储嵌入向量的列表
            valid_chunk_ids = []  # 存储有效文本块ID的列表

            # 分批处理文本块嵌入计算
            for i in range(0, len(chunk_texts), batch_size):
                # 获取当前批次的文本和对应的文本块ID
                batch_texts = chunk_texts[i:i + batch_size]
                batch_chunk_ids = chunk_ids[i:i + batch_size]

                try:
                    # 使用qa_encoder对整个批次的文本进行编码
                    batch_embeddings = self.qa_encoder.encode(batch_texts, convert_to_tensor=True)

                    # 将编码结果存储到缓存和列表中
                    for j, chunk_id in enumerate(batch_chunk_ids):
                        self.chunk_embedding_cache[chunk_id] = batch_embeddings[j]
                        embeddings_list.append(batch_embeddings[j].cpu().numpy())
                        valid_chunk_ids.append(chunk_id)
                        total_processed += 1

                except Exception as e:
                    # 如果批处理编码失败，记录错误并回退到逐个文本块编码
                    logger.error(f"Error encoding chunk batch {i // batch_size}: {str(e)}")
                    for j, chunk_id in enumerate(batch_chunk_ids):
                        try:
                            # 对单个文本块进行编码
                            chunk_text = self.chunk2id[chunk_id]
                            embedding = torch.tensor(self.qa_encoder.encode(chunk_text)).float().to(self.device)
                            self.chunk_embedding_cache[chunk_id] = embedding
                            embeddings_list.append(embedding.cpu().numpy())
                            valid_chunk_ids.append(chunk_id)
                            total_processed += 1
                        except Exception as e2:
                            logger.error(f"Error encoding chunk {chunk_id}: {str(e2)}")
                            continue

            # 如果成功生成了嵌入向量，则构建FAISS索引
            if embeddings_list:
                try:
                    logger.info("正在为文本块嵌入构建FAISS索引...")
                    # 将嵌入向量列表转换为NumPy数组
                    embeddings_array = np.array(embeddings_list)
                    dimension = embeddings_array.shape[1]

                    # 创建FAISS索引用于快速相似性搜索，使用内积进行余弦相似度计算
                    self.chunk_faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                    self.chunk_faiss_index.add(embeddings_array.astype('float32'))

                    # 建立文本块ID与索引位置的双向映射关系
                    for i, chunk_id in enumerate(valid_chunk_ids):
                        self.chunk_id_to_index[chunk_id] = i  # 文本块ID到索引位置的映射
                        self.index_to_chunk_id[i] = chunk_id  # 索引位置到文本块ID的映射

                    logger.info(f"FAISS索引已构建，包含 {len(valid_chunk_ids)} 个文本块")
                except Exception as e:
                    logger.error(f"构建文本块的FAISS索引时出错: {str(e)}")

            # 标记文本块嵌入预计算完成
            self.chunk_embeddings_precomputed = True
            logger.info(
                f"文本块嵌入已预计算完成，共处理 {total_processed} 个文本块 (缓存大小: {len(self.chunk_embedding_cache)})")

            # 尝试将文本块嵌入缓存保存到磁盘
            self._save_chunk_embedding_cache()

    def _save_chunk_embedding_cache(self):
        """将文本块嵌入缓存保存到磁盘"""
        cache_path = f"{self.cache_dir}/{self.dataset}/chunk_embedding_cache.pt"
        try:
            if not self.chunk_embedding_cache:
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # 创建用于保存到磁盘的numpy缓存字典
            numpy_cache = {}
            # 遍历文本块嵌入缓存中的每个文本块ID和对应的嵌入向量
            for chunk_id, embed in self.chunk_embedding_cache.items():
                if embed is not None:
                    try:
                        # 将嵌入向量转换为可以在CPU上处理的numpy数组格式
                        if hasattr(embed, 'detach'):
                            numpy_cache[chunk_id] = embed.detach().cpu().numpy()
                        elif isinstance(embed, np.ndarray):
                            numpy_cache[chunk_id] = embed
                        else:
                            numpy_cache[chunk_id] = np.array(embed)
                    except Exception as e:
                        continue

            if not numpy_cache:
                return False

            try:
                # 创建用于PyTorch保存的张量缓存字典
                tensor_cache = {}
                # 将numpy数组转换回PyTorch张量
                for chunk_id, embed_array in numpy_cache.items():
                    if isinstance(embed_array, np.ndarray):
                        tensor_cache[chunk_id] = torch.from_numpy(embed_array).float()
                    else:
                        tensor_cache[chunk_id] = embed_array

                # 使用PyTorch的save方法保存缓存
                torch.save(tensor_cache, cache_path)
            except Exception as torch_error:
                # 如果PyTorch保存失败，回退到使用numpy保存
                cache_path_npz = cache_path.replace('.pt', '.npz')
                # 使用numpy的压缩格式保存
                np.savez_compressed(cache_path_npz, **numpy_cache)
                cache_path = cache_path_npz

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"Saved chunk embedding cache with {len(numpy_cache)} entries to {cache_path} (size: {file_size} bytes)")
            return True

        except Exception as e:
            return False

    def _load_chunk_embedding_cache(self):
        """从磁盘加载文本块嵌入缓存"""
        cache_path = f"{self.cache_dir}/{self.dataset}/chunk_embedding_cache.pt"
        cache_path_npz = cache_path.replace('.pt', '.npz')

        # 首先尝试加载.npyz格式的缓存文件
        if os.path.exists(cache_path_npz):
            try:
                # 使用numpy加载压缩的.npyz文件
                numpy_cache = np.load(cache_path_npz)

                if len(numpy_cache.files) == 0:
                    return False

                # 清空当前的文本块嵌入缓存
                self.chunk_embedding_cache.clear()

                # 遍历缓存中的所有文本块ID
                for chunk_id in numpy_cache.files:
                    try:
                        # 加载文本块的嵌入数组
                        embed_array = numpy_cache[chunk_id]
                        # 将numpy数组转换为PyTorch张量并移到指定设备
                        embed_tensor = torch.from_numpy(embed_array).float().to(self.device)
                        # 存储到文本块嵌入缓存中
                        self.chunk_embedding_cache[chunk_id] = embed_tensor
                    except Exception as e:
                        continue

                # 关闭numpy缓存文件
                numpy_cache.close()

                logger.info(
                    f"Loaded chunk embedding cache with {len(self.chunk_embedding_cache)} entries from {cache_path_npz}")
                return True

            except Exception as e:
                logger.error(f"Failed to load chunk embedding cache from {cache_path_npz}: {e}")
                return False

        # 如果.npyz格式不存在或加载失败，回退到.pt格式
        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:
                    return False

                # 尝试使用PyTorch加载.pt文件
                try:
                    cpu_cache = torch.load(cache_path, map_location='cpu', weights_only=False)
                except TypeError:
                    # 处理旧版本PyTorch可能不支持weights_only参数的情况
                    cpu_cache = torch.load(cache_path, map_location='cpu')
                except Exception as e:
                    # 处理numpy序列化相关的问题
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

                # 检查加载的缓存是否为空
                if not cpu_cache:
                    logger.warning(f"Chunk embedding cache is empty from {cache_path}")
                    return False

                # 清空当前文本块嵌入缓存
                self.chunk_embedding_cache.clear()

                # 遍历加载的缓存项
                for chunk_id, embed in cpu_cache.items():
                    if embed is not None:
                        try:
                            # 将嵌入向量转换为PyTorch张量并移到指定设备
                            if isinstance(embed, np.ndarray):
                                embed_tensor = torch.from_numpy(embed).float()
                            else:
                                embed_tensor = embed.cpu() if hasattr(embed, 'cpu') else embed

                            if self.device == "cuda" and torch.cuda.is_available():
                                embed_tensor = embed_tensor.to(self.device)
                            else:
                                embed_tensor = embed_tensor.to("cpu")

                            self.chunk_embedding_cache[chunk_id] = embed_tensor
                        except Exception as e:
                            logger.error(f"Warning: Failed to load chunk embedding for {chunk_id}: {e}")
                            continue

                # 如果成功加载了文本块嵌入缓存
                if self.chunk_embedding_cache:
                    try:
                        # 准备构建FAISS索引所需的数据
                        embeddings_list = []
                        valid_chunk_ids = []

                        # 收集所有嵌入向量和对应的文本块ID
                        for chunk_id, embed in self.chunk_embedding_cache.items():
                            embeddings_list.append(embed.cpu().numpy())
                            valid_chunk_ids.append(chunk_id)

                        # 构建FAISS索引
                        embeddings_array = np.array(embeddings_list)
                        dimension = embeddings_array.shape[1]

                        # 创建FAISS索引用于快速相似性搜索
                        self.chunk_faiss_index = faiss.IndexFlatIP(dimension)
                        self.chunk_faiss_index.add(embeddings_array.astype('float32'))

                        # 建立文本块ID与索引位置的双向映射关系
                        self.chunk_id_to_index.clear()
                        self.index_to_chunk_id.clear()
                        for i, chunk_id in enumerate(valid_chunk_ids):
                            self.chunk_id_to_index[chunk_id] = i
                            self.index_to_chunk_id[i] = chunk_id

                    except Exception as e:
                        return False

                # 检查加载的缓存与当前数据是否一致
                if not self._check_chunk_cache_consistency():
                    return False

                logger.info(
                    f"从 {cache_path} 加载了包含 {len(self.chunk_embedding_cache)} 个条目的文本块嵌入缓存 (文件大小: {file_size} 字节)")
                return True

            except Exception as e:
                logger.error(f"Error loading chunk embedding cache: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted chunk cache file: {cache_path}")
                except Exception as e:
                    logger.error(f"Error removing corrupted chunk cache file: {cache_path}: {e}")
        else:
            logger.info(f"未找到文本块缓存文件: {cache_path}")
        return False

    def _check_chunk_cache_consistency(self):
        """检查加载的文本块缓存是否与当前文本块保持一致"""
        try:
            # 获取当前文本块ID集合（从chunk2id字典中获取所有键）
            current_chunk_ids = set(self.chunk2id.keys())
            # 获取缓存中文本块ID集合（从chunk_embedding_cache字典中获取所有键）
            cached_chunk_ids = set(self.chunk_embedding_cache.keys())

            # 计算当前文本块中存在但缓存中缺失的文本块
            missing_chunks = current_chunk_ids - cached_chunk_ids
            if missing_chunks:
                logger.info(f"Chunk cache missing {len(missing_chunks)} chunks from current chunks")
                return False

            # 计算缓存中存在但当前文本块中不存在的文本块（多余的文本块）
            extra_chunks = cached_chunk_ids - current_chunk_ids
            # 如果多余文本块数量超过当前文本块总数的10%，记录日志并返回False
            if len(extra_chunks) > len(current_chunk_ids) * 0.1:
                logger.info(
                    f"Chunk cache has too many extra chunks: {len(extra_chunks)} extra vs {len(current_chunk_ids)} current")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking chunk cache consistency: {e}")
            return False

    def _chunk_embedding_retrieval(self, question_embed: torch.Tensor, top_k: int = 20) -> Dict:
        try:
            # 检查文本块嵌入是否已预计算且FAISS索引是否存在
            if not self.chunk_embeddings_precomputed or self.chunk_faiss_index is None:
                logger.info("Warning: Chunk embeddings not precomputed, skipping chunk retrieval")
                # 如果未预计算，则返回空结果
                return {
                    "chunk_ids": [],
                    "scores": [],
                    "chunk_contents": []
                }

            # 将问题嵌入转换为numpy数组并调整形状以适应FAISS搜索
            query_embed_np = question_embed.cpu().numpy().reshape(1, -1).astype('float32')
            # 使用FAISS索引执行相似性搜索，返回top_k个最相似的文本块
            scores, indices = self.chunk_faiss_index.search(query_embed_np, min(top_k, self.chunk_faiss_index.ntotal))

            # 初始化存储结果的列表
            chunk_ids = []  # 文本块ID列表
            similarity_scores = []  # 相似性分数列表
            chunk_contents = []  # 文本块内容列表

            # 遍历搜索结果
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                # 检查索引是否有效且存在于索引到文本块ID的映射中
                if idx != -1 and idx in self.index_to_chunk_id:
                    # 获取文本块ID
                    chunk_id = self.index_to_chunk_id[idx]
                    chunk_ids.append(chunk_id)
                    # 添加相似性分数
                    similarity_scores.append(float(score))

                    # 获取文本块内容
                    if chunk_id in self.chunk2id:
                        chunk_contents.append(self.chunk2id[chunk_id])
                    else:
                        chunk_contents.append(f"[Missing content for chunk {chunk_id}]")

            return {
                "chunk_ids": chunk_ids,
                "scores": similarity_scores,
                "chunk_contents": chunk_contents
            }

        except Exception as e:
            logger.error(f"Error in chunk embedding retrieval: {str(e)}")
            return {
                "chunk_ids": [],
                "scores": [],
                "chunk_contents": []
            }

    def _rerank_chunks_by_relevance(self, chunk_results: Dict, question_embed: torch.Tensor, top_k: int = 10) -> Dict:
        """
        使用语义相似度重新排列文本块的相关性

        参数:
            chunk_results: 包含chunk_ids, scores和chunk_contents的字典
            question_embed: 查询嵌入张量
            top_k: 返回的顶部文本块数量

        返回:
            更新了评分的重新排列的文本块结果
        """
        try:
            # 从chunk_results中提取文本块ID、原始评分和内容
            chunk_ids = chunk_results.get('chunk_ids', [])
            original_scores = chunk_results.get('scores', [])
            chunk_contents = chunk_results.get('chunk_contents', [])

            if not chunk_ids or not chunk_contents:
                return chunk_results

            # 存储文本块相似度信息的列表
            chunk_similarities = []
            # 遍历所有文本块及其内容
            for i, (chunk_id, content) in enumerate(zip(chunk_ids, chunk_contents)):
                try:
                    # 对文本块内容进行编码，生成嵌入向量
                    chunk_embed = torch.tensor(self.qa_encoder.encode(content)).float().to(self.device)

                    # 计算问题嵌入与文本块嵌入之间的余弦相似度
                    similarity = F.cosine_similarity(question_embed, chunk_embed, dim=0).item()
                    similarity = max(0.0, similarity)  # Ensure non-negative

                    # 获取原始FAISS评分
                    faiss_score = original_scores[i] if i < len(original_scores) else 0.0
                    # 结合FAISS评分和语义相似度评分，计算综合评分
                    combined_score = (faiss_score + similarity) / 2.0  # Average of both scores

                    # 将文本块信息和综合评分添加到列表中,
                    chunk_similarities.append((chunk_id, content, combined_score, i))

                except Exception as e:
                    logger.error(f"Error calculating similarity for chunk {chunk_id}: {str(e)}")
                    faiss_score = original_scores[i] if i < len(original_scores) else 0.0
                    chunk_similarities.append((chunk_id, content, faiss_score, i))

            # 根据综合评分对文本块进行降序排序
            chunk_similarities.sort(key=lambda x: x[2], reverse=True)

            # 获取前top_k个文本块
            top_chunks = chunk_similarities[:top_k]

            # 提取重新排序后的文本块ID、评分和内容
            reranked_chunk_ids = [chunk_id for chunk_id, _, _, _ in top_chunks]
            reranked_scores = [score for _, _, score, _ in top_chunks]
            reranked_contents = [content for _, content, _, _ in top_chunks]

            # 返回重新排序的结果
            return {
                "chunk_ids": reranked_chunk_ids,
                "scores": reranked_scores,
                "chunk_contents": reranked_contents
            }

        except Exception as e:
            logger.error(f"Error in chunk reranking: {str(e)}")
            return chunk_results
