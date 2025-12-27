import json
import os
import threading
import time
from concurrent import futures
from typing import Any, Dict, List, Tuple

import nanoid
import networkx as nx
import tiktoken
import json_repair

import numpy as np
from sentence_transformers import SentenceTransformer

from config import get_config
from utils import call_llm_api, graph_processor, tree_comm
from utils.logger import logger


class KTBuilder:
    """
   知识图谱构建器主类，负责从文本文档中提取信息并构建成多层知识图谱
   """

    def __init__(self, dataset_name, schema_path=None, mode=None, config=None, is_incremental=False):
        """
        初始化KTBuilder实例，
        新增 is_incremental 参数控制是否热加载旧图谱

        Args:
            dataset_name: 数据集名称
            schema_path: 模式文件路径（可选）
            mode: 处理模式（"agent" 或标准模式）
            config: 配置对象（可选）
        """
        # 加载配置
        if config is None:
            config = get_config()

        self.config = config
        self.dataset_name = dataset_name
        # 加载模式定义
        self.schema = self.load_schema(schema_path or config.get_dataset_config(dataset_name).schema_path)
        self.is_incremental = is_incremental

        # 初始化NetworkX图结构
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0

        logger.info("正在初始化语义模型 (BGE-M3)...")
        self.embedder = SentenceTransformer(config.embeddings.model_name, device=config.embeddings.device)
        self.node_embeddings_cache = {"ids": [], "vecs": None}

        if self.is_incremental:
            self._hot_load_existing_graph()
        else:
            logger.info("🆕 [标准模式] 初始化空图谱...")

        # 不需要分块的数据集列表
        self.datasets_no_chunk = config.construction.datasets_no_chunk
        self.token_len = 0
        # 线程锁用于并发安全
        self.lock = threading.Lock()
        # LLM客户端实例
        self.llm_client = call_llm_api.LLMCompletionCall()
        # 存储所有文本块
        self.all_chunks = {}
        # 设置处理模式
        self.mode = mode or config.construction.mode

    def _hot_load_existing_graph(self):
        """
        【新增】读取 output/graphs/xxx_new.json (List格式) 并加载到 self.graph
        """
        old_graph_path = os.path.join(self.config.output.graphs_dir, f"{self.dataset_name}_new.json")

        if os.path.exists(old_graph_path):
            logger.info(f"🔄 [增量模式] 正在加载旧图谱: {old_graph_path}")
            try:
                with open(old_graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)  # 这里 data 是一个 List

                if not isinstance(data, list):
                    logger.error("⚠️ 旧图谱格式异常（非List），跳过加载")
                    return

                count = 0
                for item in data:
                    # 严格按照 demo_new.json 结构解析
                    s_node = item.get("start_node")
                    e_node = item.get("end_node")
                    relation = item.get("relation")

                    if s_node and e_node:
                        # 提取节点名称作为 ID
                        s_name = s_node.get("properties", {}).get("name", f"unknown_{count}")
                        e_name = e_node.get("properties", {}).get("name", f"unknown_{count + 1}")

                        # 添加节点和属性
                        self.graph.add_node(s_name, **s_node)
                        self.graph.add_node(e_name, **e_node)

                        # 添加边
                        self.graph.add_edge(s_name, e_name, relation=relation)
                        count += 1

                # 恢复计数器 (简单策略：基于当前节点数，避免新生成的 IDentity_0 冲突)
                # 虽然加载的节点用的是 Name 作 ID，但新节点会用 entity_X
                self.node_counter = self.graph.number_of_nodes() + 1000

                logger.info(f"✅ 热加载完成，恢复节点数: {self.graph.number_of_nodes()}，边数: {count}")

                # 立即构建语义索引
                self._precompute_graph_embeddings()

            except Exception as e:
                logger.error(f"❌ 热加载失败: {e}，将回退到空图谱")
        else:
            logger.warning(f"⚠️ 未找到旧图谱文件 {old_graph_path}，将开始全新构建")

    def _precompute_graph_embeddings(self):
        """【新增】为图谱中的 Entity 节点计算向量"""
        if self.graph.number_of_nodes() == 0: return

        nodes_text = []
        nodes_id = []

        for n, d in self.graph.nodes(data=True):
            # 过滤：只对 'entity' 类型的节点做索引，忽略 'attribute' 节点
            if d.get('label') == 'entity':
                # 获取名称，优先用 properties.name，没有则用 ID
                name = d.get('properties', {}).get('name', str(n))
                if name:
                    nodes_text.append(name)
                    nodes_id.append(n)

        if nodes_text:
            logger.info(f"🔍 [图驱动] 正在为 {len(nodes_text)} 个历史实体生成索引...")
            embeddings = self.embedder.encode(nodes_text, normalize_embeddings=True)
            self.node_embeddings_cache = {"ids": nodes_id, "vecs": embeddings}

    def _get_relevant_subgraph_context(self, chunk_text: str, top_k=3) -> str:
        """【新增】检索与 chunk 相关的旧知识"""
        if self.node_embeddings_cache["vecs"] is None:
            return "暂无历史记录。"

        # 1. 编码 Chunk (取前512字符)
        chunk_vec = self.embedder.encode([chunk_text[:512]], normalize_embeddings=True)

        # 2. 向量相似度
        similarities = np.dot(self.node_embeddings_cache["vecs"], chunk_vec.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        context_lines = []
        seen = set()

        with self.lock:
            for idx in top_indices:
                node_id = self.node_embeddings_cache["ids"][idx]

                # 获取 1-hop 邻居
                edges = list(self.graph.out_edges(node_id, data=True)) + \
                        list(self.graph.in_edges(node_id, data=True))

                for u, v, d in edges:
                    # 简单的边去重
                    edge_key = tuple(sorted((u, v)))
                    if edge_key in seen: continue
                    seen.add(edge_key)

                    u_name = self.graph.nodes[u].get('properties', {}).get('name', u)
                    v_name = self.graph.nodes[v].get('properties', {}).get('name', v)
                    rel = d.get('relation', 'related')
                    context_lines.append(f"[{u_name}, {rel}, {v_name}]")

        if not context_lines: return "暂无历史记录。"
        # 返回前 15 条，避免 Prompt 溢出
        return "\n".join(context_lines[:15])

    def load_schema(self, schema_path) -> Dict[str, Any]:
        """
           加载模式定义文件

           Args:
               schema_path: 模式文件路径

           Returns:
               解析后的模式字典，如果文件不存在则返回空字典
           """
        try:
            with open(schema_path) as f:
                schema = json.load(f)
                return schema
        except FileNotFoundError:
            return dict()

    def chunk_text(self, text) -> Tuple[List[str], Dict[str, str]]:
        """
            将文本分割成块，为每个文本块生成唯一的标识符。
            Args:
                text: 输入文本

            Returns:
                (chunks列表, chunk_id到chunk文本的映射)
            """
        if self.dataset_name in self.datasets_no_chunk:
            chunks = [f"{text.get('title', '')} {text.get('text', '')}".strip()
                      if isinstance(text, dict) else str(text)]
        else:
            chunks = [str(text)]

        chunk2id = {}
        for chunk in chunks:
            try:
                # 为每个文本块生成唯一的8位nanoid作为标识符
                chunk_id = nanoid.generate(size=8)
                chunk2id[chunk_id] = chunk
            except Exception as e:
                logger.warning(f"Failed to generate chunk id with nanoid: {type(e).__name__}: {e}")

        with self.lock:
            self.all_chunks.update(chunk2id)

        return chunks, chunk2id

    def _clean_text(self, text: str) -> str:
        """
           清理文本内容，移除不安全字符

           Args:
               text: 原始文本

           Returns:
               清理后的文本
           """
        # 如果输入文本为空（None、空字符串等），直接返回占位符 [EMPTY_TEXT]
        if not text:
            return "[EMPTY_TEXT]"

        if self.dataset_name == "graphrag-bench":
            # 安全字符集合
            safe_chars = {
                *" .:,!?()-+=[]{}()\\/|_^~<>*&%$#@!;\"'`"
            }
            # 保留字母数字，空白，安全字符
            cleaned = "".join(
                char for char in text
                if char.isalnum() or char.isspace() or char in safe_chars
            ).strip()
        else:
            # 更严格的安全字符
            safe_chars = {
                *" .:,!?()-+="
            }
            cleaned = "".join(
                char for char in text
                if char.isalnum() or char.isspace() or char in safe_chars
            ).strip()

        return cleaned if cleaned else "[EMPTY_AFTER_CLEANING]"

    def save_chunks_to_file(self):
        """
        将文本块保存到文件中，支持增量更新已有文件
        """
        os.makedirs("output/chunks", exist_ok=True)
        chunk_file = f"output/chunks/{self.dataset_name}.txt"

        existing_data = {}
        # 如果文件已存在，尝试读取其中的内容
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # 只处理非空且包含制表符的行
                        if line and "\t" in line:
                            # 解析行格式: "id: {id} \tChunk: {chunk text}"
                            parts = line.split("\t", 1)
                            # 验证行格式是否正确
                            if len(parts) == 2 and parts[0].startswith("id: ") and parts[1].startswith("Chunk: "):
                                # 提取chunk ID
                                chunk_id = parts[0][4:]
                                # 提取chunk文本 (跳过"Chunk: "前缀)
                                chunk_text = parts[1][7:]
                                # 将现有数据存入字典
                                existing_data[chunk_id] = chunk_text
            except Exception as e:
                logger.warning(f"Failed to parse existing chunks from {chunk_file}: {type(e).__name__}: {e}")

        # 合并现有数据和新的文本块数据，新数据会覆盖同ID的旧数据
        all_data = {**existing_data, **self.all_chunks}

        # 将所有数据写入文件
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk_id, chunk_text in all_data.items():
                f.write(f"id: {chunk_id}\tChunk: {chunk_text}\n")

        logger.info(f"文本块数据已保存到 {chunk_file} ({len(all_data)} 个文本块)")

    def extract_with_llm(self, prompt: str):
        """
       调用LLM API提取信息

       Args:
           prompt: 发送给LLM的提示词

       Returns:
           LLM返回的JSON格式响应
       """
        logger.info(f"prompt:{prompt}")
        # 调用LLM客户端的API接口，传入提示词获取响应
        response = self.llm_client.call_api(prompt)
        # 使用json_repair库修复并解析LLM返回的JSON响应
        # 这可以处理LLM返回的不完整或格式略有错误的JSON
        parsed_dict = json_repair.loads(response)
        # 将解析后的字典对象重新序列化为格式化的JSON字符串
        # ensure_ascii=False确保中文等非ASCII字符正常显示
        parsed_json = json.dumps(parsed_dict, ensure_ascii=False)
        return parsed_json

    def token_cal(self, text: str):
        """
           计算文本的token数量

           Args:
               text: 待计算文本

           Returns:
               token数量
           """
        # 使用tiktoken库获取cl100k_base编码器
        # cl100k_base是与GPT-3.5和GPT-4兼容的编码器
        encoding = tiktoken.get_encoding("cl100k_base")

        # 将文本编码为token序列，并返回序列长度
        return len(encoding.encode(text))

    def _get_construction_prompt(self, chunk: str) -> str:
        """
            根据数据集名称生成相应的构建提示词

            Args:
                chunk: 文本块内容

            Returns:
                格式化后的提示词
            """

        # 获取推荐的模式定义并转换为JSON字符串
        recommend_schema = json.dumps(self.schema, ensure_ascii=False)

        # 优先从配置中获取提示词类型
        # 如果配置中没有为该数据集指定 prompt_type，则使用默认的 "general"
        prompt_type = "general"
        if self.config and hasattr(self.config, 'get_dataset_config'):
            dataset_config = self.config.get_dataset_config(self.dataset_name)
            # 尝试从数据集配置中获取 prompt_type
            prompt_type = getattr(dataset_config, 'prompt_type', prompt_type)

        # 调用配置管理器获取格式化后的提示词
        # 参数说明：
        # - "construction": 提示词类别为构建类
        # - prompt_type: 具体的提示词类型
        # - schema: 模式定义JSON字符串
        # - chunk: 当前处理的文本块内容
        return self.config.get_prompt_formatted(
            "construction",
            prompt_type,
            schema=recommend_schema,
            chunk=chunk)

    # 【新增】增量 Prompt 获取方法
    def _get_incremental_construction_prompt(self, chunk: str) -> str:
        recommend_schema = json.dumps(self.schema, ensure_ascii=False)

        # 1. 获取图驱动上下文
        examples_context = self._get_relevant_subgraph_context(chunk)

        # 2. 映射到增量 Prompt 模板 (例如 general -> general_incremental)
        prompt_type = "general_incremental"
        if self.config and hasattr(self.config, 'get_dataset_config'):
            dataset_config = self.config.get_dataset_config(self.dataset_name)
            # 尝试从数据集配置中获取 prompt_type
            prompt_type = getattr(dataset_config, 'prompt_type', prompt_type)+"_incremental"

        # 3. 注入 examples
        return self.config.get_prompt_formatted(
            "construction",
            prompt_type,
            schema=recommend_schema,
            chunk=chunk,
            examples=examples_context
        )

    def _validate_and_parse_llm_response(self, prompt: str, llm_response: str) -> dict:
        """
           验证并解析LLM响应

           Args:
               prompt: 发送的提示词
               llm_response: LLM响应

           Returns:
               解析后的字典，如果无效则返回None
           """
        if llm_response is None:
            return None

        try:
            # 累计计算提示词和响应的token总长度
            self.token_len += self.token_cal(prompt + llm_response)
            # 使用json_repair库解析LLM响应为Python字典
            return json_repair.loads(llm_response)
        except Exception as e:
            llm_response_str = str(llm_response) if llm_response is not None else "None"
            return None

    def _find_or_create_entity(self, entity_name: str, chunk_id: int, nodes_to_add: list,
                               entity_type: str = None) -> str:
        """
            查找现有实体或创建新实体（批处理模式）

            Args:
                entity_name: 实体名称
                chunk_id: 文本块ID
                nodes_to_add: 待添加节点列表
                entity_type: 实体类型（可选）

            Returns:
                实体节点ID
            """
        with self.lock:
            # 在当前图中查找具有相同名称的实体节点
            entity_node_id = next(
                (
                    n
                    for n, d in self.graph.nodes(data=True)
                    if d.get("label") == "entity" and d["properties"]["name"] == entity_name  # 筛选条件：标签为"entity"且名称匹配
                ),
                None,
            )

            # 如果未找到同名实体节点，则创建新节点
            if not entity_node_id:
                # 生成新的实体节点ID，格式为"entity_序号"
                entity_node_id = f"entity_{self.node_counter}"
                properties = {"name": entity_name, "chunk id": chunk_id}
                if entity_type:
                    properties["schema_type"] = entity_type

                nodes_to_add.append((
                    entity_node_id,
                    {
                        "label": "entity",
                        "properties": properties,
                        "level": 2
                    }
                ))
                self.node_counter += 1

        return entity_node_id

    def _validate_triple_format(self, triple: list) -> tuple:
        """
           验证并规范化三元组格式

           Args:
               triple: 原始三元组列表

           Returns:
               规范化后的(subject, predicate, object)元组，无效则返回None
           """
        try:
            if len(triple) > 3:
                triple = triple[:3]
            elif len(triple) < 3:
                return None

            return tuple(triple)
        except Exception as e:
            return None

    def _process_attributes(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None) -> tuple[list, list]:
        """
        处理提取的属性信息

        Args:
            extracted_attr: 提取的属性字典，格式为 {实体名: [属性列表]}
            chunk_id: 文本块ID，标识属性信息的来源
            entity_types: 实体类型映射（可选），格式为 {实体名: 类型}

        Returns:
            (待添加节点列表, 待添加边列表)
        """
        # 初始化待添加的节点列表和边列表
        nodes_to_add = []
        edges_to_add = []

        # 遍历提取的属性字典
        for entity, attributes in extracted_attr.items():
            # 遍历实体的所有属性
            for attr in attributes:
                # 创建属性节点ID，格式为"attr_序号"
                attr_node_id = f"attr_{self.node_counter}"

                # 将属性节点添加到待添加列表中
                nodes_to_add.append((
                    attr_node_id,
                    {
                        "label": "attribute",  # 节点标签为"attribute"
                        "properties": {"name": attr, "chunk id": chunk_id},  # 节点属性包含属性名和来源文本块ID
                        "level": 1,  # 节点层级为第1层（属性层）
                    }
                ))
                self.node_counter += 1

                # 获取实体的类型信息（如果提供了entity_types）
                entity_type = entity_types.get(entity) if entity_types else None
                # 查找或创建实体节点（使用批处理模式）
                entity_node_id = self._find_or_create_entity(entity, chunk_id, nodes_to_add, entity_type)
                # 将实体节点与属性节点之间的关系边添加到待添加列表
                # 关系类型为"has_attribute"，表示实体拥有该属性
                edges_to_add.append((entity_node_id, attr_node_id, "has_attribute"))

        return nodes_to_add, edges_to_add

    def _process_triples(self, extracted_triples: list, chunk_id: int, entity_types: dict = None) -> tuple[list, list]:
        """
            处理提取的三元组信息

            Args:
                extracted_triples: 提取的三元组列表，每个元素为 [subject, predicate, object]
                chunk_id: 文本块ID，标识三元组信息的来源
                entity_types: 实体类型映射（可选），格式为 {实体名: 类型}

            Returns:
                (待添加节点列表, 待添加边列表)
            """
        # 初始化待添加的节点列表和边列表
        nodes_to_add = []
        edges_to_add = []

        # 遍历提取的所有三元组
        for triple in extracted_triples:
            # 验证并规范化三元组格式
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue

            # 解包验证后的三元组为subject、predicate、object
            subj, pred, obj = validated_triple

            # 获取主语和宾语的类型信息（如果提供了entity_types）
            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None

            # 查找或创建主语实体节点（使用批处理模式）
            subj_node_id = self._find_or_create_entity(subj, chunk_id, nodes_to_add, subj_type)
            # 查找或创建宾语实体节点（使用批处理模式）
            obj_node_id = self._find_or_create_entity(obj, chunk_id, nodes_to_add, obj_type)

            # 将主语节点与宾语节点之间的关系边添加到待添加列表
            # 关系类型为三元组中的谓词(predicate)
            edges_to_add.append((subj_node_id, obj_node_id, pred))

        return nodes_to_add, edges_to_add

    def process_level1_level2(self, chunk: str, id: int):
        """
           处理第1层（属性）和第2层（三元组）信息的标准模式

           Args:
               chunk: 文本块内容
               id: 文本块ID
        """
        # 生成构建知识图谱的提示词
        # 根据模式选择 Prompt 方法 ---
        if self.is_incremental:
            logger.info("使用增量构建提示词")
            prompt = self._get_incremental_construction_prompt(chunk)
        else:
            logger.info("使用完整构建提示词")
            prompt = self._get_construction_prompt(chunk)

        # 调用LLM API提取信息
        llm_response = self.extract_with_llm(prompt)

        # 验证并解析LLM响应
        parsed_response = self._validate_and_parse_llm_response(prompt, llm_response)
        if not parsed_response:
            return

        # 从解析后的响应中提取属性、三元组和实体类型信息
        extracted_attr = parsed_response.get("attributes", {})  # 属性信息字典
        extracted_triples = parsed_response.get("triples", [])  # 三元组列表
        entity_types = parsed_response.get("entity_types", {})  # 实体类型映射

        # 处理属性信息，生成属性节点和"has_attribute"边
        attr_nodes, attr_edges = self._process_attributes(extracted_attr, id, entity_types)
        # 处理三元组信息，生成实体节点间的关系边
        triple_nodes, triple_edges = self._process_triples(extracted_triples, id, entity_types)

        # 合并所有待添加的节点和边
        all_nodes = attr_nodes + triple_nodes
        all_edges = attr_edges + triple_edges

        with self.lock:
            for node_id, node_data in all_nodes:
                self.graph.add_node(node_id, **node_data)

            for u, v, relation in all_edges:
                self.graph.add_edge(u, v, relation=relation)

    def _find_or_create_entity_direct(self, entity_name: str, chunk_id: int, entity_type: str = None) -> str:
        """
            查找现有实体或创建新实体（直接操作图模式，用于agent模式）

            Args:
                entity_name: 实体名称
                chunk_id: 文本块ID
                entity_type: 实体类型（可选）

            Returns:
                实体节点ID
            """
        # 在当前图中查找具有相同名称的实体节点
        entity_node_id = next(
            (
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("label") == "entity" and d["properties"]["name"] == entity_name
            ),
            None,
        )

        # 如果未找到同名实体节点，则创建新节点并直接添加到图中
        if not entity_node_id:
            entity_node_id = f"entity_{self.node_counter}"
            properties = {"name": entity_name, "chunk id": chunk_id}
            if entity_type:
                properties["schema_type"] = entity_type

            self.graph.add_node(
                entity_node_id,
                label="entity",
                properties=properties,
                level=2
            )
            self.node_counter += 1

        return entity_node_id

    def _process_attributes_agent(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None):
        """
           处理属性信息（agent模式，直接操作图）

           Args:
               extracted_attr: 提取的属性字典
               chunk_id: 文本块ID
               entity_types: 实体类型映射（可选）
           """
        # 遍历提取的属性字典中的每个实体及其属性列表
        for entity, attributes in extracted_attr.items():
            # 遍历当前实体的所有属性
            for attr in attributes:
                # Create attribute node
                attr_node_id = f"attr_{self.node_counter}"
                # 直接将属性节点添加到知识图谱中
                self.graph.add_node(
                    attr_node_id,
                    label="attribute",
                    properties={
                        "name": attr,
                        "chunk id": chunk_id
                    },
                    level=1,
                )
                self.node_counter += 1

                entity_type = entity_types.get(entity) if entity_types else None
                entity_node_id = self._find_or_create_entity_direct(entity, chunk_id, entity_type)
                self.graph.add_edge(entity_node_id, attr_node_id, relation="has_attribute")

    def _process_triples_agent(self, extracted_triples: list, chunk_id: int, entity_types: dict = None):
        """
       处理三元组信息（agent模式，直接操作图）

       Args:
        extracted_triples: 提取的三元组列表，每个元素为 [subject, predicate, object]
        chunk_id: 文本块ID，标识三元组信息的来源
        entity_types: 实体类型映射（可选），格式为 {实体名: 类型}
       """
        # 遍历提取的所有三元组
        for triple in extracted_triples:
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue

            # 解包验证后的三元组为subject、predicate、object
            subj, pred, obj = validated_triple

            # 获取主语和宾语的类型信息（如果提供了entity_types）
            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None

            # 查找或创建主语实体节点（使用agent模式的直接操作方法）
            subj_node_id = self._find_or_create_entity_direct(subj, chunk_id, subj_type)
            # 查找或创建宾语实体节点（使用agent模式的直接操作方法
            obj_node_id = self._find_or_create_entity_direct(obj, chunk_id, obj_type)

            # 直接在图中添加主语节点与宾语节点之间的关系边
            # 关系类型为三元组中的谓词(predicate)
            self.graph.add_edge(subj_node_id, obj_node_id, relation=pred)

    def process_level1_level2_agent(self, chunk: str, id: int):
        """
           处理第1层和第2层信息的agent模式，支持模式演化

           Args:
               chunk: 文本块内容
               id: 文本块ID
           """
        # 生成构建知识图谱的提示词
        # 根据模式选择 Prompt 方法 ---
        if self.is_incremental:
            logger.info("使用增量构建提示词")
            prompt = self._get_incremental_construction_prompt(chunk)
        else:
            logger.info("使用完整构建提示词")
            prompt = self._get_construction_prompt(chunk)
        # 调用LLM API提取信息
        llm_response = self.extract_with_llm(prompt)

        # 验证并解析LLM响应（复用已有的辅助方法）
        parsed_response = self._validate_and_parse_llm_response(prompt, llm_response)
        if not parsed_response:
            return

        # 处理模式演化：检查是否有新的模式类型被发现
        new_schema_types = parsed_response.get("new_schema_types", {})
        if new_schema_types:
            # 如果有新类型，则更新模式定义文件
            self._update_schema_with_new_types(new_schema_types)

        # 从解析后的响应中提取属性、三元组和实体类型信息
        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})

        with self.lock:
            # 处理属性信息（agent模式，直接操作图）
            self._process_attributes_agent(extracted_attr, id, entity_types)
            # 处理三元组信息（agent模式，直接操作图）
            self._process_triples_agent(extracted_triples, id, entity_types)

    def _update_schema_with_new_types(self, new_schema_types: Dict[str, List[str]]):
        """
        使用agent发现的新类型更新模式文件

        Args:
            new_schema_types: 新类型字典
        """
        try:
            # 定义数据集名称到模式文件路径的映射关系
            schema_paths = {
                "hotpot": "schemas/hotpot.json",
                "2wiki": "schemas/2wiki.json",
                "musique": "schemas/musique.json",
                "novel": "schemas/novels_chs.json",
                "graphrag-bench": "schemas/graphrag-bench.json"
            }

            # 根据当前数据集名称获取对应的模式文件路径
            schema_path = schema_paths.get(self.dataset_name)
            if not schema_path:
                return

            # 读取当前的模式文件内容
            with open(schema_path, 'r', encoding='utf-8') as f:
                current_schema = json.load(f)

            updated = False

            # 处理新发现的节点类型
            if "nodes" in new_schema_types:
                for new_node in new_schema_types["nodes"]:
                    # 检查新节点类型是否已存在于当前模式中
                    if new_node not in current_schema.get("Nodes", []):
                        # 如果不存在，则添加到节点类型列表中
                        current_schema.setdefault("Nodes", []).append(new_node)
                        updated = True

            # 处理新发现的关系类型
            if "relations" in new_schema_types:
                for new_relation in new_schema_types["relations"]:
                    # 检查新关系类型是否已存在于当前模式中
                    if new_relation not in current_schema.get("Relations", []):
                        # 如果不存在，则添加到关系类型列表中
                        current_schema.setdefault("Relations", []).append(new_relation)
                        updated = True

            # 处理新发现的属性类型
            if "attributes" in new_schema_types:
                for new_attribute in new_schema_types["attributes"]:
                    # 检查新属性类型是否已存在于当前模式中
                    if new_attribute not in current_schema.get("Attributes", []):
                        # 如果不存在，则添加到属性类型列表中
                        current_schema.setdefault("Attributes", []).append(new_attribute)
                        updated = True

            # 如果有更新发生，则保存更新后的模式到文件
            if updated:
                with open(schema_path, 'w', encoding='utf-8') as f:
                    json.dump(current_schema, f, ensure_ascii=False, indent=2)

                # Update the in-memory schema
                self.schema = current_schema

        except Exception as e:
            logger.error(f"Failed to update schema for dataset '{self.dataset_name}': {type(e).__name__}: {e}")

    def process_level4(self):
        """
        使用Tree-Comm算法处理社区（第4层）
        """
        logger.info("筛选出图中所有level为2的节点（实体节点）")
        level2_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('label') == 'entity']

        # 记录开始时间，用于性能统计
        start_comm = time.time()

        # 初始化FastTreeComm算法实例
        _tree_comm = tree_comm.FastTreeComm(
            self.graph,
            # 从配置中获取嵌入模型参数
            embedding_model=self.config.tree_comm.embedding_model,
            # 从配置中获取结构权重参数
            struct_weight=self.config.tree_comm.struct_weight,
        )

        logger.info("使用Tree-Comm算法检测社区，输入为level2的节点列表")
        comm_to_nodes = _tree_comm.detect_communities(level2_nodes)

        logger.info("为检测出的社区创建超级节点（level 4），并附带关键词信息")
        _tree_comm.create_super_nodes_with_keywords(comm_to_nodes, level=4)

        # 可选功能：将关键词连接到社区（当前被注释掉）
        # _tree_comm.add_keywords_to_level3(comm_to_nodes)
        # connect keywords to communities (optional)
        # self._connect_keywords_to_communities()

        # 记录结束时间并计算耗时
        end_comm = time.time()
        logger.info(f"社区索引耗时: {end_comm - start_comm}s")

    def _connect_keywords_to_communities(self):
        """
            将关键词连接到社区（可选功能）
            """
        # comm_names = [self.graph.nodes[n]['properties']['name'] for n, d in self.graph.nodes(data=True) if d['level'] == 4]
        comm_nodes = [n for n, d in self.graph.nodes(data=True) if d['level'] == 4]
        kw_nodes = [n for n, d in self.graph.nodes(data=True) if d['label'] == 'keyword']
        with self.lock:
            for comm in comm_nodes:
                comm_name = self.graph.nodes[comm]['properties']['name'].lower()
                for kw in kw_nodes:
                    kw_name = self.graph.nodes[kw]['properties']['name'].lower()
                    if kw_name in comm_name or comm_name in kw_name:
                        self.graph.add_edge(kw, comm, relation="describes")

    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
           处理单个文档

           Args:
               doc: 文档字典

           Returns:
               处理结果列表
           """
        try:
            if not doc:
                raise ValueError("Document is empty or None")

            # 将文档切分为多个文本块，并建立块ID到块内容的映射
            chunks, chunk2id = self.chunk_text(doc)

            if not chunks or not chunk2id:
                raise ValueError(
                    f"No valid chunks generated from document. Chunks: {len(chunks)}, Chunk2ID: {len(chunk2id)}")

            # 遍历所有文本块进行处理
            for chunk in chunks:
                try:
                    # 从chunk2id映射中查找当前chunk对应的ID
                    id = next(key for key, value in chunk2id.items() if value == chunk)
                except StopIteration:
                    # 如果找不到对应ID，则生成一个新的nanoid作为ID
                    id = nanoid.generate(size=8)
                    chunk2id[id] = chunk

                # 根据配置的模式选择不同的处理方法
                if self.mode == "agent":
                    # agent模式：支持模式演化的处理方式
                    self.process_level1_level2_agent(chunk, id)
                else:
                    # 标准模式：基础的知识图谱构建方式
                    self.process_level1_level2(chunk, id)

        except Exception as e:
            error_msg = f"Error processing document: {type(e).__name__}: {str(e)}"
            raise Exception(error_msg) from e

    def process_all_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
            并发处理所有文档

            Args:
                documents: 文档列表
            """

        # 计算最大工作线程数，取配置值和CPU核心数+4中的较小值
        max_workers = min(self.config.construction.max_workers, (os.cpu_count() or 1) + 4)

        # 记录开始处理时间，用于性能统计
        start_construct = time.time()
        total_docs = len(documents)

        logger.info(f"开始处理 {total_docs} 个文档，使用 {max_workers} 个工作线程...")

        # 初始化变量用于跟踪处理状态
        all_futures = []
        processed_count = 0
        failed_count = 0

        try:
            # 创建线程池执行器，使用计算得出的最大工作线程数
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有文档处理任务到线程池，并存储future对象
                all_futures = [executor.submit(self.process_document, doc) for doc in documents]

                # 遍历已完成的任务，处理结果和异常
                for i, future in enumerate(futures.as_completed(all_futures)):
                    try:
                        # 获取任务执行结果（此处不使用返回值）
                        future.result()
                        processed_count += 1

                        # 每处理10个文档或全部处理完成时输出进度信息
                        if processed_count % 10 == 0 or processed_count == total_docs:
                            # 计算已用时间和平均每个文档处理时间
                            elapsed_time = time.time() - start_construct
                            avg_time_per_doc = elapsed_time / processed_count if processed_count > 0 else 0
                            remaining_docs = total_docs - processed_count
                            # 估算剩余处理时间
                            estimated_remaining_time = remaining_docs * avg_time_per_doc

                            logger.info(f"进度: 已处理 {processed_count}/{total_docs} 个文档 "
                                        f"({processed_count / total_docs * 100:.1f}%) "
                                        f"[{failed_count} 个失败] "
                                        f"预计剩余时间: {estimated_remaining_time:.1f} 秒")

                    except Exception:
                        failed_count += 1

        except Exception:
            return

        end_construct = time.time()
        logger.info(f"构建耗时: {end_construct - start_construct}s")
        logger.info(f"成功处理: {processed_count}/{total_docs} 个文档")
        logger.info(f"失败: {failed_count} 个文档")

        logger.info(f"🚀🚀🚀🚀 {'正在处理第3层和第4层':^20} 🚀🚀🚀🚀")
        logger.info(f"{'➖' * 20}")

        # 执行三元组去重操作
        self.triple_deduplicate()
        # 处理第4层社区检测
        self.process_level4()

    def triple_deduplicate(self):
        """
           去除重复的三元组
           """
        """deduplicate triples in lv1 and lv2"""
        new_graph = nx.MultiDiGraph()

        for node, node_data in self.graph.nodes(data=True):
            new_graph.add_node(node, **node_data)

        seen_keys = set()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if (u, v, key) not in seen_keys:
                seen_keys.add((u, v, key))
                new_graph.add_edge(u, v, key=key, **data)
        self.graph = new_graph

    def format_output(self) -> List[Dict[str, Any]]:
        """
            将图转换为指定的输出格式

            Returns:
                格式化的边列表
            """
        output = []

        for u, v, data in self.graph.edges(data=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]

            relationship = {
                "start_node": {
                    "label": u_data["label"],
                    "properties": u_data["properties"],
                },
                "relation": data["relation"],
                "end_node": {
                    "label": v_data["label"],
                    "properties": v_data["properties"],
                },
            }
            output.append(relationship)

        return output

    def save_graphml(self, output_path: str):
        """
           保存图为GraphML格式

           Args:
               output_path: 输出路径
           """
        graph_processor.save_graph(self.graph, output_path)

    def build_knowledge_graph(self, corpus):
        """
       构建知识图谱的主入口点

       Args:
           corpus: 语料库文件路径

       Returns:
           格式化的图输出
       """
        logger.info(f"========{'开始构建知识图谱':^20}========")
        logger.info(f"{'➖' * 30}")

        # 读取语料库文件，使用json_repair处理可能存在的JSON格式问题
        with open(corpus, 'r', encoding='utf-8') as f:
            documents = json_repair.load(f)

        # 调用处理所有文档的方法，这是构建过程的核心步骤
        self.process_all_documents(documents)

        # 记录处理完成日志，并输出累计使用的token数量
        logger.info(f"所有处理完成，消耗token数: {self.token_len}")

        # 将文本块保存到文件中，供后续分析或调试使用
        self.save_chunks_to_file()

        # 将内部图结构格式化为输出格式
        output = self.format_output()

        # 构造输出JSON文件路径
        json_output_path = f"output/graphs/{self.dataset_name}_new.json"
        os.makedirs("output/graphs", exist_ok=True)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"图谱已保存到 {json_output_path}")

        # 返回格式化的图谱数据
        return output
