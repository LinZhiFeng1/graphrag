#!/usr/bin/env python3

import os
import re
import sys
import json
import asyncio
import glob
import shutil
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from utils.logger import logger
import ast

# 从models包导入GraphRAG核心组件
try:
    from models.constructor import kt_gen as constructor
    from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever
    from config import get_config, ConfigManager

    GRAPHRAG_AVAILABLE = True
    logger.info("✅ GraphRAG components loaded successfully")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    logger.error(f"⚠️  GraphRAG components not available: {e}")

app = FastAPI(title="GraphRAG Unified Interface", version="1.0.0")

# Mount static files (assets directory)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
# Mount frontend directory for frontend assets
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections: Dict[str, WebSocket] = {}
config = None


# WebSocket连接管理器，用于管理WebSocket连接
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)


manager = ConnectionManager()


# 数据模型定义，定义各种API请求和响应的数据模型
# Request/Response models
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_name: Optional[str] = None
    files_count: Optional[int] = None


class GraphConstructionRequest(BaseModel):
    dataset_name: str


class GraphConstructionResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None


class QuestionRequest(BaseModel):
    question: str
    dataset_name: str
    alpha: Optional[float] = 1.0  # 新增
    beta: Optional[float] = 0.0  # 新增


class QuestionResponse(BaseModel):
    answer: str
    sub_questions: List[Dict]
    retrieved_triples: List[str]
    retrieved_chunks: List[str]
    reasoning_steps: List[Dict]
    visualization_data: Dict


class GraphConstructionIncrementalRequest(BaseModel):
    dataset_name: str  # 数据源（新上传的，例如 aviation_1）
    target_dataset_name: Optional[str] = None  # 目标库（要合并到的，例如 aviation）


async def send_progress_update(client_id: str, stage: str, progress: int, message: str):
    """通过WebSocket向客户端发送进度更新"""
    await manager.send_message({
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }, client_id)


# 清理指定数据集的缓存文件
async def clear_cache_files(dataset_name: str):
    """在构建图谱之前清除指定数据集的所有缓存文件"""
    try:
        # 清除FAISS向量索引缓存目录
        # FAISS是Facebook AI Similarity Search的缩写，用于快速相似性搜索
        faiss_cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")

        # 清除输出的文本块文件
        # 这些文件包含了处理后的文本分块数据
        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            logger.info(f"Cleared chunk file: {chunk_file}")

        # 清除输出的图谱文件
        # 这是之前构建的知识图谱JSON文件
        graph_file = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info(f"Cleared graph file: {graph_file}")

        # 清除其他匹配数据集名称模式的缓存文件
        # 使用通配符模式匹配可能存在的其他相关缓存文件
        cache_patterns = [
            f"output/logs/{dataset_name}_*.log",
            f"output/chunks/{dataset_name}_*",
            f"output/graphs/{dataset_name}_*"
        ]

        # 遍历所有文件模式并删除匹配的文件
        for pattern in cache_patterns:
            for file_path in glob.glob(pattern):
                # 使用glob模块查找匹配指定模式的所有文件路径
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"已清除缓存文件: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"已清除缓存目录: {file_path}")
                except Exception as e:
                    logger.warning(f"清除失败 {file_path}: {e}")

        logger.info(f"数据集缓存清理完成: {dataset_name}")

    except Exception as e:
        logger.error(f"Error clearing cache files for {dataset_name}: {e}")
        # Don't raise exception, just log the error


# 根路径返回前端主页
@app.get("/")
async def read_root():
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "GraphRAG Unified Interface is running!", "status": "ok"}


# 返回服务状态信息
@app.get("/api/status")
async def get_status():
    return {
        "message": "后端初始化完毕",
        "status": "ok",
        "graphrag_available": GRAPHRAG_AVAILABLE
    }


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # 建立WebSocket连接并将其添加到连接管理器中
    await manager.connect(websocket, client_id)
    try:
        # 持续监听来自客户端的消息
        while True:
            # 接收客户端发送的文本数据
            data = await websocket.receive_text()
            # 注意：这里接收到数据后没有进行任何处理，只是保持连接活跃
            # 实际应用中可能会对接收到的数据进行处理
    except WebSocketDisconnect:
        # 当WebSocket连接断开时，从连接管理器中移除该连接
        manager.disconnect(client_id)


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...), client_id: str = "default"):
    """上传文件并为图谱构建做准备"""
    try:
        # 使用第一个文件的原始文件名（不含扩展名）作为数据集名称
        main_file = files[0]
        original_name = os.path.splitext(main_file.filename)[0]
        # 清理文件名，使其符合文件系统命名规范
        dataset_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        dataset_name = dataset_name.replace(' ', '_')

        # 如果数据集已存在，则添加计数器后缀以避免冲突
        base_name = dataset_name
        counter = 1
        while os.path.exists(f"data/uploaded/{dataset_name}"):
            dataset_name = f"{base_name}_{counter}"
            counter += 1

        # 创建上传目录
        upload_dir = f"data/uploaded/{dataset_name}"
        os.makedirs(upload_dir, exist_ok=True)

        # 向客户端发送进度更新消息
        await send_progress_update(client_id, "upload", 10, "Starting file upload...")

        # 处理上传的文件
        corpus_data = []
        for i, file in enumerate(files):
            # 保存文件到磁盘
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # 根据文件类型处理文件内容
            if file.filename.endswith('.txt'):
                # 处理文本文件
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                corpus_data.append({
                    "title": file.filename,
                    "text": content
                })
            elif file.filename.endswith('.json'):
                # 处理JSON文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 如果是列表格式，扩展数据；否则添加单个对象
                        if isinstance(data, list):
                            corpus_data.extend(data)
                        else:
                            corpus_data.append(data)
                except:
                    # 如果JSON解析失败，则当作普通文本处理
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    corpus_data.append({
                        "title": file.filename,
                        "text": content
                    })

            # 更新进度（10%基础进度 + 文件处理进度）
            progress = 10 + (i + 1) * 80 // len(files)
            await send_progress_update(client_id, "upload", progress, f"Processed {file.filename}")

        # 保存语料库数据到corpus.json文件
        corpus_path = f"{upload_dir}/corpus.json"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

        # 创建数据集配置文件
        await create_dataset_config()

        # 发送上传完成的进度更新
        await send_progress_update(client_id, "upload", 100, "Upload completed successfully!")

        # 返回上传成功的响应
        return FileUploadResponse(
            success=True,
            message="Files uploaded successfully",
            dataset_name=dataset_name,
            files_count=len(files)
        )

    except Exception as e:
        await send_progress_update(client_id, "upload", 0, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def create_dataset_config():
    """
    创建标准数据集配置文件，包含预定义的节点类型、关系类型和属性定义
    该配置文件用于指导知识图谱的构建过程，确保图谱结构的一致性
    """
    # 总是使用demo.json模式文件以保证一致性
    schema_path = "schemas/demo.json"
    os.makedirs("schemas", exist_ok=True)

    # 检查demo.json模式文件是否存在，如果不存在则创建标准模式
    if not os.path.exists(schema_path):
        demo_schema = {
            "Nodes": [
                "person",  # 人物
                "location",  # 地点
                "organization",  # 组织机构
                "event",  # 事件
                "object",  # 物体
                "concept",  # 概念
                "time_period",  # 时间段
                "creative_work",  # 创作作品
                "biological_entity",  # 生物实体
                "natural_phenomenon"  # 自然现象
            ],
            "Relations": [
                "is_a",  # 是...的一种
                "part_of",  # 是...的一部分
                "located_in",  # 位于...
                "created_by",  # 由...创建
                "used_by",  # 被...使用
                "participates_in",  # 参与...
                "related_to",  # 相关...
                "belongs_to",  # 属于...
                "influences",  # 影响...
                "precedes",  # 在...之前
                "arrives_in",  # 到达...
                "comparable_to"  # 可比较...
            ],
            "Attributes": [
                "name",  # 名称
                "date",  # 日期
                "size",  # 大小
                "type",  # 类型
                "description",  # 描述
                "status",  # 状态
                "quantity",  # 数量
                "value",  # 价值
                "position",  # 位置
                "duration",  # 持续时间
                "time"  # 时间
            ]
        }

        with open(schema_path, 'w') as f:
            json.dump(demo_schema, f, indent=2)


@app.post("/api/construct-graph", response_model=GraphConstructionResponse)
async def construct_graph(request: GraphConstructionRequest, client_id: str = "default"):
    """
    从上传的数据中构建知识图谱

    参数:
        request (GraphConstructionRequest): 包含数据集名称的请求对象
        client_id (str): 客户端ID，用于发送进度更新，默认为"default"

    返回:
        GraphConstructionResponse: 图谱构建结果响应
    """
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")

        # 从请求中获取数据集名称
        dataset_name = request.dataset_name

        # 发送进度更新：开始清理旧缓存文件，进度2%
        await send_progress_update(client_id, "construction", 2, "清理旧缓存文件...")

        # 清理构建前的所有缓存文件
        await clear_cache_files(dataset_name)

        # 发送进度更新：开始初始化图构建器，进度5%
        await send_progress_update(client_id, "construction", 5, "初始化图构建器...")

        # 获取数据集路径
        corpus_path = f"data/uploaded/{dataset_name}/corpus.json"

        # 如果上传数据集不存在，尝试使用demo数据集
        if not os.path.exists(corpus_path):
            corpus_path = "data/demo/demo_corpus.json"

        # 如果连demo数据集也不存在，则抛出404错误
        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        # 发送进度更新：开始加载配置和语料库，进度10%
        await send_progress_update(client_id, "construction", 10, "加载配置和语料库...")

        # 初始化全局配置
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        # 根据配置动态选择schema，未指定则使用默认的demo.json
        # 获取目标数据集的 schema (如果 aviation_1 没有配置，就用 aviation 的)
        # 注意：这里我们优先尝试获取 target 的配置，因为它肯定存在
        dataset_config = config.get_dataset_config(dataset_name)
        if dataset_config:
            schema_path = dataset_config.schema_path
            logger.info(f"使用的模式文件: {schema_path}")
            user_input = input("是否继续？(y/n): ").lower().strip()
            if user_input == 'y':
                logger.info(f"用户确认使用认模式文件: {schema_path}")
            elif user_input == 'n':
                logger.info(f"用户取消使用默认模式文件: {schema_path}")
                raise HTTPException(status_code=400, detail=f"用户取消使用配置: {schema_path}")
            else:
                logger.warning(f"无效输入 '{user_input}'，默认取消操作")
                raise HTTPException(status_code=400, detail=f"无效输入，取消使用配置: {schema_path}")
        else:
            # 如果配置文件中没有对应配置，使用 demo.json 但需要用户确认
            schema_path = "schemas/demo.json"
            logger.info(f"使用的模式文件: {schema_path}")
            print(f"\n⚠️  警告：找不到数据集 '{target_dataset}' 的配置，将使用默认配置 'schemas/demo.json'")
            user_input = input("是否继续？(y/n): ").lower().strip()

            if user_input == 'y':
                logger.info(f"用户确认使用默认模式文件: {schema_path}")
            elif user_input == 'n':
                logger.info(f"用户取消使用默认模式文件: {schema_path}")
                raise HTTPException(status_code=400, detail=f"用户取消使用默认配置: {schema_path}")
            else:
                logger.warning(f"无效输入 '{user_input}'，默认取消操作")
                raise HTTPException(status_code=400, detail=f"无效输入，取消使用默认配置: {schema_path}")

        # 初始化KTBuilder图构建器
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )

        # 发送进度更新：开始实体关系抽取，进度20%
        await send_progress_update(client_id, "construction", 20, "开始实体关系抽取...")

        # 定义同步构建图谱的函数
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        # 获取事件循环，以便在执行器中运行同步函数
        loop = asyncio.get_event_loop()

        # 定义不同构建阶段的进度模拟
        stages = [
            (30, "抽取实体和关系中..."),
            (50, "社区检测中..."),
            (70, "构建层次结构中..."),
            (85, "优化图结构中..."),
        ]

        # 定义进度更新的异步函数
        async def update_progress():
            for progress, message in stages:
                await asyncio.sleep(3)  # 模拟工作时间
                await send_progress_update(client_id, "construction", progress, message)

        # 同时运行图谱构建和进度更新
        progress_task = asyncio.create_task(update_progress())

        try:
            # 在执行器中运行图谱构建函数，避免阻塞主线程
            knowledge_graph = await loop.run_in_executor(None, build_graph_sync)
            # 构建完成后取消进度更新任务
            progress_task.cancel()
        except Exception as e:
            progress_task.cancel()
            raise e

        # 发送进度更新：准备可视化数据，进度95%
        await send_progress_update(client_id, "construction", 95, "准备可视化数据...")

        # 加载构建好的图谱用于可视化
        graph_path = f"output/graphs/{dataset_name}_new.json"
        graph_vis_data = await prepare_graph_visualization(graph_path)

        # 发送进度更新：图构建完成，进度100%
        await send_progress_update(client_id, "construction", 100, "图构建完成!")

        return GraphConstructionResponse(
            success=True,
            message="Knowledge graph constructed successfully",
            graph_data=graph_vis_data
        )

    except Exception as e:
        await send_progress_update(client_id, "construction", 0, f"构建失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/construct-graph-incremental", response_model=GraphConstructionResponse)
async def construct_graph_incremental(request: GraphConstructionIncrementalRequest, client_id: str = "default"):
    """
    增量构建接口：保留旧数据，使用增量 Prompt，支持指定目标数据集
    """
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG unavailable")

        # 1. 确定源数据和目标数据
        source_dataset = request.dataset_name  # 例如 "aviation_1"
        # 如果前端没传 target，就默认是 source（兼容旧逻辑）
        target_dataset = request.target_dataset_name or source_dataset

        # 检查源数据集和目标数据集是否相同
        if source_dataset == target_dataset:
            raise HTTPException(
                status_code=400,
                detail=f"源数据集 '{source_dataset}' 和目标数据集 '{target_dataset}' 不能相同，这可能导致数据覆盖或无限循环"
            )

        logger.info(f"🔄 增量任务: 源数据[{source_dataset}] -> 合并入 -> 目标图谱[{target_dataset}]")

        await send_progress_update(client_id, "construction", 5, "🚀 启动增量构建 (热加载中)...")
        # 2. 确定语料路径 (使用源数据的语料)
        corpus_path = f"data/uploaded/{source_dataset}/corpus.json"
        if not os.path.exists(corpus_path):
            # 尝试回退逻辑
            if source_dataset == "demo": corpus_path = "data/demo/demo_corpus.json"

        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail=f"Source corpus not found: {corpus_path}")

        # 发送进度更新：开始加载配置和语料库，进度10%
        await send_progress_update(client_id, "construction", 10, "加载配置和语料库...")

        # 初始化全局配置
        # 3. 初始化构建器 (使用【目标】数据集的配置和图谱)
        # 这样 KTBuilder 会去加载 target_dataset_new.json
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        # 根据配置动态选择schema，未指定则使用默认的demo.json
        # 获取目标数据集的 schema (如果 aviation_1 没有配置，就用 aviation 的)
        # 注意：这里我们优先尝试获取 target 的配置，因为它肯定存在
        dataset_config = config.get_dataset_config(target_dataset)
        if dataset_config:
            schema_path = dataset_config.schema_path
            logger.info(f"使用的模式文件: {schema_path}")
            user_input = input("是否继续？(y/n): ").lower().strip()
            if user_input == 'y':
                logger.info(f"用户确认使用认模式文件: {schema_path}")
            elif user_input == 'n':
                logger.info(f"用户取消使用默认模式文件: {schema_path}")
                raise HTTPException(status_code=400, detail=f"用户取消使用配置: {schema_path}")
            else:
                logger.warning(f"无效输入 '{user_input}'，默认取消操作")
                raise HTTPException(status_code=400, detail=f"无效输入，取消使用配置: {schema_path}")
        else:
            # 如果配置文件中没有对应配置，使用 demo.json 但需要用户确认
            schema_path = "schemas/demo.json"
            logger.info(f"使用的模式文件: {schema_path}")
            print(f"\n⚠️  警告：找不到数据集 '{target_dataset}' 的配置，将使用默认配置 'schemas/demo.json'")
            user_input = input("是否继续？(y/n): ").lower().strip()

            if user_input == 'y':
                logger.info(f"用户确认使用默认模式文件: {schema_path}")
            elif user_input == 'n':
                logger.info(f"用户取消使用默认模式文件: {schema_path}")
                raise HTTPException(status_code=400, detail=f"用户取消使用默认配置: {schema_path}")
            else:
                logger.warning(f"无效输入 '{user_input}'，默认取消操作")
                raise HTTPException(status_code=400, detail=f"无效输入，取消使用默认配置: {schema_path}")



        # ⚠️ 关键点：初始化 Builder 时用 target_dataset
        builder = constructor.KTBuilder(
            target_dataset,  # <--- 名字传 Target (aviation)
            schema_path,
            mode=config.construction.mode,
            config=config,
            is_incremental=True
        )

        await send_progress_update(client_id, "construction", 20, f"开始从 {source_dataset} 增量抽取...")

        # 4. 执行构建 (传入源数据的语料路径)
        def build_graph_sync():
            # ⚠️ 关键点：构建时用 source corpus
            return builder.build_knowledge_graph(corpus_path)

        # 获取事件循环，以便在执行器中运行同步函数
        loop = asyncio.get_event_loop()

        # 定义不同构建阶段的进度模拟
        stages = [
            (30, "抽取实体和关系中..."),
            (50, "社区检测中..."),
            (70, "构建层次结构中..."),
            (85, "优化图结构中..."),
        ]

        # 定义进度更新的异步函数
        async def update_progress():
            for progress, message in stages:
                await asyncio.sleep(3)  # 模拟工作时间
                await send_progress_update(client_id, "construction", progress, message)

        # 同时运行图谱构建和进度更新
        progress_task = asyncio.create_task(update_progress())

        try:
            # 在执行器中运行图谱构建函数，避免阻塞主线程
            await loop.run_in_executor(None, build_graph_sync)
            # 构建完成后取消进度更新任务
            progress_task.cancel()
        except Exception as e:
            progress_task.cancel()
            raise e

        # 发送进度更新：准备可视化数据，进度95%
        await send_progress_update(client_id, "construction", 95, "准备可视化数据...")

        # 加载构建好的图谱用于可视化
        graph_path = f"output/graphs/{target_dataset}_new.json"
        graph_vis_data = await prepare_graph_visualization(graph_path)

        # 发送进度更新：图构建完成，进度100%
        await send_progress_update(client_id, "construction", 100, "图构建完成!")

        return GraphConstructionResponse(
            success=True,
            message=f"Merged {source_dataset} into {target_dataset}",
            graph_data=graph_vis_data
        )
    except Exception as e:
        await send_progress_update(client_id, "construction", 0, f"构建失败: {str(e)}")
        logger.error(f"构建失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 加载并解析图数据文件
# 参数 graph_path: 图谱数据文件的路径
# 返回值: 包含节点、连线、分类和统计数据的字典，供ECharts等可视化库使用
async def prepare_graph_visualization(graph_path: str) -> Dict:
    """准备可视化图数据"""
    try:
        # 检查图谱文件是否存在
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
        else:
            # 如果文件不存在，返回空的可视化数据结构
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

        # 根据不同的图谱数据格式进行处理
        if isinstance(graph_data, list):
            # 如果数据是列表格式，表示GraphRAG的关系列表格式
            return convert_graphrag_format(graph_data)
        elif isinstance(graph_data, dict) and "nodes" in graph_data:
            # 如果数据是字典格式且包含"nodes"键，表示标准的图谱格式{nodes: [], edges: []}
            return convert_standard_format(graph_data)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return {"nodes": [], "links": [], "categories": [], "stats": {}}


def convert_graphrag_format(graph_data: List) -> Dict:
    """
    将GraphRAG格式的数据转换为ECharts可视化所需的格式。

    参数:
        graph_data (List): GraphRAG格式的图数据，是一个包含多个关系项的列表。
                          每个关系项通常包含起始节点、结束节点和关系信息。

    返回:
        Dict: 符合ECharts要求的图数据格式，包含节点(nodes)、连线(links)、
              分类(categories)和统计信息(stats)等字段。
    """

    # 初始化节点字典和连线列表
    # nodes_dict: 用于存储去重后的节点，键为节点ID，值为节点信息
    # links: 存储节点之间的关系连线
    nodes_dict = {}
    links = []

    # 遍历GraphRAG格式的图数据中的每一项关系
    for item in graph_data:
        if not isinstance(item, dict):
            continue

        # 提取关系的起始节点、结束节点和关系类型
        start_node = item.get("start_node", {})
        end_node = item.get("end_node", {})
        relation = item.get("relation", "related_to")

        # 处理起始节点信息
        start_id = ""
        end_id = ""
        if start_node:
            start_id = start_node.get("properties", {}).get("name", "")
            if start_id and start_id not in nodes_dict:
                nodes_dict[start_id] = {
                    "id": start_id,
                    "name": start_id[:30],
                    "category": start_node.get("properties", {}).get("schema_type", start_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": start_node.get("properties", {})
                }

        # 处理结束节点信息（与起始节点类似）
        if end_node:
            end_id = end_node.get("properties", {}).get("name", "")
            if end_id and end_id not in nodes_dict:
                nodes_dict[end_id] = {
                    "id": end_id,
                    "name": end_id[:30],
                    "category": end_node.get("properties", {}).get("schema_type", end_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": end_node.get("properties", {})
                }

        # 如果起始节点和结束节点都存在，则建立它们之间的关系连线
        if start_id and end_id:
            links.append({
                "source": start_id,
                "target": end_id,
                "name": relation,
                "value": 1
            })

    # 创建节点分类信息
    # 从所有节点中提取唯一的分类名称
    categories_set = set()
    for node in nodes_dict.values():
        categories_set.add(node["category"])

    # 为每个分类分配颜色，使用HSL色彩空间确保颜色区分度
    categories = []
    for i, cat_name in enumerate(categories_set):
        categories.append({
            "name": cat_name,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(categories_set)}, 70%, 60%)"
            }
        })

    nodes = list(nodes_dict.values())

    # 返回符合ECharts要求的图数据结构
    return {
        "nodes": nodes[:500],  # 限制节点数量以提升可视化效果
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }


def convert_standard_format(graph_data: Dict) -> Dict:
    """将标准格式 {nodes: [], edges: []} 转换为 ECharts 格式"""
    # 初始化节点、连线和分类列表，用于存储转换后的数据
    nodes = []
    links = []
    categories = []

    # 提取唯一的节点类型作为分类
    node_types = set()
    for node in graph_data.get("nodes", []):
        node_type = node.get("type", "entity")
        node_types.add(node_type)

    # 为每个节点类型创建分类，并分配不同颜色
    for i, node_type in enumerate(node_types):
        categories.append({
            "name": node_type,
            "itemStyle": {
                # 使用HSL色彩空间为不同分类分配颜色，确保视觉区分度
                # 色相值根据分类数量均匀分布，饱和度70%，亮度60%
                "color": f"hsl({i * 360 / len(node_types)}, 70%, 60%)"
            }
        })

    # 处理节点数据，将原始节点转换为ECharts格式
    for node in graph_data.get("nodes", []):
        nodes.append({
            "id": node.get("id", ""),
            "name": node.get("name", node.get("id", ""))[:30],
            "category": node.get("type", "entity"),
            "value": len(node.get("attributes", [])),
            "symbolSize": min(max(len(node.get("attributes", [])) * 3 + 15, 15), 40),
            "attributes": node.get("attributes", [])
        })

    # 处理边数据，将原始边转换为ECharts格式
    for edge in graph_data.get("edges", []):
        links.append({
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "name": edge.get("relation", "related_to"),
            "value": edge.get("weight", 1)
        })

    return {
        "nodes": nodes[:500],  # Limit for performance
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }


# 实现智能问答功能，采用IRCoT(Iterative Retrieval with Chain-of-Thought)
@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, client_id: str = "default"):
    """Process question using agent mode (iterative retrieval + reasoning) and return answer."""
    try:
        # 检查 GraphRAG 组件是否可用
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")

        # 获取数据集名称和问题内容
        dataset_name = request.dataset_name
        question = request.question
        logger.info(f"处理问题: {question}")
        logger.info(f"数据集名称: {dataset_name}")
        logger.info(f"alpha:{request.alpha},beta:{request.beta}")

        # 发送进度更新到客户端
        await send_progress_update(client_id, "retrieval", 10, "初始化检索系统 (agent 模式)...")

        # 确定图谱文件路径
        graph_path = f"output/graphs/{dataset_name}_new.json"

        if not os.path.exists(graph_path):
            graph_path = "output/graphs/demo_new.json"
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")

        # 初始化配置和组件
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        # 根据配置动态选择schema，未指定则使用默认的demo.json
        schema_path = config.get_dataset_config(dataset_name).schema_path if config else "schemas/demo.json"
        logger.info(f"使用的模式文件: {schema_path}")

        # 初始化问题分解器和检索器
        graphq = decomposer.GraphQ(dataset_name, config=config)
        logger.info("问题分解器初始化完成")
        kt_retriever = retriever.KTRetriever(
            dataset_name,
            graph_path,
            recall_paths=config.retrieval.recall_paths,
            schema_path=schema_path,
            top_k=config.retrieval.top_k_filter,
            mode="agent",  # 强制 agent 模式
            config=config
        )
        logger.info("检索器初始化完成")

        await send_progress_update(client_id, "retrieval", 40, "构建索引...")
        # 构建检索索引
        logger.info("开始构建索引")
        kt_retriever.build_indices()
        logger.info("索引构建完成")

        # Helper functions (复用 main.py 逻辑的精简版)
        def _dedup(items):
            return list({x: None for x in items}.keys())

        def _merge_chunk_contents(ids, mapping):
            return [mapping.get(i, f"[Missing content for chunk {i}]") for i in ids]

        # Step 1: decomposition
        await send_progress_update(client_id, "retrieval", 50, "问题分解...")
        logger.info("开始进行问题分解")
        try:
            # 使用 GraphQ 进行问题分解
            decomposition = graphq.decompose(question, schema_path)
            sub_questions = decomposition.get("sub_questions", [])
            involved_types = decomposition.get("involved_types", {})
        except Exception as e:
            logger.error(f"Decompose failed: {e}")
            sub_questions = [{"sub-question": question}]
            involved_types = {"nodes": [], "relations": [], "attributes": []}
        logger.info("问题分解完成")

        reasoning_steps = []
        all_triples = set()
        all_chunk_ids = set()
        all_chunk_contents: Dict[str, str] = {}

        # Step 2: initial retrieval for each sub-question
        await send_progress_update(client_id, "retrieval", 65, "初始检索...")
        import time as _time
        logger.info("开始进行初始检索")
        # 对每个子问题进行初步检索
        for idx, sq in enumerate(sub_questions):
            sq_text = sq.get("sub-question", question)

            logger.info(f"开始处理检索结果，子问题 {idx + 1}/{len(sub_questions)}: {sq_text}")
            retrieval_results, elapsed = kt_retriever.process_retrieval_results(
                sq_text,
                top_k=config.retrieval.top_k_filter,
                involved_types=involved_types,
                alpha=request.alpha,  # 传入
                beta=request.beta  # 传入
            )
            logger.info(f"检索完成")
            # 收集检索到的三元组和文本块
            triples = retrieval_results.get('triples', []) or []
            # logger.info(f"检索到三元组：{triples}")
            chunk_ids = retrieval_results.get('chunk_ids', []) or []
            chunk_contents = retrieval_results.get('chunk_contents', []) or []
            if isinstance(chunk_contents, dict):
                for cid, ctext in chunk_contents.items():
                    all_chunk_contents[cid] = ctext
            else:
                for i_c, cid in enumerate(chunk_ids):
                    if i_c < len(chunk_contents):
                        all_chunk_contents[cid] = chunk_contents[i_c]
            all_triples.update(triples)
            # logger.info(f"全部三元组：{all_triples}")
            all_chunk_ids.update(chunk_ids)
            reasoning_steps.append({
                "type": "sub_question",
                "question": sq_text,
                "triples": triples[:10],
                "triples_count": len(triples),
                "chunks_count": len(chunk_ids),
                "processing_time": elapsed,
                "chunk_contents": list(all_chunk_contents.values())[:3]
            })

        # Step 3: IRCoT iterative refinement
        await send_progress_update(client_id, "retrieval", 75, "迭代推理...")
        logger.info("开始进行迭代推理")
        # 进行多轮迭代推理，最多3轮
        max_steps = getattr(getattr(config.retrieval, 'agent', object()), 'max_steps', 3)
        current_query = question
        thoughts = []

        # Initial answer attempt
        # todo:感觉排序有问题
        initial_triples = _dedup(list(all_triples))
        initial_chunk_ids = list(set(all_chunk_ids))
        initial_chunk_contents = _merge_chunk_contents(initial_chunk_ids, all_chunk_contents)
        context_initial = "=== Triples ===\n" + "\n".join(initial_triples[:20]) + "\n=== Chunks ===\n" + "\n".join(
            initial_chunk_contents[:10])
        init_prompt = kt_retriever.generate_prompt(question, context_initial)
        try:
            initial_answer = kt_retriever.generate_answer(init_prompt)
        except Exception as e:
            initial_answer = f"Initial answer failed: {e}"
        thoughts.append(f"Initial: {initial_answer[:200]}")
        final_answer = initial_answer

        import re as _re
        for step in range(1, max_steps + 1):
            loop_triples = _dedup(list(all_triples))
            loop_chunk_ids = list(set(all_chunk_ids))
            loop_chunk_contents = _merge_chunk_contents(loop_chunk_ids, all_chunk_contents)
            # 构建当前上下文
            loop_ctx = "=== Triples ===\n" + "\n".join(loop_triples[:20]) + "\n=== Chunks ===\n" + "\n".join(
                loop_chunk_contents[:10])
            # 生成推理提示词
            loop_prompt = f"""
You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.
Current Question: {question}
Current Iteration Query: {current_query}
Knowledge Context:\n{loop_ctx}
Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}
Instructions:
1. If enough info answer with: So the answer is: <answer>
2. Else propose new query with: The new query is: <query>
Your reasoning:
"""
            try:
                # 生成推理结果
                reasoning = kt_retriever.generate_answer(loop_prompt)
            except Exception as e:
                reasoning = f"Reasoning error: {e}"
            thoughts.append(reasoning[:400])
            reasoning_steps.append({
                "type": "ircot_step",
                "question": current_query,
                "triples": loop_triples[:10],
                "triples_count": len(loop_triples),
                "chunks_count": len(loop_chunk_ids),
                "processing_time": 0,
                "chunk_contents": loop_chunk_contents[:3],
                "thought": reasoning[:300]
            })
            # 根据推理结果判断是否需要继续迭代或输出答案
            if "So the answer is:" in reasoning:
                m = _re.search(r"So the answer is:\s*(.*)", reasoning, flags=_re.IGNORECASE | _re.DOTALL)
                # 提取最终答案
                final_answer = m.group(1).strip() if m else reasoning
                break
            if "The new query is:" not in reasoning:
                final_answer = initial_answer or reasoning
                break
            # 提取新查询，继续下一轮迭代
            new_query = reasoning.split("The new query is:", 1)[1].strip().splitlines()[0]
            if not new_query or new_query == current_query:
                final_answer = initial_answer or reasoning
                break
            current_query = new_query
            await send_progress_update(client_id, "retrieval", min(90, 75 + step * 5), f"迭代检索 Step {step}...")
            try:
                new_ret, _ = kt_retriever.process_retrieval_results(current_query, top_k=config.retrieval.top_k_filter)
                new_triples = new_ret.get('triples', []) or []
                new_chunk_ids = new_ret.get('chunk_ids', []) or []
                new_chunk_contents = new_ret.get('chunk_contents', []) or []
                if isinstance(new_chunk_contents, dict):
                    for cid, ctext in new_chunk_contents.items():
                        all_chunk_contents[cid] = ctext
                else:
                    for i_c, cid in enumerate(new_chunk_ids):
                        if i_c < len(new_chunk_contents):
                            all_chunk_contents[cid] = new_chunk_contents[i_c]
                all_triples.update(new_triples)
                all_chunk_ids.update(new_chunk_ids)
            except Exception as e:
                logger.error(f"Iterative retrieval failed: {e}")
                break

        # Final aggregation
        final_triples = _dedup(list(all_triples))[:20]
        final_chunk_ids = list(set(all_chunk_ids))
        final_chunk_contents = _merge_chunk_contents(final_chunk_ids, all_chunk_contents)[:10]

        await send_progress_update(client_id, "retrieval", 100, "答案生成完成!")

        # 准备可视化数据
        visualization_data = {
            "subqueries": prepare_subquery_visualization(sub_questions, reasoning_steps),
            "knowledge_graph": prepare_retrieved_graph_visualization(final_triples),
            "reasoning_flow": prepare_reasoning_flow_visualization(reasoning_steps),
            "retrieval_details": {
                "total_triples": len(final_triples),
                "total_chunks": len(final_chunk_contents),
                "sub_questions_count": len(sub_questions),
                "triples_by_subquery": [s.get("triples_count", 0) for s in reasoning_steps if
                                        s.get("type") == "sub_question"]
            }
        }

        logger.info(f"最终答案: {final_answer}")
        logger.info(f"子问题: {sub_questions}")
        logger.info(f"最终三元组: {final_triples}")
        logger.info(f"最终文本块: {final_chunk_contents}")

        return QuestionResponse(
            answer=final_answer,
            sub_questions=sub_questions,
            retrieved_triples=final_triples,
            retrieved_chunks=final_chunk_contents,
            reasoning_steps=reasoning_steps,
            visualization_data=visualization_data
        )
    except Exception as e:
        await send_progress_update(client_id, "retrieval", 0, f"问答处理失败: {str(e)}")
        logger.error(f"处理问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def prepare_subquery_visualization(sub_questions: List[Dict], reasoning_steps: List[Dict]) -> Dict:
    """子问题分解可视化"""
    # 初始化节点列表，添加原始问题节点
    # 节点包含ID、名称、分类和符号大小等属性
    nodes = [{"id": "original", "name": "原始问题", "category": "question", "symbolSize": 40}]
    # 初始化连接线列表，用于表示节点间的关系
    links = []

    # 遍历所有子问题，为每个子问题创建可视化节点
    for i, sub_q in enumerate(sub_questions):
        # 为每个子问题生成唯一ID
        sub_id = f"sub_{i}"
        # 添加子问题节点到节点列表
        nodes.append({
            "id": sub_id,
            "name": sub_q.get("sub-question", "")[:20] + "...",
            "category": "sub_question",
            "symbolSize": 30
        })
        # 添加从原始问题到子问题的连接线，表示问题分解关系
        links.append({"source": "original", "target": sub_id, "name": "分解为"})

    return {
        "nodes": nodes,
        "links": links,
        "categories": [
            {"name": "question", "itemStyle": {"color": "#ff6b6b"}},
            {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}}
        ]
    }


def prepare_retrieved_graph_visualization(triples: List[str]) -> Dict:
    """检索到的知识图谱可视化"""
    # 初始化节点、连线和节点集合
    nodes = []
    links = []
    node_set = set()

    # 遍历前10个三元组（限制数量以提高可视化性能）
    for triple in triples[:10]:
        try:
            if triple.startswith('[') and triple.endswith(']'):
                try:
                    parts = ast.literal_eval(triple)
                except Exception:
                    continue
                # 确保三元组包含3个部分（主体、关系、客体）
                if len(parts) == 3:
                    source, relation, target = parts

                    for entity in [source, target]:
                        # 避免重复添加相同节点
                        if entity not in node_set:
                            node_set.add(entity)
                            # 添加节点信息到nodes列表
                            nodes.append({
                                "id": str(entity),
                                "name": str(entity)[:20],
                                "category": "entity",
                                "symbolSize": 20
                            })

                    # 添加关系连线信息到links列表
                    links.append({
                        "source": str(source),
                        "target": str(target),
                        "name": str(relation)
                    })
        except:
            continue

    return {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": "entity", "itemStyle": {"color": "#95de64"}}]
    }


def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict]) -> Dict:
    """推理流程可视化"""
    steps_data = []
    for i, step in enumerate(reasoning_steps):
        steps_data.append({
            "step": i + 1,
            "type": step.get("type", "unknown"),
            "question": step.get("question", "")[:50],
            "triples_count": step.get("triples_count", 0),
            "chunks_count": step.get("chunks_count", 0),
            "processing_time": step.get("processing_time", 0)
        })

    return {
        "steps": steps_data,
        "timeline": [step["processing_time"] for step in steps_data]
    }


@app.get("/api/datasets")
async def get_datasets():
    """获取可用数据集列表"""
    datasets = []

    # 检查已上传数据集
    upload_dir = "data/uploaded"
    if os.path.exists(upload_dir):
        for item in os.listdir(upload_dir):
            item_path = os.path.join(upload_dir, item)
            if os.path.isdir(item_path):
                corpus_path = os.path.join(item_path, "corpus.json")
                if os.path.exists(corpus_path):
                    graph_path = f"output/graphs/{item}_new.json"
                    status = "ready" if os.path.exists(graph_path) else "needs_construction"
                    datasets.append({
                        "name": item,
                        "type": "uploaded",
                        "status": status
                    })

    # 加入demo数据集
    demo_corpus = "data/demo/demo_corpus.json"
    if os.path.exists(demo_corpus):
        demo_graph = "output/graphs/demo_new.json"
        status = "ready" if os.path.exists(demo_graph) else "needs_construction"
        datasets.append({
            "name": "demo",
            "type": "demo",
            "status": status
        })

    return {"datasets": datasets}


@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """删除指定数据集及其关联文件"""
    try:
        #  禁止删除demo数据集，因为它是系统内置的示例数据集
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot delete demo dataset")

        #  创建一个列表来记录所有被删除的文件和目录路径
        deleted_files = []

        # 删除数据集目录（包含上传的原始文件和corpus.json）
        dataset_dir = f"data/uploaded/{dataset_name}"
        if os.path.exists(dataset_dir):
            import shutil
            # 递归删除整个目录树
            shutil.rmtree(dataset_dir)
            # 记录被删除的目录路径
            deleted_files.append(dataset_dir)

        # 删除生成的知识图谱文件
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            # 记录被删除的图谱文件路径
            deleted_files.append(graph_path)

        # 删除数据集特定的模式文件（如果存在）
        # schema_path = f"schemas/{dataset_name}.json"
        # if os.path.exists(schema_path):
        #     os.remove(schema_path)
        #     deleted_files.append(schema_path)

        # 删除FAISS检索缓存目录
        cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            deleted_files.append(cache_dir)

        # 删除文本分块文件
        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            deleted_files.append(chunk_file)

        return {
            "success": True,
            "message": f"Dataset '{dataset_name}' deleted successfully",
            "deleted_files": deleted_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


# 路径参数 dataset_name: 数据集名称
# 查询参数 client_id: 客户端ID，用于发送进度更新
@app.post("/api/datasets/{dataset_name}/reconstruct")
async def reconstruct_dataset(dataset_name: str, client_id: str = "default"):
    """重新构建指定数据集的图谱"""
    try:
        # 检查GraphRAG组件是否可用
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")

        # 检查数据集是否存在
        corpus_path = f"data/uploaded/{dataset_name}/corpus.json"
        if not os.path.exists(corpus_path):
            if dataset_name == "demo":
                corpus_path = "data/demo/demo_corpus.json"
            else:
                raise HTTPException(status_code=404, detail="Dataset not found")

        # 发送进度更新：开始重新构图，进度5%
        await send_progress_update(client_id, "reconstruction", 5, "开始重新构图...")

        # 删除现有图谱文件
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            await send_progress_update(client_id, "reconstruction", 15, "已删除旧图谱文件...")

        # 删除现有的缓存文件
        cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            await send_progress_update(client_id, "reconstruction", 25, "已清理缓存文件...")

        # 删除现有的文本文件
        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

        await send_progress_update(client_id, "reconstruction", 35, "重新初始化图构建器...")

        # 初始化配置
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        # 根据配置动态选择schema，未指定则使用默认的demo.json
        schema_path = config.get_dataset_config(dataset_name).schema_path if config else "schemas/demo.json"
        logger.info(f"使用的模式文件: {schema_path}")

        # 初始化KTBuilder图构建器
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )

        await send_progress_update(client_id, "reconstruction", 50, "开始重新构建图谱...")

        # 定义同步构建图谱的函数
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        # 获取事件循环，以便在执行器中运行同步函数
        loop = asyncio.get_event_loop()

        # 定义不同阶段的进度模拟
        stages = [
            (65, "重新抽取实体和关系中..."),
            (80, "重新进行社区检测中..."),
            (90, "重新构建层次结构中..."),
            (95, "重新优化图结构中..."),
        ]

        # 定义进度更新的异步函数
        async def update_progress():
            for progress, message in stages:
                await asyncio.sleep(2)  # Simulate work time
                await send_progress_update(client_id, "reconstruction", progress, message)

        # 同时运行图谱构建和进度更新
        progress_task = asyncio.create_task(update_progress())

        try:
            # 在执行器中运行图谱构建函数，避免阻塞
            knowledge_graph = await loop.run_in_executor(None, build_graph_sync)
            # 构建完成后取消进度更新任务
            progress_task.cancel()
        except Exception as e:
            progress_task.cancel()
            raise e

        # 发送进度更新：图谱重构完成，进度100%
        await send_progress_update(client_id, "reconstruction", 100, "图谱重构完成!")

        return {
            "success": True,
            "message": "Dataset reconstructed successfully",
            "dataset_name": dataset_name
        }

    except Exception as e:
        await send_progress_update(client_id, "reconstruction", 0, f"重构失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 获取指定数据集的图谱可视化数据，如果文件不存在则返回示例数据。
@app.get("/api/graph/{dataset_name}")
async def get_graph_data(dataset_name: str):
    """获取图谱可视化数据"""
    graph_path = f"output/graphs/{dataset_name}_new.json"

    if not os.path.exists(graph_path):
        # 文件不存在，返回示例数据
        return {
            "nodes": [
                {"id": "node1", "name": "示例实体1", "category": "person", "value": 5, "symbolSize": 25},
                {"id": "node2", "name": "示例实体2", "category": "location", "value": 3, "symbolSize": 20},
            ],
            "links": [
                {"source": "node1", "target": "node2", "name": "位于", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {"total_nodes": 2, "total_edges": 1, "displayed_nodes": 2, "displayed_edges": 1}
        }

    return await prepare_graph_visualization(graph_path)


# 添加复制数据集的接口
@app.post("/api/datasets/{source_dataset_name}/copy")
async def copy_dataset(source_dataset_name: str, request: Dict[str, str]):
    """
    复制数据集接口
    参数:
        source_dataset_name: 源数据集名称
        request: 包含目标数据集名称的请求体
    """
    try:
        # 获取目标数据集名称
        target_dataset_name = request.get("target_dataset_name")

        if not target_dataset_name:
            raise HTTPException(status_code=400, detail="Target dataset name is required")

        # 检查源数据集是否存在
        source_dir = f"data/uploaded/{source_dataset_name}"
        if not os.path.exists(source_dir) and source_dataset_name != "demo":
            raise HTTPException(status_code=404, detail=f"Source dataset '{source_dataset_name}' not found")

        # 不许复制demo数据集
        if source_dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot copy demo dataset")

        # 检查目标数据集是否已存在
        target_dir = f"data/uploaded/{target_dataset_name}"
        if os.path.exists(target_dir):
            raise HTTPException(status_code=409, detail=f"Target dataset '{target_dataset_name}' already exists")

        # 创建目标目录
        os.makedirs(target_dir, exist_ok=True)

        # 复制上传的数据集文件
        source_corpus = f"{source_dir}/corpus.json"
        if os.path.exists(source_corpus):
            shutil.copy2(source_corpus, f"{target_dir}/corpus.json")

        # 复制 schema 文件（如果存在）
        source_schema = f"schemas/{source_dataset_name}.json"
        target_schema = f"schemas/{target_dataset_name}.json"
        if os.path.exists(source_schema):
            shutil.copy2(source_schema, target_schema)

        # 复制生成的图谱文件（如果存在）
        source_graph = f"output/graphs/{source_dataset_name}_new.json"
        target_graph = f"output/graphs/{target_dataset_name}_new.json"
        if os.path.exists(source_graph):
            shutil.copy2(source_graph, target_graph)

        # 复制文本块文件（如果存在）
        source_chunk = f"output/chunks/{source_dataset_name}.txt"
        target_chunk = f"output/chunks/{target_dataset_name}.txt"
        if os.path.exists(source_chunk):
            shutil.copy2(source_chunk, target_chunk)

        # 复制 FAISS 缓存目录（如果存在）
        source_cache_dir = f"retriever/faiss_cache_new/{source_dataset_name}"
        target_cache_dir = f"retriever/faiss_cache_new/{target_dataset_name}"
        if os.path.exists(source_cache_dir):
            shutil.copytree(source_cache_dir, target_cache_dir)

        logger.info(f"Dataset '{source_dataset_name}' copied to '{target_dataset_name}' successfully")

        return {
            "success": True,
            "message": f"Dataset '{source_dataset_name}' copied to '{target_dataset_name}' successfully",
            "source_dataset": source_dataset_name,
            "target_dataset": target_dataset_name
        }

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        logger.error(f"Error copying dataset '{source_dataset_name}' to '{target_dataset_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to copy dataset: {str(e)}")


@app.put("/api/datasets/{dataset_name}/rename")
async def rename_dataset(dataset_name: str, request: Dict[str, str]):
    """
   重命名数据集接口
   参数:
       dataset_name: 原数据集名称
       request: 包含新数据集名称的请求体
   """
    try:
        # 获取新数据集名称
        new_dataset_name = request.get("new_dataset_name")

        if not new_dataset_name:
            raise HTTPException(status_code=400, detail="需要新数据集名称")

        # 验证新名称格式
        if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9 _-]+$', new_dataset_name):
            raise HTTPException(
                status_code=400,
                detail="新数据集名称格式错误，请使用中文、英文、数字、下划线、中划线、空格"
            )

        # 不允许重命名demo数据集
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="不能重命名demo数据集")

        # 检查原数据集是否存在
        source_dir = f"data/uploaded/{dataset_name}"
        if not os.path.exists(source_dir):
            raise HTTPException(status_code=404, detail=f"源数据集 '{dataset_name}' 不存在")

        # 检查新数据集名称是否已存在
        target_dir = f"data/uploaded/{new_dataset_name}"
        if os.path.exists(target_dir):
            raise HTTPException(status_code=409, detail=f"目标数据集 '{new_dataset_name}' 已经存在")

        # 检查新名称的schema文件是否已存在
        target_schema = f"schemas/{new_dataset_name}.json"
        if os.path.exists(target_schema):
            raise HTTPException(status_code=409,
                                detail=f"目标数据集 '{new_dataset_name}' 的schema文件已经存在")

        # 检查新名称的图谱文件是否已存在
        target_graph = f"output/graphs/{new_dataset_name}_new.json"
        if os.path.exists(target_graph):
            raise HTTPException(status_code=409,
                                detail=f"图谱文件 '{new_dataset_name}' 已经存在")

        # 检查新名称的chunk文件是否已存在
        target_chunk = f"output/chunks/{new_dataset_name}.txt"
        if os.path.exists(target_chunk):
            raise HTTPException(status_code=409,
                                detail=f"文本块文件 '{new_dataset_name}' 已经存在")

        # 执行重命名操作
        # 1. 重命名上传的数据集目录
        os.rename(source_dir, target_dir)

        # 3. 重命名图谱文件（如果存在）
        source_graph = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(source_graph):
            shutil.move(source_graph, target_graph)

        # 4. 重命名chunk文件（如果存在）
        source_chunk = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(source_chunk):
            shutil.move(source_chunk, target_chunk)

        # 5. 重命名FAISS缓存目录（如果存在）
        source_faiss_cache = f"retriever/faiss_cache_new/{dataset_name}"
        target_faiss_cache = f"retriever/faiss_cache_new/{new_dataset_name}"
        if os.path.exists(source_faiss_cache):
            shutil.move(source_faiss_cache, target_faiss_cache)

        logger.info(f"Dataset '{dataset_name}' renamed to '{new_dataset_name}' successfully")

        return {
            "success": True,
            "message": f"Dataset '{dataset_name}' renamed to '{new_dataset_name}' successfully"
        }

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise

    except Exception as e:
        logger.error(f"Error renaming dataset '{dataset_name}' to '{new_dataset_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rename dataset: {str(e)}")


# 应用启动时创建必要的目录结构，并通过Uvicorn启动FastAPI应用服务
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    os.makedirs("data/uploaded", exist_ok=True)
    os.makedirs("output/graphs", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    os.makedirs("schemas", exist_ok=True)

    logger.info("后端服务启动成功")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)
