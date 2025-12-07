import os
import time
import json
import requests

from openai import OpenAI
from dotenv import load_dotenv

from utils.logger import logger

# 加载环境变量配置
load_dotenv()

class LLMCompletionCall:
    def __init__(self):
        """
       初始化LLM客户端配置
       从环境变量中读取模型参数，如果没有设置则使用默认值
       """
        # 获取LLM模型名称，默认为"deepseek-chat"
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        # 获取LLM服务基础URL，默认为DeepSeek官方API地址
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        # 获取API密钥，如果未提供则抛出错误
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        if not self.llm_api_key:
            raise ValueError("LLM API key not provided")
        # 初始化OpenAI客户端实例
        self.client = OpenAI(base_url=self.llm_base_url, api_key = self.llm_api_key)

    def call_api(self, content: str) -> str:
        """
        调用LLM API生成文本，包含重试机制

        Args:
            content: 提示词内容

        Returns:
            生成的文本响应

        Raises:
            Exception: 当API调用失败时抛出异常
        """
            
        try:
            # 创建聊天补全请求
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.3
            )
            # 清理响应内容，去除可能的代码块标记和json标识符
            clean_completion = completion.choices[0].message.content.strip("```").strip("json")
            return clean_completion
            
        except Exception as e:
            logger.error(f"LLM api calling failed. Error: {e}")
            raise e 