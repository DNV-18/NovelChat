import os
import requests
from functools import lru_cache
from typing import Any, List, Tuple

from src.config import settings

class APIEmbeddingModel:
    def __init__(self, base_url, api_key, model_name):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint = f"{self.base_url}/embeddings"

    def encode(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": texts
        }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

class APIRerankerModel:
    def __init__(self, base_url, api_key, model_name):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

        if self.base_url.endswith("/v1"):
            self.endpoint = f"{self.base_url}/rerank"
        else:
            self.endpoint = f"{self.base_url}/v1/rerank"

    def predict(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        scores = []
        for q, doc in sentence_pairs:
            payload = {
                "model": self.model_name,
                "query": q,
                "documents": [doc]
            }
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            scores.append(data["results"][0]["relevance_score"])
        return scores

class ModelFactory:
    """
    统一的模型工厂类，负责分发系统所需的各类 AI 模型实例。
    采用单例模式管理本地深度学习模型，避免重复加载炸显存。
    """
    
    _embedding_model_instance = None
    _reranker_model_instance = None

    @classmethod
    def get_llm(cls, model_tier: str = "smart") -> Any:
        """
        获取大语言模型 (LLM) 客户端/实例。
        
        :param model_tier: "smart" (用于主控Agent等复杂任务) 或 "cheap" (用于上下文补全、路由等简单任务)
        :return: 配置好 API Key 和 Base URL 的 LLM 客户端对象
        """
        from openai import OpenAI
        
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        if model_tier == "smart":
            client.default_model = settings.smart_llm_model
        elif model_tier == "cheap":
            client.default_model = settings.cheap_llm_model
        else:
            raise ValueError(f"未知的 LLM 层级: {model_tier}")
            
        return client

    @classmethod
    def get_embedding_model(cls) -> Any:
        """
        获取稠密向量模型 (Dense Embedding Model)。
        用于语义切分和向量入库。采用单例模式。
        """
        if cls._embedding_model_instance is None:
            print("🚀 连接到本地 Embedding 模型...")
            cls._embedding_model_instance = APIEmbeddingModel(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                model_name=settings.embedding_model_name
            )
            print("✅ Embedding 模型连接完成！")
            
        return cls._embedding_model_instance

    @classmethod
    def get_reranker_model(cls) -> Any:
        """
        获取交叉编码器重排模型 (Cross-Encoder Reranker)。
        用于多路召回后的二次精准排序。采用单例模式。
        """
        if cls._reranker_model_instance is None:
            print("⚖️ 连接本地 Reranker 模型...")
            cls._reranker_model_instance = APIRerankerModel(
                base_url=settings.reranker_base_url,
                api_key=settings.reranker_api_key,
                model_name=settings.reranker_model_name
            )
            print("✅ Reranker 模型连接完成！")
            
        return cls._reranker_model_instance

    @classmethod
    def get_sparse_tokenizer(cls) -> Any:
        """
        获取稀疏检索/分词器 (Sparse/Tokenizer Model)。
        如果使用 Milvus 的倒排索引，可能不需要在 Python 端加载模型，
        但如果使用 Splade 或 BGE-m3 提取稀疏向量，则需在此初始化。
        """
        # TODO: 根据最终调研的 Sparse 方案在此实现初始化逻辑
        pass

# ---------------------------------------------------------
# 业务层调用示例 ：
# ---------------------------------------------------------
if __name__ == "__main__":
    import json
    
    print("=" * 50)
    print("测试 LLM 模型")
    try:
        smart_llm = ModelFactory.get_llm(model_tier="smart")
        response = smart_llm.chat.completions.create(
            model=smart_llm.default_model,
            messages=[{"role": "user", "content": "你好，请简单介绍一下你自己。"}],
            max_tokens=16384
        )
        print("LLM 思考过程:", response.choices[0].message.reasoning)
        print("LLM 响应:", response.choices[0].message.content)
        print("LLM 结束理由:", response.choices[0].finish_reason)
        print("LLM tokens 使用情况:", json.dumps(response.usage.model_dump(), indent=2))
    except Exception as e:
        print("LLM 请求失败:", e)

    print("=" * 50)
    print("测试 Embedding 模型")
    try:
        embed_model = ModelFactory.get_embedding_model()
        vectors = embed_model.encode(["我爱北京天安门", "天安门上太阳升"])
        print("向量维度:", len(vectors[0]))
        print("第一条前5个维度的值:", vectors[0][:5])
    except Exception as e:
        print("Embedding 请求失败:", e)

    print("=" * 50)
    print("测试 Reranker 模型")
    try:
        reranker = ModelFactory.get_reranker_model()
        scores = reranker.predict([("北京的景点", "天安门广场非常好玩"), ("北京的景点", "苹果手机很实用")])
        print("关联度得分:", scores)
    except Exception as e:
        print("Reranker 请求失败:", e)
    print("=" * 50)