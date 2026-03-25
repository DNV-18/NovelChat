import os
from functools import lru_cache
from typing import Any

from src.config import settings

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
        
        # 实际开发中，这些配置应从 src.config.settings 中读取
        api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 通过为客户端绑定一个默认模型名，方便业务层直接使用
        if model_tier == "smart":
            client.default_model = "gpt-4o" # 或 Claude 3.5 Sonnet 等顶级模型
        elif model_tier == "cheap":
            client.default_model = "gpt-4o-mini" # 或 Qwen-Turbo 等高性价比模型
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
            print("🚀 首次加载 Embedding 模型，这可能需要一点时间...")
            from sentence_transformers import SentenceTransformer
            
            # 例如选用 BAAI 的 bge-large-zh-v1.5 或其他优秀中文模型
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
            # 内部缓存实例
            cls._embedding_model_instance = SentenceTransformer(model_name)
            print("✅ Embedding 模型加载完成！")
            
        return cls._embedding_model_instance

    @classmethod
    def get_reranker_model(cls) -> Any:
        """
        获取交叉编码器重排模型 (Cross-Encoder Reranker)。
        用于多路召回后的二次精准排序。采用单例模式。
        """
        if cls._reranker_model_instance is None:
            print("⚖️ 首次加载 Reranker 模型，准备分配显存...")
            from sentence_transformers import CrossEncoder
            
            # 例如选用 BGE 的重排模型
            model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
            # 内部缓存实例
            cls._reranker_model_instance = CrossEncoder(model_name)
            print("✅ Reranker 模型加载完成！")
            
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
# 业务层调用示例 (伪代码)：
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 业务：需要给图谱提取实体（属于复杂任务）
    smart_llm = ModelFactory.get_llm(model_tier="smart")
    # response = smart_llm.chat.completions.create(model=smart_llm.default_model, messages=[...])
    
    # 2. 业务：需要把长文本切成句子算相似度
    embed_model = ModelFactory.get_embedding_model()
    # vectors = embed_model.encode(["句子1", "句子2"])
    
    # 3. 业务：需要给多路召回的结果打分
    reranker = ModelFactory.get_reranker_model()
    # scores = reranker.predict([("问题", "候选答案1"), ("问题", "候选答案2")])