import requests
from typing import Any, Dict, List, Tuple

from src.config import settings
from src.utils.tracing import traceable

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
    _async_llm_client_instance = None

    @staticmethod
    def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """浅到中等深度字典合并，主要用于 extra_body 配置拼接。"""
        merged = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = ModelFactory._merge_dict(merged[k], v)
            else:
                merged[k] = v
        return merged

    @staticmethod
    def _resolve_model_name(model_tier: str, model_name: str | None = None) -> str:
        """统一解析模型名。"""
        if model_name is not None:
            return model_name
        if model_tier == "smart":
            return settings.smart_llm_model
        if model_tier == "cheap":
            return settings.cheap_llm_model
        raise ValueError(f"未知的 LLM 层级: {model_tier}")

    @staticmethod
    def _resolve_enable_thinking(model_tier: str, enable_thinking: bool | None = None) -> bool:
        """统一解析是否开启思考。"""
        if enable_thinking is not None:
            return enable_thinking
        if model_tier == "smart":
            return True
        if model_tier == "cheap":
            return False
        raise ValueError(f"未知的 LLM 层级: {model_tier}")

    @classmethod
    def _build_extra_body(
        cls,
        model_tier: str,
        enable_thinking: bool | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """统一构造 extra_body。"""
        resolved_enable_thinking = cls._resolve_enable_thinking(model_tier, enable_thinking)
        base_extra_body: Dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": resolved_enable_thinking}
        }
        if not extra_body:
            return base_extra_body
        return cls._merge_dict(base_extra_body, extra_body)

    @classmethod
    def get_llm(cls) -> Any:
        """
        获取同步 LLM 客户端。
        仅负责创建客户端，不负责模型选择。
        """
        from openai import OpenAI
        
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        
        return OpenAI(api_key=api_key, base_url=base_url)
    
    @classmethod
    @traceable(name="llm_chat_completion", run_type="llm")
    def chat_completion(
        cls,
        messages: List[Dict[str, str]],
        model_tier: str = "smart",
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 16384,
        extra_body: Dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        统一同步 Chat Completions 调用入口。
        与 chat_completion_async 参数语义保持一致。
        """
        resolved_model = cls._resolve_model_name(model_tier=model_tier, model_name=model_name)
        resolved_extra_body = cls._build_extra_body(
            model_tier=model_tier,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

        client = cls.get_llm()
        request_payload: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": resolved_extra_body,
        }
        if tools is not None:
            request_payload["tools"] = tools
        if tool_choice is not None:
            request_payload["tool_choice"] = tool_choice
        if kwargs:
            request_payload.update(kwargs)

        return client.chat.completions.create(**request_payload)

    @classmethod
    def get_async_llm(cls) -> Any:
        """
        获取异步 LLM 客户端。
        用于批量并发任务（如上下文注入），采用单例模式复用连接配置。
        """
        from openai import AsyncOpenAI

        if cls._async_llm_client_instance is None:
            cls._async_llm_client_instance = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        return cls._async_llm_client_instance

    @classmethod
    @traceable(name="llm_chat_completion_async", run_type="llm")
    async def chat_completion_async(
        cls,
        messages: List[Dict[str, str]],
        model_tier: str = "cheap",
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 16384,
        extra_body: Dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        统一异步 Chat Completions 调用入口。
        业务代码不直接触碰 SDK 细节，全部经由工厂转发。
        """
        resolved_model = cls._resolve_model_name(model_tier=model_tier, model_name=model_name)
        resolved_extra_body = cls._build_extra_body(
            model_tier=model_tier,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

        client = cls.get_async_llm()
        request_payload: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": resolved_extra_body,
        }
        if tools is not None:
            request_payload["tools"] = tools
        if tool_choice is not None:
            request_payload["tool_choice"] = tool_choice
        if kwargs:
            request_payload.update(kwargs)

        return await client.chat.completions.create(**request_payload)


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
    # print("测试 LLM 模型")
    # try:
    #     response = ModelFactory.chat_completion(
    #         messages=[{"role": "user", "content": "你好，请简单介绍一下你自己。"}],
    #         model_tier="cheap",
    #         max_tokens=16384,
    #     )
    #     print("LLM 响应对象:", response)
    #     # print("LLM 思考过程:", response.choices[0].message.reasoning)
    #     # print("LLM 响应:", response.choices[0].message.content)
    #     # print("LLM 结束理由:", response.choices[0].finish_reason)
    #     # print("LLM tokens 使用情况:", json.dumps(response.usage.model_dump(), indent=2))
    # except Exception as e:
    #     print("LLM 请求失败:", e)

    # print("=" * 50)
    # print("测试 Embedding 模型")
    # try:
    #     embed_model = ModelFactory.get_embedding_model()
    #     vectors = embed_model.encode(["我爱北京天安门", "天安门上太阳升"])
    #     print("向量维度:", len(vectors[0]))
    #     print("第一条前5个维度的值:", vectors[0][:5])
    # except Exception as e:
    #     print("Embedding 请求失败:", e)

    # print("=" * 50)
    print("测试 Reranker 模型")
    try:
        reranker = ModelFactory.get_reranker_model()
        scores = reranker.predict([("北京的景点", "天安门广场非常好玩"), ("北京的景点", "苹果手机很实用")])
        print("关联度得分:", scores)
    except Exception as e:
        print("Reranker 请求失败:", e)
    print("=" * 50)