import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录，方便后续拼接绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    系统全局配置类。
    基于 pydantic-settings，它会自动从环境变量或 .env 文件中读取同名字段。
    如果没有在 .env 中配置，则会使用这里定义的默认值。
    """
    
    # ==========================================
    # 🤖 1. LLM 与 本地模型选型配置
    # ==========================================
    openai_api_key: str = Field(..., description="LLM API Key")
    openai_base_url: str = Field("http://localhost:6006/v1", description="LLM Base URL")
    
    # 默认的主力/聪明模型 (主控 Agent 使用)
    smart_llm_model: str = Field("Qwen3.5-27B", description="主控 Agent 使用的高级模型")
    # 默认的廉价/快速模型 (路由、补充上下文使用)
    cheap_llm_model: str = Field("Qwen3.5-27B", description="流水线任务使用的高性价比模型")
    
    # 稠密向量 Embedding 模型名称
    embedding_api_key: str = Field("EMPTY", description="Embedding API Key")
    embedding_base_url: str = Field("http://localhost:5005/v1", description="Embedding Base URL")
    embedding_model_name: str = Field("Qwen3-Embedding-8B", description="文本向量化模型")
    
    # 交叉编码器 Reranker 模型名称
    reranker_api_key: str = Field("EMPTY", description="Reranker API Key")
    reranker_base_url: str = Field("http://localhost:7007/v1", description="Reranker Base URL")
    reranker_model_name: str = Field("Qwen3-Reranker-4B", description="多路召回重排序模型")

    # ==========================================
    # 📚 2. 小说数据处理与切片配置
    # ==========================================
    novel_raw_file_path: Path = Field(
        default=BASE_DIR / "data" / "raw" / "tsxk.txt",
        description="原始小说文本路径"
    )
    novel_file_encoding: str = Field("gb18030", description="小说文本编码")
    heading_max_line_chars: int = Field(40, description="标题行最大长度阈值")

    # 微观切分：语义切分的上下文限制
    chunk_max_tokens: int = Field(800, description="每个微观 Chunk 的最大 Token 数量")
    semantic_breakpoint_percentile: int = Field(90, description="语义断点百分位阈值")
    context_inject_max_concurrency: int = Field(20, description="上下文注入并发数")
    context_inject_model_tier: str = Field("cheap", description="上下文注入使用的模型层级")
    graph_extract_model_tier: str = Field("cheap", description="实体关系抽取使用的模型层级")
    community_summary_model_tier: str = Field("cheap", description="社区摘要生成使用的模型层级")
    community_summary_max_concurrency: int = Field(4, description="社区摘要同层并发上限")

    # Tokenizer 配置
    local_tokenizer_path: str = Field("/home/dnv/models/Qwen3.5-27B", description="本地 tokenizer 目录")

    # ==========================================
    # 🕸️ 3. 数据库连接配置
    # ==========================================
    # Neo4j 
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j 连接地址")
    neo4j_username: str = Field("neo4j", description="Neo4j 用户名")
    neo4j_password: str = Field("admin123", description="Neo4j 密码")
    
    # Milvus
    milvus_uri: str = Field("http://localhost:19530", description="Milvus 连接地址")
    milvus_db_name: str = Field("novel_chat", description="Milvus 默认数据库名称")
    milvus_collection_name: str = Field("novel_chunks", description="存放小说切片的集合名称")
    milvus_community_summary_collection: str = Field("community_summaries", description="社区摘要集合名称")
    milvus_vector_dim: int = Field(4096, description="Milvus 稠密向量维度")
    milvus_insert_batch_size: int = Field(500, description="Milvus 批量写入 batch 大小")

    # ==========================================
    # 🧠 4. 记忆管理 (Agentic Memory) 路径配置
    # ==========================================
    # KV 档案表本地文件路径
    memory_kv_path: Path = Field(
        default=BASE_DIR / "data" / "memory" / "kv_profile.json", 
        description="KV 档案表 (JSON) 存储路径"
    )
    # 每轮对话自动记忆在 Milvus 中的专属 Collection
    milvus_chat_memory_collection: str = Field("chat_turn_memories", description="对话轮次记忆集合名称")
    memory_event_summary_max_chars: int = Field(12000, description="对话摘要最大字数")
    memory_event_summary_model_tier: str = Field("smart", description="对话摘要使用的模型层级")

    # ==========================================
    # ⚙️ 5. 系统运行参数
    # ==========================================
    main_agent_model_tier: str = Field("smart", description="主对话 Agent 使用的模型层级")
    query_router_model_tier: str = Field("smart", description="Query Router 使用的模型层级")
    query_router_history_turns: int = Field(2, description="Query Router 参与判定的最近历史轮数")
    max_retries: int = Field(3, description="API 请求失败重试次数")
    rrf_k: int = Field(60, description="RRF (倒数秩融合) 的平滑常数 k")
    top_k_retrieval: int = Field(60, description="单路召回的初始 Chunk 数量")
    top_k_rerank: int = Field(10, description="Cross-Encoder 精排后最终喂给大模型的 Chunk 数量")
    global_summary_top_k: int = Field(7, description="GLOBAL 模式宏观摘要召回数量")
    global_detail_chunk_top_k: int = Field(3, description="GLOBAL 模式微观细节补充切片数量")
    memory_summary_top_k: int = Field(3, description="MEMORY 模式返回的记忆摘要数量")
    memory_detail_chunk_top_k: int = Field(7, description="MEMORY 模式补充的原著切片数量")
    user_graph_memory_top_k: int = Field(3, description="命中当前用户实体时，从图谱证据池语义召回的记忆摘要数量")
    rerank_threshold: float = Field(0.70, description="Cross-Encoder 精排绝对阈值，低于该分数的结果将被剔除")

    # 声明从哪读取环境变量
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # 忽略 .env 中未在类中定义的额外字段
    )

# 实例化全局配置对象 (单例)
# 其他模块只需执行: `from src.config import settings`
settings = Settings()

if __name__ == "__main__":
    # 简单测试配置是否加载成功
    print(f"✅ Config Loaded Successfully!")
    print(f"🔹 LLM Base URL: {settings.openai_base_url}")
    print(f"🔹 Smart Model: {settings.smart_llm_model}")
    print(f"🔹 KV Memory Path: {settings.memory_kv_path}")