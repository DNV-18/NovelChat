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
    openai_base_url: str = Field("https://api.openai.com/v1", description="LLM Base URL")
    
    # 默认的主力/聪明模型 (主控 Agent 使用)
    smart_llm_model: str = Field("gpt-4o", description="主控 Agent 使用的高级模型")
    # 默认的廉价/快速模型 (路由、补充上下文使用)
    cheap_llm_model: str = Field("gpt-4o-mini", description="流水线任务使用的高性价比模型")
    
    # 稠密向量 Embedding 模型名称 (自动下载)
    embedding_model_name: str = Field("BAAI/bge-large-zh-v1.5", description="文本向量化模型")
    # 交叉编码器 Reranker 模型名称 (自动下载)
    reranker_model_name: str = Field("BAAI/bge-reranker-base", description="多路召回重排序模型")

    # ==========================================
    # 📚 2. 小说数据处理与切片配置
    # ==========================================
    # 宏观切分：默认的正则表达式（根据小说排版可能需要调整）
    chapter_split_regex: str = Field(r"第[零一二三四五六七八九十百千]+章", description="章节切分正则")
    
    # 微观切分：语义切分的上下文限制
    chunk_max_tokens: int = Field(800, description="每个微观 Chunk 的最大 Token 数量")
    chunk_overlap_tokens: int = Field(50, description="切片硬截断时的安全重叠区大小")
    semantic_chunk_threshold: float = Field(0.85, description="语义切分的余弦相似度断崖阈值 (0-1)")

    # ==========================================
    # 🕸️ 3. 数据库连接配置
    # ==========================================
    # Neo4j 
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j 连接地址")
    
    # Milvus
    milvus_uri: str = Field("http://localhost:19530", description="Milvus 连接地址")
    milvus_db_name: str = Field("novel_chat", description="Milvus 默认数据库名称")
    milvus_collection_name: str = Field("novel_chunks", description="存放小说切片的集合名称")

    # ==========================================
    # 🧠 4. 记忆管理 (Agentic Memory) 路径配置
    # ==========================================
    # KV 档案表本地文件路径
    memory_kv_path: Path = Field(
        default=BASE_DIR / "data" / "memory" / "kv_profile.json", 
        description="KV 档案表 (JSON) 存储路径"
    )
    # 长期事件集在 Milvus 中的专属 Collection
    milvus_event_collection: str = Field("long_term_events", description="长期事件记忆集合名称")

    # ==========================================
    # ⚙️ 5. 系统运行参数
    # ==========================================
    max_retries: int = Field(3, description="API 请求失败重试次数")
    rrf_k: int = Field(60, description="RRF (倒数秩融合) 的平滑常数 k")
    top_k_retrieval: int = Field(60, description="单路召回的初始 Chunk 数量")
    top_k_rerank: int = Field(6, description="Cross-Encoder 精排后最终喂给大模型的 Chunk 数量")

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