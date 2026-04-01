# 🌌 NovelChat — Agentic RAG for Long-Form Novels

> 一套专为长篇小说打造的智能问答系统，融合 **GraphRAG 宏观知识图谱**、**Milvus 混合检索**与**主动式 Agentic 记忆**，让大模型真正"读完"整本书。

---

## 🗺️ 项目总览

NovelChat 将一本百万字级中文小说（默认示例：《吞噬星空》）转化为可交互的知识库，用户既可以用自然语言提问剧情细节，也可以与助理进行角色扮演聊天，系统会自动记录并在后续对话中召回用户的专属偏好与历史记忆。

### 核心亮点

| 特性 | 技术实现 |
|------|----------|
| 🔍 三路混合检索 | Dense (语义向量) + Sparse (BM25) + Graph (Neo4j 图谱) 经 RRF 融合后精排 |
| 🌐 GraphRAG 全局视野 | Leiden 社区发现 + 层次化摘要，支持"人物势力全景"等宏观问题 |
| 🧠 Agentic 长期记忆 | KV 档案 + 图谱关系边 + Milvus 对话记忆库，三层立体记忆结构 |
| 🚦 智能查询路由 | 自动判定 LOCAL / GLOBAL / MEMORY / DIRECT 四种检索模式 |
| 📊 Ragas 评估 | 每轮回答后台异步评估忠实度（Faithfulness）与上下文精度 |
| 💬 双前端接入 | Streamlit Web UI + 飞书机器人 Webhook 两种接入方式 |

---

## 🏗️ 系统架构

```
用户提问
   │
   ▼
┌──────────────────────────────────────────────────┐
│                  Query Router                    │
│  判定模式 (LOCAL / GLOBAL / MEMORY / DIRECT)      │
│  + 指代消解 + 实体提取                            │
└────────────────────┬─────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
     LOCAL / GLOBAL          MEMORY
          │                     │
   ┌──────┴──────┐        ┌──────────┐
   │ Dense 向量  │        │对话记忆库│
   │ BM25 全文   │        │Dense+BM25│
   │ Neo4j 图谱  │        └──────────┘
   └─────┬───────┘
         │ RRF 融合
         │ Cross-Encoder 精排
         ▼
┌────────────────────────────────────────────────────┐
│               NovelAgent (主控大模型)               │
│  注入：KV 档案 + 检索上下文 + 最近 2 轮历史          │
│  工具：update_kv_profile / add_graph_memory        │
└────────────────────────────────────────────────────┘
         │
    后台异步执行
    ├── 保存对话记忆到 Milvus
    ├── 更新 Neo4j 记忆图谱
    └── Ragas 评估打分
```

---

## 📂 目录结构

```
NovelChat/
├── main.py                    # Streamlit Web UI 入口
├── feishu_bot.py              # 飞书机器人 FastAPI 入口
├── docker-compose.yml         # 数据库基础设施 (Milvus + Neo4j)
├── setup.sh                   # 项目骨架初始化脚本
├── pyproject.toml             # Python 依赖管理
├── .env.example               # 环境变量模板
│
└── src/
    ├── config.py              # 全局配置（基于 pydantic-settings）
    ├── agent/
    │   ├── main_agent.py      # NovelAgent 主控逻辑
    │   └── query_router.py    # 四模式智能查询路由
    ├── retrieval/
    │   └── hybrid_retriever.py # 混合检索 + RRF 融合 + Cross-Encoder 精排
    ├── memory/
    │   └── tools.py           # Agentic 记忆工具 (KV / 图谱 / 对话记忆)
    ├── graphrag/
    │   ├── graph_extractor.py # 实体关系抽取 → Neo4j
    │   ├── community_summarizer.py # Leiden 社区划分 + 层次摘要
    │   └── pipeline.py        # Phase 2 离线建图总控
    ├── ingestion/
    │   ├── document_parser.py # 原始 TXT 按篇章物理切分
    │   ├── semantic_chunker.py # 语义微切分（Embedding 相似度断崖）
    │   ├── context_injector.py # 上下文注入（调 LLM 补背景）
    │   ├── milvus_indexer.py  # 向量化写入 Milvus
    │   └── pipeline.py        # Phase 1 离线数据处理总控
    └── utils/
        ├── model_factory.py   # 统一模型工厂（LLM / Embedding / Reranker）
        └── prompts.py         # 所有 Prompt 模板
```

---

## 🚀 快速开始

### 1. 前置依赖

- Python ≥ 3.10
- Docker & Docker Compose（用于启动 Milvus 和 Neo4j）
- 已部署的 OpenAI 兼容 LLM 服务（推荐 Qwen 系列）
- 已部署的 Embedding 服务（推荐 Qwen3-Embedding-8B）
- 已部署的 Reranker 服务（推荐 Qwen3-Reranker-4B）

### 2. 启动数据库

```bash
docker compose up -d
```

这会启动以下服务：

| 服务 | 端口 | 说明 |
|------|------|------|
| Milvus | 19530 | 向量数据库（混合搜索） |
| Neo4j | 7474 / 7687 | 图数据库（GraphRAG） |
| MinIO | 9000 / 9001 | Milvus 对象存储依赖 |
| etcd | 2379 | Milvus 元数据存储依赖 |

### 3. 配置环境变量

```bash
cp .env.example .env
# 按实际情况修改 .env 文件
```

关键配置项：

```ini
# LLM 服务
OPENAI_API_KEY="your-key"
OPENAI_BASE_URL="http://localhost:6006/v1"
SMART_LLM_MODEL="Qwen3.5-27B"

# Embedding 服务
EMBEDDING_BASE_URL="http://localhost:5005/v1"
EMBEDDING_MODEL_NAME="Qwen3-Embedding-8B"

# Reranker 服务
RERANKER_BASE_URL="http://localhost:7007/v1"
RERANKER_MODEL_NAME="Qwen3-Reranker-4B"

# 小说文件路径
NOVEL_RAW_FILE_PATH="/path/to/your_novel.txt"
NOVEL_FILE_ENCODING="gb18030"  # 常见中文编码，按实际调整
```

### 4. 安装 Python 依赖

```bash
pip install -e .
```

### 5. 离线数据预处理（Phase 1：文本 → Milvus）

```bash
python -m src.ingestion.pipeline
```

该流水线依次执行：
1. **物理切片**：按篇/章结构解析原始 TXT
2. **语义微切分**：基于 Embedding 相似度找語義断崖切分
3. **上下文注入**：调大模型为每个 Chunk 补充前置背景
4. **向量入库**：批量写入 Milvus（Dense + Sparse 双索引）

### 6. 离线建图（Phase 2：Milvus → Neo4j GraphRAG）

```bash
python -m src.graphrag.pipeline
```

该流水线依次执行：
1. **实体关系抽取**：调大模型从每个 Chunk 中提取人物/地点/事件三元组
2. **写入 Neo4j**：构建知识图谱
3. **Leiden 社区划分**：分层聚类，生成层次化社区
4. **社区摘要**：为每个社区生成文本摘要并向量化写入 Milvus

### 7. 启动服务

**Streamlit Web UI：**

```bash
streamlit run main.py
```

**飞书机器人（需先配置飞书应用凭证）：**

```bash
uvicorn feishu_bot:app --host 0.0.0.0 --port 8080
```

---

## 🔍 检索模式详解

QueryRouter 根据用户提问自动选择最优策略：

| 模式 | 触发场景 | 检索路径 |
|------|----------|----------|
| `LOCAL` | 询问具体情节、人物行为等微观问题 | Dense + BM25 + Graph → RRF → Cross-Encoder |
| `GLOBAL` | 询问势力格局、世界观等宏观问题 | 社区摘要（全局视野）+ 原文细节切片 |
| `MEMORY` | 涉及"上次你说的"、用户专属设定等 | 对话记忆库 + 原文相关切片 |
| `DIRECT` | 打招呼、闲聊等无需检索的问题 | 直接回复，跳过所有检索 |

---

## 🧠 Agentic 记忆三层结构

```
┌─────────────────────────────────────────────────┐
│  Layer 1: KV 档案 (本地 JSON 文件)               │
│  存储：用户昵称、偏好、角色扮演设定等结构化信息    │
│  工具：update_kv_profile                         │
├─────────────────────────────────────────────────┤
│  Layer 2: Neo4j 记忆图谱                         │
│  存储：用户实体间的关系边 (Agent_Memory 类型)     │
│  工具：add_graph_memory                          │
├─────────────────────────────────────────────────┤
│  Layer 3: Milvus 对话记忆库                      │
│  存储：每轮对话的语义摘要 + 原始问答全文           │
│  召回：Dense + BM25 双路 RRF 融合               │
└─────────────────────────────────────────────────┘
```

大模型在生成回复时可通过 Function Calling 主动调用 `update_kv_profile` 和 `add_graph_memory`；每轮对话结束后，系统后台异步将完整对话摘要存入 Milvus。

---

## 💬 飞书机器人使用

在飞书中 @ 机器人，发送文字消息即可开始对话。支持以下内置指令：

| 指令 | 效果 |
|------|------|
| `删除记忆` | 查看记忆删除帮助 |
| `删除记忆 1` | 清空 KV 档案 |
| `删除记忆 2` | 删除图谱 + 重建对话记忆 |
| `删除记忆 12` | 同时执行以上两项 |

---

## ⚙️ 关键配置参数

以下参数可在 `.env` 或 `src/config.py` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_MAX_TOKENS` | 800 | 每个微观 Chunk 最大 Token 数 |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | 90 | 语义断点百分位阈值（越高切片越少越长） |
| `MILVUS_VECTOR_DIM` | 4096 | 向量维度（需与 Embedding 模型匹配） |
| `RRF_K` (rrf_k) | 60 | RRF 平滑常数，越大各路权重越均匀 |
| `TOP_K_RETRIEVAL` | 60 | 单路初始召回数量 |
| `TOP_K_RERANK` | 10 | Cross-Encoder 精排后送给大模型的数量 |
| `RERANK_THRESHOLD` | 0.70 | Cross-Encoder 绝对阈值，低于则剔除 |
| `GLOBAL_SUMMARY_TOP_K` | 7 | GLOBAL 模式摘要召回数量 |

---

## 📊 Ragas 质量评估

每轮问答后，系统在后台自动执行 [Ragas](https://github.com/explodinggradients/ragas) 评估，评估结果写入 `logs/rag_eval_logs.jsonl`：

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "question": "罗峰是谁？",
  "scores": {
    "faithfulness": 0.95,
    "context_precision": 0.87
  }
}
```

---

## 🔧 开发与调试

**直接在终端交互（CLI 模式）：**

```bash
python -m src.agent.main_agent
```

**测试查询路由：**

```bash
python -m src.agent.query_router
```

**查看 Milvus 数据样本：**

```bash
python scripts/read_milvus_samples.py
```

**清空记忆：**

```bash
python scripts/delete_memory.py
```

---

## 📦 依赖栈

| 类别 | 库 |
|------|----|
| LLM 交互 | `openai`, `tiktoken` |
| 向量模型 | `sentence-transformers`, `torch` |
| 向量数据库 | `pymilvus >= 2.4.0` |
| 图数据库 | `neo4j >= 5.15.0` |
| 图算法 | `networkx`, `cdlib`, `leidenalg`, `igraph` |
| NLP | `spacy` |
| 配置管理 | `pydantic-settings`, `pydantic` |
| Web UI | `streamlit` |
| API 服务 | `fastapi`, `uvicorn` |
| 评估 | `ragas`, `langsmith` |

---

## 📜 License

MIT
