#!/bin/bash

# 1. 创建主项目目录
mkdir -p NovelChat
cd NovelChat

# 2. 创建核心代码目录结构 (采用标准的 src layout)
# data: 存放原始小说、Milvus切片备份、KV档案表JSON等数据文件
mkdir -p data/raw data/processed data/memory

# src: 核心业务代码库
# ingestion: 负责物理切片、语义微切分、指代消解
mkdir -p src/ingestion
# graphrag: 负责实体抽取、Neo4j图谱构建、Leiden社区划分、各级摘要总结
mkdir -p src/graphrag
# retrieval: 负责Milvus混合检索、Neo4j局部/全局检索
mkdir -p src/retrieval
# memory: 负责KV档案表读写、长事件入库、GraphRAG记忆节点更新（Tools）
mkdir -p src/memory
# agent: 负责路由判定(Router)、重排序(Cross-Encoder RRF)、LLM问答生成与记忆调度
mkdir -p src/agent
# utils: 存放通用工具
mkdir -p src/utils

# 3. 创建入口文件和现代 Python 项目配置文件
touch main.py          # 系统的入口启动文件
touch .env             # 环境变量（仅存放数据库密码、API Keys等敏感信息，切勿提交到 Git）
touch .gitignore       # Git 忽略文件（记得在里面加上 .env 和 __pycache__/）
touch README.md        # 项目说明文档

# 写入现代 Python 包管理标准：pyproject.toml 模板
echo "📦 正在生成依赖清单 (pyproject.toml)..."
cat <<EOF > pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "novel-chat"
version = "0.1.0"
description = "A novel RAG system with Agentic memory, Milvus Hybrid Search, and Neo4j GraphRAG"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    # --- 基础配置与环境变量 ---
    "python-dotenv>=1.0.0",
    "pydantic-settings>=2.1.0",
    "pydantic>=2.5.0",
    
    # --- LLM 交互与大模型 ---
    "openai>=1.10.0",        # OpenAI 官方 SDK (用于调用 GPT 或兼容 API 的本地模型)
    "tiktoken>=0.5.0",       # Token 计算工具
    
    # --- 向量模型与重排序 (Cross-Encoder) ---
    "sentence-transformers>=2.3.0", # 用于加载 Embedding 模型和 BGE-Reranker
    "torch>=2.0.0",                 # 深度学习底层依赖
    
    # --- 向量数据库 (Milvus) ---
    "pymilvus>=2.4.0",       # 必须大于 2.4.0 才完美支持内置混合检索与稀疏向量
    
    # --- 图数据库与图算法 (Neo4j & GraphRAG) ---
    "neo4j>=5.15.0",         # Neo4j 官方 Python 驱动
    "networkx>=3.2.0",       # 内存级图拓扑计算
    "cdlib>=0.3.0",          # 社区发现算法库 (包含 Leiden 算法封装)
    "leidenalg>=0.10.0",     # 核心 Leiden 算法底层 C++ 绑定
    "igraph>=0.11.0",        # cdlib 和 leidenalg 的强依赖
    
    # --- NLP 与文本处理 ---
    "spacy>=3.7.0",          # 强大的 NLP 库，用于极简路由时的实体提取
    
    # --- 其他实用工具 ---
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "tqdm>=4.66.0",          # 进度条 (处理小说切片时极度需要)
    "aiohttp>=3.9.0",        # 异步网络请求
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.11.0",
    "isort>=5.12.0",
]
EOF

# 4. 创建各个模块的 __init__.py（将目录变为 Python 包）
touch src/__init__.py
touch src/ingestion/__init__.py
touch src/graphrag/__init__.py
touch src/retrieval/__init__.py
touch src/memory/__init__.py
touch src/agent/__init__.py
touch src/utils/__init__.py

# 5. 创建配置管理与基础工具文件
touch src/config.py        # 核心配置类：负责读取 .env 并设置各种不敏感的默认变量 (如 Chunk Size, 重试次数等)
touch src/utils/prompts.py # 统一存放所有供大模型使用的 Prompt 模板
touch src/utils/model_factory.py

# 写入 Git 忽略文件
cat <<EOF > .gitignore
.env
__pycache__/
*.pyc
.venv/
venv/
data/raw/*
data/processed/*
data/memory/*
!data/**/.gitkeep
EOF

echo "✅ 现代化 Python 项目骨架创建完毕！目录结构如下："
# 如果系统安装了 tree 命令，可以展示目录树（可选）
if command -v tree &> /dev/null; then
    tree -L 3
else
    echo "请在资源管理器或 VSCode 中查看新建的目录。"
fi