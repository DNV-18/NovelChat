import os
import json
import uuid
import time
from neo4j import GraphDatabase
from pymilvus import connections, db, utility, FieldSchema, CollectionSchema, DataType, Collection

from src.config import settings
from src.utils.model_factory import ModelFactory
from src.utils.prompts import (
    MEMORY_EVENT_SUMMARY_SYSTEM_PROMPT,
    build_memory_event_summary_user_prompt,
)

# ==========================================
# 1. 供 LLM 调用的 JSON Schema 定义 (OpenAI 格式)
# ==========================================
AGENT_MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_kv_profile",
            "description": "用于写入【稳定且长期有效】的用户设定到 KV 档案。仅在用户明确表达固定偏好/硬性约束时调用（如称呼、语言、禁忌、角色设定）。不要用于一次性任务进展、临时情绪或可过期事实。每次调用只写一条 key-value。",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "设定键名，要求简短稳定、可复用。示例：'称呼偏好'、'回答语言'、'禁忌话题'、'职业背景'。"
                    },
                    "value": {
                        "type": "string",
                        "description": "设定值，使用可直接复用的明确表述。避免模糊词。示例：'默认使用中文'、'称呼我为阿泽'、'对花生过敏'。"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_event_to_long_term",
            "description": "用于写入【中长篇且可能在未来检索复用】的事件记忆到长期向量库。适用于复杂排障过程、项目背景、阶段结论、关键决策记录。不要用于单句偏好（应使用 update_kv_profile），也不要用于显式实体关系（应使用 add_graph_memory）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_description": {
                        "type": "string",
                        "description": "事件完整描述，建议包含背景-过程-结果-影响，保证脱离上下文也可理解；尽量避免只给关键词。"
                    }
                },
                "required": ["event_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_graph_memory",
            "description": "用于写入【明确的实体-关系-实体】图谱记忆。仅在关系可结构化表达时调用（如‘[当前用户] 喜欢 罗峰’、‘[当前用户] 负责 搜索模块’）。不要用于长段事件叙述（应使用 save_event_to_long_term），也不要用于全局偏好标签（应使用 update_kv_profile）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "关系起点实体。若指用户本人，必须使用固定实体名 '[当前用户]'。"
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "关系终点实体，使用稳定实体名（人物、模块、组织、物品等），避免代词。"
                    },
                    "relation": {
                        "type": "string",
                        "description": "关系词，建议使用简短动词或动宾短语，如：'喜欢'、'负责'、'使用'、'排斥'。避免长句。"
                    }
                },
                "required": ["source_entity", "target_entity", "relation"]
            }
        }
    }
]

# ==========================================
# 2. 记忆管理器 (执行 Tool 物理落地的核心类)
# ==========================================
class MemoryManager:
    """
    记忆管理器：负责执行 LLM 发出的工具调用请求，将记忆分别落盘到 JSON、Milvus 和 Neo4j 中。
    """
    def __init__(self):
        # 1. 确保 KV Profile 存储目录存在
        self.kv_path = settings.memory_kv_path
        os.makedirs(os.path.dirname(self.kv_path), exist_ok=True)
        if not os.path.exists(self.kv_path):
            with open(self.kv_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False)

        # 2. 连接 Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri, 
            auth=(settings.neo4j_username, settings.neo4j_password)
        )

        # 3. 初始化 Milvus 和 Embedding 模型
        self._ensure_milvus_db(settings.milvus_db_name)
        self.milvus_collection_name = settings.milvus_event_collection
        self.embedding_model = ModelFactory.get_embedding_model()
        self._init_milvus_collection()

    def _ensure_milvus_db(self, db_name: str):
        """确保目标 Milvus 数据库存在，不存在则自动创建。"""
        connections.connect("default", uri=settings.milvus_uri)
        existing_dbs = db.list_database()
        if db_name not in existing_dbs:
            print(f"🛠️ Milvus 数据库 '{db_name}' 不存在，正在自动创建...")
            db.create_database(db_name)
        connections.connect("default", uri=settings.milvus_uri, db_name=db_name)

    @staticmethod
    def _extract_response_text(response) -> str:
        """兼容提取 OpenAI 兼容响应中的文本。"""
        try:
            message = response.choices[0].message
        except Exception:
            return ""

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                else:
                    txt = getattr(item, "text", "")
                if txt:
                    parts.append(str(txt))
            return "".join(parts).strip()
        for field in ("reasoning_content", "reasoning"):
            val = getattr(message, field, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _init_milvus_collection(self):
        """确保 Milvus 中存在长期事件记忆库"""
        if not utility.has_collection(self.milvus_collection_name, using="default"):
            print(f"🛠️ 正在初始化专属长期记忆向量库: {self.milvus_collection_name}...")
            fields = [
                FieldSchema(name="event_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="timestamp", dtype=DataType.INT64), # 记录记忆发生的时间戳
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=8192), # 【新增】存放高密度摘要
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # 存放完整原文
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus_vector_dim)
            ]
            schema = CollectionSchema(fields, description="Agentic 长期事件记忆库")
            collection = Collection(self.milvus_collection_name, schema, using="default")
            collection.create_index(
                field_name="dense_vector", 
                index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
            )
        self.event_collection = Collection(self.milvus_collection_name, using="default")

    # ---------------------------------------------------------
    # 工具 1: 更新 KV 档案表
    # ---------------------------------------------------------
    def update_kv_profile(self, key: str, value: str) -> str:
        """更新 JSON 档案"""
        try:
            with open(self.kv_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            
            profile[key] = value
            
            with open(self.kv_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
                
            print(f"💾 [主动记忆]: KV 档案已更新 -> {key}: {value}")
            return f"成功更新用户档案: {key} = {value}"
        except Exception as e:
            return f"更新档案失败: {e}"

    # ---------------------------------------------------------
    # 工具 2: 长篇事件切块存入 Milvus
    # ---------------------------------------------------------
    def save_event_to_long_term(self, event_description: str) -> str:
        """长文存入向量库：先生成摘要，再将摘要向量化，最后原文和摘要一并入库"""
        try:
            if not event_description or not event_description.strip():
                return "记录长期记忆失败: event_description 不能为空"

            event_id = f"evt_{uuid.uuid4().hex[:8]}"
            timestamp = int(time.time())
            
            # 1. 调用大模型对长文本进行精炼总结 (使用便宜快速的小模型即可)
            print("🧠 正在为长期记忆生成高密度摘要...")
            response = ModelFactory.chat_completion(
                messages=[
                    {"role": "system", "content": MEMORY_EVENT_SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_memory_event_summary_user_prompt(
                            event_description=event_description,
                            max_chars=settings.memory_event_summary_max_chars,
                        ),
                    },
                ],
                model_tier=settings.memory_event_summary_model_tier,
                temperature=0.1,
            )
            summary = self._extract_response_text(response) or "该事件暂无可提炼摘要。"
            
            # 2. 【核心优化】：仅对“摘要”进行向量化计算，大幅提高检索精准度
            vector = self.embedding_model.encode([summary])[0]

            if len(vector) != settings.milvus_vector_dim:
                return (
                    "记录长期记忆失败: 向量维度不匹配, "
                    f"expected={settings.milvus_vector_dim}, got={len(vector)}"
                )
            
            # 3. 入库：同时保存摘要(用于可视化展示)和原文(用于召回给大模型阅读)
            self.event_collection.insert([
                [event_id], [timestamp], [summary], [event_description], [vector]
            ])
            self.event_collection.flush()
            
            print(f"💾 [主动记忆]: 新长文事件已存入向量库 -> 摘要: {summary[:30]}...")
            return f"成功将事件记录至长期记忆库，事件 ID: {event_id}"
        except Exception as e:
            return f"记录长期记忆失败: {e}"

    # ---------------------------------------------------------
    # 工具 3: 将关系打入 Neo4j 图谱
    # ---------------------------------------------------------
    def add_graph_memory(self, source_entity: str, target_entity: str, relation: str) -> str:
        """写入图数据库建立连接"""
        try:
            # 安全处理：去除 relation 中的特殊字符，防止 Cypher 注入
            safe_rel = "".join(c for c in relation if c.isalnum() or c == "_")
            if not safe_rel:
                safe_rel = "RELATED_TO"
                
            # 动态关系名称必须使用反引号 ` 包裹
            cypher = f"""
            MERGE (n1:Entity {{id: $source}})
            ON CREATE SET n1.type = 'User_Node', n1.description = 'Agentic Memory Entity'
            
            MERGE (n2:Entity {{id: $target}})
            ON CREATE SET n2.type = 'User_Node', n2.description = 'Agentic Memory Entity'
            
            MERGE (n1)-[r:`{safe_rel}`]->(n2)
            ON CREATE SET r.created_by = 'Agent_Memory'
            RETURN r
            """
            with self.neo4j_driver.session() as session:
                session.run(cypher, source=source_entity, target=target_entity)
                
            print(f"💾 [主动记忆]: 图谱网络已更新 -> [{source_entity}] -({relation})-> [{target_entity}]")
            return f"成功在知识图谱中建立连接: {source_entity} -> {target_entity}"
        except Exception as e:
            return f"添加图谱记忆失败: {e}"

    def close(self):
        self.neo4j_driver.close()

# ==========================================
# 获取系统当前全部的 KV 记忆 (用于在聊天的每一轮拼接到 Prompt)
# ==========================================
def get_current_kv_profile() -> str:
    """读取并格式化当前的 KV 记忆，准备送入 System Prompt"""
    try:
        if not os.path.exists(settings.memory_kv_path):
            return "当前暂无特殊偏好记录。"
        with open(settings.memory_kv_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        if not profile:
            return "当前暂无特殊偏好记录。"
        
        lines = [f"- {k}: {v}" for k, v in profile.items()]
        return "\n".join(lines)
    except Exception:
        return "无法读取用户档案。"