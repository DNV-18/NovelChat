import os
import json
import uuid
import time
import re
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
            "description": "用于写入【稳定且长期有效】的用户偏好与硬约束到 KV 档案（最高优先级，后续轮次持续生效）。仅在用户明确给出长期规则时调用：称呼偏好、回答语言、禁忌、固定角色设定、长期口味。不要用于普通事实、一次性任务进展、临时情绪或可过期信息。每次调用只写一条 key-value。",
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
            "name": "add_graph_memory",
            "description": "用于写入【当前用户 -> 目标实体】的关系型记忆（普通相关记忆，不是硬性偏好）。仅在可结构化成关系三元组时调用：如‘[当前用户] 喜欢 罗峰’、‘[当前用户] 负责 搜索模块’。不要用于长期硬约束偏好（应使用 update_kv_profile）。系统会自动为每轮对话生成摘要并写入向量库，且由后端自动把事件 ID 绑定为图关系证据；本工具只负责关系三元组本身。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "关系起点实体。必须为 '[当前用户]'。若传入其他值，系统会自动修正为 '[当前用户]'。"
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
        self.milvus_collection_name = settings.milvus_chat_memory_collection
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

    @staticmethod
    def _normalize_summary_text(text: str) -> str:
        """清洗模型输出，便于稳定判断“是否应跳过入库”。"""
        if not text:
            return ""
        cleaned = text.strip()
        # 去掉常见包裹符号（引号/反引号）
        cleaned = cleaned.strip("\"'`“”‘’")
        # 压缩多空白
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _should_skip_memory_summary(cls, summary: str) -> bool:
        """判断摘要是否属于闲聊或空占位，需跳过入库。"""
        normalized = cls._normalize_summary_text(summary)
        if not normalized:
            return True

        lowered = normalized.lower()
        empty_markers = {
            "无",
            "暂无",
            "无内容",
            "空",
            "空字符串",
            "none",
            "null",
            "n/a",
            "na",
            "skip",
            "无有效信息",
            "纯闲聊",
        }
        if lowered in empty_markers or normalized in empty_markers:
            return True

        # 兜底：若模型输出“这是闲聊，跳过”等说明性文本，也视为跳过
        chitchat_hints = ("闲聊", "无需记录", "不需要记录", "跳过", "无长期记忆价值")
        if any(hint in normalized for hint in chitchat_hints):
            return True

        return False

    def _init_milvus_collection(self):
        """确保 Milvus 中存在对话轮次记忆库。"""
        expected_fields = [
            "event_id",
            "timestamp",
            "user_query",
            "ai_response",
            "summary",
            "dense_vector",
        ]

        if utility.has_collection(self.milvus_collection_name, using="default"):
            existing = Collection(self.milvus_collection_name, using="default")
            existing_fields = [f.name for f in existing.schema.fields]
            if existing_fields != expected_fields:
                raise RuntimeError(
                    "Milvus 长期记忆集合字段与新架构不兼容。"
                    f" 当前字段={existing_fields}，期望字段={expected_fields}。"
                    "请清空或重建该集合后重试。"
                )
            self.event_collection = existing
            return

        if not utility.has_collection(self.milvus_collection_name, using="default"):
            print(f"🛠️ 正在初始化专属长期记忆向量库: {self.milvus_collection_name}...")
            fields = [
                FieldSchema(name="event_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="timestamp", dtype=DataType.INT64), # 记录记忆发生的时间戳
                FieldSchema(name="user_query", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="ai_response", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=8192), # 【新增】存放高密度摘要
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus_vector_dim)
            ]
            schema = CollectionSchema(fields, description="Agentic 对话轮次长期记忆库")
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

    def clear_kv_profile(self) -> str:
        """清空本地 KV 档案。"""
        try:
            with open(self.kv_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            return "成功清空 KV Profile。"
        except Exception as e:
            return f"清空 KV Profile 失败: {e}"

    def clear_user_graph_and_memory_db(self) -> str:
        """清理 [当前用户] 出发的图谱边，并重建记忆摘要库。"""
        try:
            deleted_edges = 0
            cypher = """
            MATCH (:UserMemory {id: '[当前用户]'})-[r]->()
            WITH collect(r) AS rels
            FOREACH (rel IN rels | DELETE rel)
            RETURN size(rels) AS deleted_edges
            """
            with self.neo4j_driver.session() as session:
                row = session.run(cypher).single()
                deleted_edges = int(row["deleted_edges"] or 0) if row else 0

            if utility.has_collection(self.milvus_collection_name, using="default"):
                utility.drop_collection(self.milvus_collection_name, using="default")

            # 立即重建集合，保证当前进程和下次启动都不会出现“集合不存在”。
            self._init_milvus_collection()

            return (
                f"成功清理用户图谱记忆：删除边 {deleted_edges} 条；"
                f"并已重建记忆摘要集合 {self.milvus_collection_name}。"
            )
        except Exception as e:
            return f"清理用户图谱记忆/记忆摘要库失败: {e}"

    def clear_all_memories(self) -> str:
        """同时清空 KV 档案与用户图谱/记忆摘要库。"""
        kv_msg = self.clear_kv_profile()
        db_msg = self.clear_user_graph_and_memory_db()
        return f"{kv_msg} | {db_msg}"

    # ---------------------------------------------------------
    # 对话轮次自动记忆: 每轮用户问题 + AI回答 -> 摘要入 Milvus
    # ---------------------------------------------------------
    def save_chat_turn_to_memory_record(self, rewritten_query: str, ai_response: str) -> dict:
        """将一轮对话写入长期记忆，并返回结构化结果（含 event_id）。"""
        try:
            user_query = (rewritten_query or "").strip()
            answer = (ai_response or "").strip()
            if not user_query or not answer:
                return {
                    "ok": False,
                    "event_id": "",
                    "summary": "",
                    "message": "记录长期记忆失败: rewritten_query 或 ai_response 不能为空",
                }

            event_id = f"evt_{uuid.uuid4().hex[:8]}"
            timestamp = int(time.time())
            
            print("🧠 正在为当前对话轮次生成记忆摘要...")
            prompt = build_memory_event_summary_user_prompt(
                max_chars=settings.memory_event_summary_max_chars,
                user_query=user_query,
                ai_response=answer,
            )
            response = ModelFactory.chat_completion(
                messages=[
                    {"role": "system", "content": MEMORY_EVENT_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model_tier=settings.memory_event_summary_model_tier,
                temperature=0.1,
            )
            summary = self._extract_response_text(response)
            normalized_summary = self._normalize_summary_text(summary)

            # 闲聊轮次不入库
            if self._should_skip_memory_summary(normalized_summary):
                return {
                    "ok": True,
                    "event_id": "",
                    "summary": normalized_summary,
                    "message": "本轮判定为闲聊，无需写入长期记忆",
                }
            
            # 2. 【核心优化】：仅对“摘要”进行向量化计算，大幅提高检索精准度
            vector = self.embedding_model.encode([normalized_summary])[0]

            if len(vector) != settings.milvus_vector_dim:
                return {
                    "ok": False,
                    "event_id": "",
                    "summary": normalized_summary,
                    "message": (
                        "记录长期记忆失败: 向量维度不匹配, "
                        f"expected={settings.milvus_vector_dim}, got={len(vector)}"
                    ),
                }
            
            # 3. 入库：保存改写后的用户问题、AI回答与摘要
            self.event_collection.insert([
                [event_id], [timestamp], [user_query], [answer], [normalized_summary], [vector]
            ])
            self.event_collection.flush()
            
            print(f"💾 [自动记忆]: 对话轮次已存入向量库 -> 摘要: {normalized_summary[:30]}...")
            return {
                "ok": True,
                "event_id": event_id,
                "summary": normalized_summary,
                "message": f"成功将对话轮次写入长期记忆库，事件 ID: {event_id}",
            }
        except Exception as e:
            return {
                "ok": False,
                "event_id": "",
                "summary": "",
                "message": f"记录对话轮次记忆失败: {e}",
            }

    # ---------------------------------------------------------
    # 工具 3: 将关系打入 Neo4j 图谱
    # ---------------------------------------------------------
    def add_graph_memory(
        self,
        source_entity: str,
        target_entity: str,
        relation: str,
        memory_evidence_ref: str | None = None,
    ) -> str:
        """只写入 [当前用户] -> 目标实体 的单向记忆关系。"""
        try:
            # 强制约束：仅允许当前用户作为关系起点
            source = "[当前用户]"
            target = (target_entity or "").strip()
            if not target:
                return "添加图谱记忆失败: target_entity 不能为空"

            # 安全处理：去除 relation 中的特殊字符，防止 Cypher 注入
            safe_rel = "".join(c for c in relation if c.isalnum() or c == "_")
            if not safe_rel:
                safe_rel = "RELATED_TO"
                
            # 动态关系名称必须使用反引号 ` 包裹
            cypher = f"""
            MERGE (n1:UserMemory {{id: $source}})
            ON CREATE SET n1.type = 'User_Node', n1.description = 'Current user memory node'
            
            OPTIONAL MATCH (n2e:Entity {{id: $target}})
            OPTIONAL MATCH (n2m:MemoryEntity {{id: $target}})
            WITH n1, coalesce(n2e, n2m) AS n2
            CALL apoc.do.when(
                n2 IS NULL,
                'CREATE (x:MemoryEntity {{id: $target, type: "Memory_Entity", description: "Entity from user memory"}}) RETURN x AS node',
                'RETURN n2 AS node',
                {{target: $target, n2: n2}}
            ) YIELD value
            WITH n1, value.node AS n2
            
            MERGE (n1)-[r:`{safe_rel}`]->(n2)
            ON CREATE SET r.created_by = 'Agent_Memory'
            ON CREATE SET r.memory_evidence_refs = []
            WITH r
            REMOVE r.source_chunk_ids
            SET r.memory_evidence_refs = CASE
                WHEN $memory_evidence_ref IS NULL OR $memory_evidence_ref = '' THEN coalesce(r.memory_evidence_refs, [])
                WHEN $memory_evidence_ref IN coalesce(r.memory_evidence_refs, []) THEN coalesce(r.memory_evidence_refs, [])
                ELSE coalesce(r.memory_evidence_refs, []) + $memory_evidence_ref
            END
            RETURN r
            """
            with self.neo4j_driver.session() as session:
                session.run(
                    cypher,
                    source=source,
                    target=target,
                    memory_evidence_ref=(memory_evidence_ref or "").strip(),
                )
                
            print(f"💾 [主动记忆]: 图谱网络已更新 -> [{source}] -({relation})-> [{target}]")
            return f"成功在知识图谱中建立连接: {source} -> {target}"
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