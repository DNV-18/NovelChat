import asyncio
from typing import Any, Dict, List

from neo4j import GraphDatabase
from tqdm import tqdm

from src.config import settings
from src.utils.model_factory import ModelFactory


NODE_PROFILE_SYSTEM_PROMPT = """
你是小说知识图谱的节点画像整理专家。
你的任务是把同一个 Entity 在不同章节里积累的零碎描述融合成一段稳定、连贯、可检索的全局画像。

要求：
1. 只依据输入描述，不编造未出现的信息。
2. 保留别名、身份、阵营、能力、关系、性格、关键经历等有助于实体对齐的信息。
3. 如果描述之间有轻微重复，请合并去重；如果存在阶段变化，请用时间/剧情推进的方式表达。
4. 输出约 200 个中文字，直接输出画像正文，不要使用项目符号、标题或 JSON。
""".strip()


class NodeProfiler:
    """
    Step 1.5: 将 Entity.all_descriptions 汇总为全局 description，并写入 Neo4j 向量索引字段。
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_pwd: str,
        embedding_model: Any,
        batch_size: int = 100,
        max_concurrency: int = 4,
        vector_dim: int | None = None,
        clear_all_descriptions: bool = True,
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        self.embedding_model = embedding_model
        self.batch_size = max(1, batch_size)
        self.max_concurrency = max(1, max_concurrency)
        self.vector_dim = int(vector_dim or settings.milvus_vector_dim)
        self.clear_all_descriptions = clear_all_descriptions
        self._ensure_vector_index()

    def close(self):
        self.driver.close()

    def _ensure_vector_index(self):
        """初始化 Neo4j 5.x Entity 向量索引。"""
        if self.vector_dim <= 0:
            raise ValueError(f"非法向量维度: {self.vector_dim}")

        cypher = f"""
        CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.vector_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        with self.driver.session() as session:
            session.run(cypher).consume()
            session.run("CALL db.awaitIndex('entity_embedding', 300)").consume()

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """兼容 OpenAI SDK 的多种 message.content 结构。"""
        try:
            message = response.choices[0].message
        except Exception:
            return ""

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _normalize_descriptions(descriptions: Any) -> List[str]:
        """去空、去重，避免把无意义片段喂给画像 LLM。"""
        if not isinstance(descriptions, list):
            return []

        seen = set()
        clean: List[str] = []
        for item in descriptions:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            clean.append(text)
        return clean

    @staticmethod
    def _fallback_summary(entity_id: str, entity_type: str, descriptions: List[str]) -> str:
        joined = "；".join(descriptions)
        if not joined:
            return f"{entity_id}是小说中的{entity_type or '实体'}，当前缺少稳定描述。"
        return joined[:500]

    def _fetch_profile_node_ids(self) -> List[str]:
        cypher = """
        MATCH (e:Entity)
        WHERE e.all_descriptions IS NOT NULL AND size(e.all_descriptions) > 0
        RETURN e.id AS id
        ORDER BY e.id
        """
        with self.driver.session() as session:
            return [row["id"] for row in session.run(cypher) if row["id"]]

    def _fetch_profile_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        cypher = """
        UNWIND $node_ids AS node_id
        MATCH (e:Entity {id: node_id})
        RETURN
            e.id AS id,
            coalesce(e.type, '未知') AS type,
            coalesce(e.description, '') AS existing_description,
            e.all_descriptions AS all_descriptions
        """
        with self.driver.session() as session:
            return [dict(row) for row in session.run(cypher, node_ids=node_ids)]

    @staticmethod
    def _build_user_prompt(entity_id: str, entity_type: str, descriptions: List[str]) -> str:
        description_block = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(descriptions))
        return f"""
实体名称：{entity_id}
实体类型：{entity_type or '未知'}

章节零碎描述：
{description_block}

请融合为一段约 200 字的全局画像。
""".strip()

    async def _generate_profile(
        self,
        node: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, str]:
        entity_id = str(node.get("id") or "").strip()
        entity_type = str(node.get("type") or "未知").strip()
        descriptions = self._normalize_descriptions(node.get("all_descriptions"))

        if not descriptions:
            existing = str(node.get("existing_description") or "").strip()
            descriptions = [existing] if existing else []

        if not entity_id:
            return {"id": entity_id, "summary": ""}

        if not descriptions:
            return {
                "id": entity_id,
                "summary": self._fallback_summary(entity_id, entity_type, descriptions),
            }

        try:
            async with semaphore:
                response = await ModelFactory.chat_completion_async(
                    messages=[
                        {"role": "system", "content": NODE_PROFILE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": self._build_user_prompt(entity_id, entity_type, descriptions),
                        },
                    ],
                    model_tier=settings.node_profile_model_tier,
                    temperature=0.2,
                    max_tokens=1024,
                )
            summary = self._extract_response_text(response)
            if not summary:
                summary = self._fallback_summary(entity_id, entity_type, descriptions)
            return {"id": entity_id, "summary": summary}
        except Exception as e:
            print(f"⚠️ 节点画像生成失败 entity={entity_id}: {e}")
            return {
                "id": entity_id,
                "summary": self._fallback_summary(entity_id, entity_type, descriptions),
            }

    @staticmethod
    def _coerce_vector(vector: Any) -> List[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(x) for x in vector]

    def _embed_summaries(self, summaries: List[str]) -> List[List[float]]:
        vectors = self.embedding_model.encode(summaries)
        if len(vectors) != len(summaries):
            raise ValueError(f"Embedding 返回数量不匹配: input={len(summaries)}, output={len(vectors)}")

        normalized = [self._coerce_vector(vector) for vector in vectors]
        for vector in normalized:
            if len(vector) != self.vector_dim:
                raise ValueError(f"向量维度不匹配: expected={self.vector_dim}, actual={len(vector)}")
        return normalized

    def _write_profiles(self, rows: List[Dict[str, Any]]):
        if not rows:
            return

        cypher = """
        UNWIND $rows AS row
        MATCH (n:Entity {id: row.id})
        SET n.description = row.summary,
            n.embedding = row.vector,
            n.all_descriptions = CASE
                WHEN $clear_all_descriptions THEN null
                ELSE n.all_descriptions
            END
        """
        with self.driver.session() as session:
            session.run(cypher, rows=rows, clear_all_descriptions=self.clear_all_descriptions)

    async def _profile_batch(
        self,
        nodes: List[Dict[str, Any]],
        profile_pbar: tqdm,
    ) -> Dict[str, int]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [asyncio.create_task(self._generate_profile(node, semaphore)) for node in nodes]

        profile_rows: List[Dict[str, str]] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            profile_pbar.update(1)
            if result.get("id") and result.get("summary"):
                profile_rows.append(result)

        if not profile_rows:
            return {"profiled": 0, "written": 0}

        vectors = self._embed_summaries([row["summary"] for row in profile_rows])
        write_rows = [
            {"id": row["id"], "summary": row["summary"], "vector": vector}
            for row, vector in zip(profile_rows, vectors)
        ]
        self._write_profiles(write_rows)
        return {"profiled": len(profile_rows), "written": len(write_rows)}

    async def process_all_nodes(self) -> Dict[str, int]:
        node_ids = self._fetch_profile_node_ids()
        if not node_ids:
            print("ℹ️ 未发现带 all_descriptions 的 Entity 节点，跳过节点画像。")
            return {"total_nodes": 0, "profiled_nodes": 0, "written_nodes": 0}

        print(f"🧬 [Step 1.5] 发现 {len(node_ids)} 个待画像 Entity，开始汇总 description 与 embedding...")
        profiled_total = 0
        written_total = 0
        profile_pbar = tqdm(total=len(node_ids), desc="节点画像生成", unit="node")

        try:
            for start in range(0, len(node_ids), self.batch_size):
                batch_ids = node_ids[start:start + self.batch_size]
                nodes = self._fetch_profile_nodes(batch_ids)
                result = await self._profile_batch(nodes, profile_pbar)
                profiled_total += result["profiled"]
                written_total += result["written"]
        finally:
            profile_pbar.close()

        print(f"✅ 节点画像完成：生成 {profiled_total} 个，写回 {written_total} 个。")
        return {
            "total_nodes": len(node_ids),
            "profiled_nodes": profiled_total,
            "written_nodes": written_total,
        }
