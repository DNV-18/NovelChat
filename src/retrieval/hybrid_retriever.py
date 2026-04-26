import json
import logging
import re
from typing import List, Dict, Any
from pymilvus import Collection, connections
from neo4j import GraphDatabase
from langsmith import traceable

from src.config import settings
from src.utils.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridRetriever:
    """
    Phase 5: 混合检索与重排引擎 (终极形态)
    1. LOCAL 模式：执行 Dense + Sparse + Graph 三路 RRF 融合，并进行 Cross-Encoder 精排。
    2. GLOBAL 模式：双轨并行！既检索高维社区摘要，又保留底层原文切片的向量与全文召回。
    """

    def __init__(self):
        # 1. 挂载模型
        self.embedding_model = ModelFactory.get_embedding_model()
        self.reranker = ModelFactory.get_reranker_model()

        # 2. 挂载 Milvus
        connections.connect("default", uri=settings.milvus_uri, db_name=settings.milvus_db_name)
        self.chunk_collection = Collection(settings.milvus_collection_name)       # 原文表
        self.summary_collection = Collection(settings.milvus_community_summary_collection)  # 摘要表
        self.memory_collection = Collection(settings.milvus_chat_memory_collection)

        # 将常用 collection 预加载，避免首次查询抖动
        self.chunk_collection.load()
        self.summary_collection.load()
        self.memory_collection.load()

        # 3. 挂载 Neo4j (用于 LOCAL 模式的图谱线索召回)
        self.neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )

    def close(self):
        self.neo4j_driver.close()

    def _rrf(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        graph_results: List[Dict] = None,
        k: int = 60,
        id_field: str = "chunk_id",
    ) -> List[Dict]:
        """
        【倒数秩融合算法 (Reciprocal Rank Fusion) - 三路版】
        将语义、关键字、图谱三路召回的结果打分融合。
        """
        rrf_scores = {}
        chunk_data_map = {}  # 用于保存实际内容

        # 定义一个内部闭包处理每一路数据
        def process_results(results, weight=1.0):
            if not results:
                return
            for rank, item in enumerate(results):
                cid = item.get(id_field)
                if not cid:
                    continue
                # weight 允许我们微调某一路的权重，通常设为 1.0
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight / (k + rank + 1)
                chunk_data_map[cid] = item

        # 处理三路召回
        process_results(dense_results)
        process_results(sparse_results)
        if graph_results:
            process_results(graph_results, weight=2.0)  # 将图谱召回的切片同样加入 RRF 擂台！

        # 按 RRF 分数倒序排序
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [chunk_data_map[cid] for cid in sorted_cids]

    @staticmethod
    def _normalize_rerank_score(score: Any) -> float:
        try:
            return float(score)
        except Exception:
            return 0.0

    def _encode_query_vector(self, text: str) -> List[float]:
        """兼容 APIEmbeddingModel.encode 返回的 Python 列表结构。"""
        vectors = self.embedding_model.encode([text])
        if not vectors:
            raise ValueError("Embedding 返回为空")
        return vectors[0]

    def _neo4j_label_exists(self, label: str) -> bool:
        """检查 Neo4j 中标签是否存在，避免不存在标签时触发告警日志。"""
        try:
            with self.neo4j_driver.session() as session:
                rows = session.run("CALL db.labels() YIELD label RETURN label")
                labels = {row["label"] for row in rows}
                return label in labels
        except Exception:
            return False

    def _neo4j_property_key_exists(self, key: str) -> bool:
        """检查 Neo4j 中属性键是否存在，避免引用不存在属性键触发告警。"""
        try:
            with self.neo4j_driver.session() as session:
                rows = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")
                keys = {row["propertyKey"] for row in rows}
                return key in keys
        except Exception:
            return False

    @staticmethod
    def _extract_keywords(query: str, max_keywords: int = 6) -> List[str]:
        """提取少量关键词用于 TEXT_MATCH 稀疏召回。"""
        if not query:
            return []
        parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", query)
        seen = set()
        keywords = []
        for p in parts:
            if p in seen:
                continue
            seen.add(p)
            keywords.append(p)
            if len(keywords) >= max_keywords:
                break
        return keywords

    def _build_sparse_keywords(self, query: str, entities: List[str], max_keywords: int = 8) -> List[str]:
        """BM25 关键词构造：entities 非空仅用实体；为空时才走分词兜底。"""
        deduped = []
        seen = set()

        for ent in entities or []:
            token = (ent or "").strip()
            if not token or token in {"当前用户", "[当前用户]"}:
                continue
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= max_keywords:
                return deduped

        # 只要有可用实体，就不再用分词补齐，避免偏离 Router 语义。
        if deduped:
            return deduped

        for token in self._extract_keywords(query, max_keywords=max_keywords):
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= max_keywords:
                break
        return deduped

    def _retrieve_sparse_by_text_match(
        self,
        collection: Collection,
        keywords: List[str],
        text_field: str,
        id_field: str,
        output_fields: List[str],
        search_limit: int,
    ) -> List[Dict]:
        """基于 Milvus BM25 执行稀疏召回：优先 sparse_vector，失败时回退 TEXT_MATCH。"""
        if not keywords:
            return []

        search_data = " ".join(keywords)
        if not search_data.strip():
            return []

        limit = max(1, search_limit)
        has_sparse_vector = any(getattr(f, "name", "") ==
                                "sparse_vector" for f in collection.schema.fields)

        if has_sparse_vector:
            try:
                search_res = collection.search(
                    data=[search_data],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=limit,
                    output_fields=output_fields,
                )

                merged = []
                for hit in search_res[0]:
                    row = {}
                    row_id = None
                    if hasattr(hit, "entity"):
                        row_id = hit.entity.get(id_field)
                        row["sparse_score"] = hit.score
                        for field in output_fields:
                            row[field] = hit.entity.get(field)

                    if row_id is None:
                        row_id = str(getattr(hit, "id", ""))

                    if not row_id:
                        continue

                    row[id_field] = row_id
                    merged.append(row)

                return merged
            except Exception as e:
                logging.warning(
                    "⚠️ [Sparse-BM25] sparse_vector 检索失败，query='%s', error=%s", search_data, e)

        # 回退到 TEXT_MATCH，兼容未建 sparse_vector 的集合。
        escaped = search_data.replace("'", "\\'")
        expr = f"TEXT_MATCH({text_field}, '{escaped}')"
        try:
            rows = collection.query(
                expr=expr,
                limit=limit,
                output_fields=output_fields,
            )
        except Exception as e:
            logging.warning("⚠️ [Sparse-BM25] TEXT_MATCH 检索失败，query='%s', error=%s", search_data, e)
            return []

        merged = []
        for item in rows:
            row_id = item.get(id_field)
            if not row_id:
                continue
            row = {field: item.get(field) for field in output_fields}
            row[id_field] = row_id
            merged.append(row)

        return merged

    def _get_graph_chunks(self, query_vector: List[float]) -> List[Dict]:
        """
        【核心逻辑】：用 Query 向量召回相似 Entity，再从种子实体扩展 1-2 跳召回原著切片证据（参与 RRF）。
        注意：不读取 Agent_Memory 的 memory_evidence_refs。
        """
        if not query_vector:
            return []

        logging.info("🕸️ [图谱轨道] 正在通过 Entity 向量索引召回种子节点...")

        # 1) Query 向量 -> Entity 种子节点 -> 1-2 跳关系证据。
        # 若存在 created_by 属性键，则排除 Agent_Memory 边；不存在则退化为不过滤（避免告警）。
        if self._neo4j_property_key_exists("created_by"):
            base_cypher = """
            CALL db.index.vector.queryNodes('entity_embedding', $entity_top_k, $query_vector)
            YIELD node, score
            WHERE node:Entity
            MATCH path = (node)-[rels*1..2]-(neighbor:Entity)
            WHERE all(rel IN rels WHERE coalesce(rel.created_by, '') <> 'Agent_Memory')
              AND all(rel IN rels WHERE size(coalesce(rel.source_chunk_ids, [])) > 0)
            WITH node, score, rels
            UNWIND rels AS rel
            UNWIND coalesce(rel.source_chunk_ids, []) AS chunk_id
            RETURN chunk_id, count(chunk_id) AS freq, max(score) AS entity_score
            ORDER BY entity_score DESC, freq DESC
            LIMIT $chunk_limit
            """
        else:
            base_cypher = """
            CALL db.index.vector.queryNodes('entity_embedding', $entity_top_k, $query_vector)
            YIELD node, score
            WHERE node:Entity
            MATCH path = (node)-[rels*1..2]-(neighbor:Entity)
            WHERE all(rel IN rels WHERE size(coalesce(rel.source_chunk_ids, [])) > 0)
            WITH node, score, rels
            UNWIND rels AS rel
            UNWIND coalesce(rel.source_chunk_ids, []) AS chunk_id
            RETURN chunk_id, count(chunk_id) AS freq, max(score) AS entity_score
            ORDER BY entity_score DESC, freq DESC
            LIMIT $chunk_limit
            """

        chunk_ids: List[str] = []
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    base_cypher,
                    query_vector=query_vector,
                    entity_top_k=settings.graph_entity_top_k,
                    chunk_limit=settings.graph_chunk_limit,
                )
                for record in result:
                    if record["chunk_id"]:
                        chunk_ids.append(record["chunk_id"])
        except Exception as e:
            logging.warning("⚠️ [LOCAL-Graph] Neo4j Entity 向量检索失败: %s", e)
            return []

        # 去重并保序
        chunk_ids = list(dict.fromkeys(chunk_ids))

        if not chunk_ids:
            return []

        # 2) 去 Milvus 反查原著证据文本
        chunk_text_map: Dict[str, str] = {}
        if chunk_ids:
            chunk_expr = json.dumps(chunk_ids, ensure_ascii=False)
            query_res = self.chunk_collection.query(
                expr=f"chunk_id in {chunk_expr}",
                output_fields=["chunk_id", "text", "pian", "ji", "zhang"],
            )
            chunk_text_map = {hit["chunk_id"]: hit.get("text", "") for hit in query_res}

        graph_results = []
        for cid in chunk_ids:
            if cid in chunk_text_map:
                graph_results.append({"chunk_id": cid, "text": chunk_text_map[cid]})

        return graph_results

    def _retrieve_user_graph_memory_summaries(self, rewritten_query: str, top_k: int = 5) -> List[str]:
        """路线B：先向量命中相关实体，再与“当前用户”边做交集，最后去记忆库语义召回 Top-K。"""
        if not self._neo4j_label_exists("UserMemory"):
            return []
        if not self._neo4j_property_key_exists("memory_evidence_refs"):
            return []

        query_vector = self._encode_query_vector(rewritten_query)
        memory_event_ids: List[str] = []

        # 优先走路线B：Query 向量 -> Top-K 实体 -> 与当前用户关联边取交集。
        created_by_exists = self._neo4j_property_key_exists("created_by")
        event_limit = max(max(1, top_k) * 8, settings.graph_entity_top_k)
        if created_by_exists:
            narrowed_cypher = """
            CALL db.index.vector.queryNodes('entity_embedding', $entity_top_k, $query_vector)
            YIELD node, score
            WHERE node:Entity
            MATCH (u:UserMemory {id: '[当前用户]'})-[r]-(node)
            WHERE coalesce(r.created_by, '') = 'Agent_Memory'
            UNWIND coalesce(r.memory_evidence_refs, []) AS mem_evt
            WITH mem_evt, max(score) AS best_score
            RETURN mem_evt
            ORDER BY best_score DESC
            LIMIT $event_limit
            """
        else:
            narrowed_cypher = """
            CALL db.index.vector.queryNodes('entity_embedding', $entity_top_k, $query_vector)
            YIELD node, score
            WHERE node:Entity
            MATCH (u:UserMemory {id: '[当前用户]'})-[r]-(node)
            UNWIND coalesce(r.memory_evidence_refs, []) AS mem_evt
            WITH mem_evt, max(score) AS best_score
            RETURN mem_evt
            ORDER BY best_score DESC
            LIMIT $event_limit
            """

        fallback_cypher = (
            """
            MATCH (u:UserMemory {id: '[当前用户]'})-[r]->()
            WHERE coalesce(r.created_by, '') = 'Agent_Memory'
            UNWIND coalesce(r.memory_evidence_refs, []) AS mem_evt
            RETURN mem_evt
            LIMIT $event_limit
            """
            if created_by_exists
            else
            """
            MATCH (u:UserMemory {id: '[当前用户]'})-[r]->()
            UNWIND coalesce(r.memory_evidence_refs, []) AS mem_evt
            RETURN mem_evt
            LIMIT $event_limit
            """
        )

        try:
            with self.neo4j_driver.session() as session:
                rows = session.run(
                    narrowed_cypher,
                    query_vector=query_vector,
                    entity_top_k=max(1, settings.graph_entity_top_k),
                    event_limit=event_limit,
                )
                for row in rows:
                    mem_evt = (row["mem_evt"] or "").strip()
                    if mem_evt:
                        memory_event_ids.append(mem_evt)
        except Exception as e:
            logging.warning("⚠️ [用户图记忆] 路线B图谱收敛失败，回退旧逻辑: %s", e)
            with self.neo4j_driver.session() as session:
                rows = session.run(fallback_cypher, event_limit=event_limit)
                for row in rows:
                    mem_evt = (row["mem_evt"] or "").strip()
                    if mem_evt:
                        memory_event_ids.append(mem_evt)

        memory_event_ids = list(dict.fromkeys(memory_event_ids))
        if not memory_event_ids:
            return []

        safe_ids = [eid.replace("\\", "\\\\").replace("'", "\\'") for eid in memory_event_ids]
        expr = "event_id in [" + ",".join([f"'{eid}'" for eid in safe_ids]) + "]"

        try:
            search_res = self.memory_collection.search(
                data=[query_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=min(max(1, top_k), len(memory_event_ids)),
                expr=expr,
                output_fields=["event_id", "summary"],
            )
        except TypeError:
            search_res = self.memory_collection.search(
                data=[query_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=min(max(1, top_k), len(memory_event_ids)),
                filter=expr,
                output_fields=["event_id", "summary"],
            )
        except Exception as e:
            logging.warning("⚠️ [用户图记忆] 语义召回失败: %s", e)
            return []

        summaries = []
        for idx, hit in enumerate(search_res[0], start=1):
            summary = (hit.entity.get("summary") or "").strip()
            if not summary:
                continue
            summaries.append(f"[用户图记忆{idx}] {summary}")
        return summaries

    def retrieve_local(self, query: str, entities: List[str], top_k: int = 6) -> List[str]:
        """
        【微观轨道 (LOCAL)】：查原文！
        流程：Dense + BM25 + Graph -> RRF 融合 -> Reranker 二次精排
        """
        logging.info("🔍 [LOCAL 模式] 正在进行三路召回与重排...")

        # 1. 语义向量召回 (Dense)
        query_vector = self._encode_query_vector(query)
        dense_req = self.chunk_collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=settings.top_k_retrieval,
            output_fields=["chunk_id", "text"]
        )
        dense_results = []
        for hit in dense_req[0]:
            cid = str(hit.id)
            dense_results.append({"chunk_id": cid, "text": hit.entity.get("text", "")})
        logging.info("📌 [LOCAL-Dense] 召回 %d 条", len(dense_results))

        # 2. 关键字全文召回 (Sparse/BM25)
        keywords = self._build_sparse_keywords(query=query, entities=entities, max_keywords=8)
        sparse_results = self._retrieve_sparse_by_text_match(
            collection=self.chunk_collection,
            keywords=keywords,
            text_field="text",
            id_field="chunk_id",
            output_fields=["chunk_id", "text"],
            search_limit=settings.top_k_retrieval,
        )
        logging.info("📌 [LOCAL-Sparse] 实体优先关键词=%s, 命中 %d 条", keywords, len(sparse_results))

        # 3. 图谱实体召回 (Graph)
        graph_results = self._get_graph_chunks(query_vector)
        logging.info("📌 [LOCAL-Graph] 命中 %d 条", len(graph_results))

        # 4. RRF 倒数秩融合 (三路全开)
        # 此时 rrf_fused_chunks 里面可能有 100~180 个去重后的切片
        rrf_fused_chunks = self._rrf(
            dense_results,
            sparse_results,
            graph_results,
            k=settings.rrf_k,
            id_field="chunk_id",
        )
        logging.info("🧩 [LOCAL-RRF] 融合后去重 %d 条", len(rrf_fused_chunks))

        # 【优化：精排前的截断漏斗】
        # 绝对不能把 100 多条全送给昂贵的 Cross-Encoder，我们只取 RRF 得分最高的前 60 条！
        rrf_fused_chunks = rrf_fused_chunks[:settings.top_k_retrieval]

        # 5. Cross-Encoder 终极精排
        if not rrf_fused_chunks:
            return []

        logging.info(f"⚖️ [精排阶段] 对 RRF 粗筛出的 Top {len(rrf_fused_chunks)} 个切片进行 Cross-Encoder 打分...")
        pairs = [[query, chunk["text"]] for chunk in rrf_fused_chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(rrf_fused_chunks, scores):
            chunk["rerank_score"] = self._normalize_rerank_score(score)

        passed_chunks = [
            c for c in rrf_fused_chunks
            if c["rerank_score"] >= settings.rerank_threshold
        ]
        dropped = len(rrf_fused_chunks) - len(passed_chunks)
        logging.info(
            "🧹 [阈值截断] threshold=%.4f, 保留=%d, 剔除=%d",
            settings.rerank_threshold,
            len(passed_chunks),
            dropped,
        )

        if not passed_chunks:
            logging.info("🚫 [LOCAL] 阈值过滤后无可用切片，返回空列表")
            return []

        passed_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return [c["text"] for c in passed_chunks[:top_k]]

    def retrieve_memory_records(self, query: str, entities: List[str], top_k: int = 2) -> List[str]:
        """
        【MEMORY 记忆轨】：对话记忆库做 Dense + Sparse 双路召回，再做内部 RRF 融合。
        不走 Cross-Encoder，避免把私有记忆召回链路变重。
        """
        if self.memory_collection is None:
            logging.info("ℹ️ [MEMORY] 记忆集合不可用，跳过记忆召回")
            return []

        logging.info("🧠 [MEMORY 模式] 正在召回用户历史对话记忆...")
        query_vector = self._encode_query_vector(query)

        dense_req = self.memory_collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=settings.top_k_retrieval,
            output_fields=["event_id", "summary", "user_query", "ai_response", "timestamp"],
        )
        dense_results = []
        for hit in dense_req[0]:
            event_id = hit.entity.get("event_id") or str(hit.id)
            dense_results.append(
                {
                    "memory_id": event_id,
                    "summary": hit.entity.get("summary", ""),
                    "user_query": hit.entity.get("user_query", ""),
                    "ai_response": hit.entity.get("ai_response", ""),
                    "timestamp": hit.entity.get("timestamp", 0),
                }
            )
        logging.info("📌 [MEMORY-Dense] 召回 %d 条", len(dense_results))

        keywords = self._build_sparse_keywords(query=query, entities=entities, max_keywords=8)
        sparse_rows = self._retrieve_sparse_by_text_match(
            collection=self.memory_collection,
            keywords=keywords,
            text_field="summary",
            id_field="event_id",
            output_fields=["event_id", "summary", "user_query", "ai_response", "timestamp"],
            search_limit=settings.top_k_retrieval,
        )
        sparse_results = [
            {
                "memory_id": row.get("event_id"),
                "summary": row.get("summary", ""),
                "user_query": row.get("user_query", ""),
                "ai_response": row.get("ai_response", ""),
                "timestamp": row.get("timestamp", 0),
            }
            for row in sparse_rows
            if row.get("event_id")
        ]
        logging.info("📌 [MEMORY-Sparse] 实体优先关键词=%s, 命中 %d 条", keywords, len(sparse_results))

        fused_memories = self._rrf(
            dense_results=dense_results,
            sparse_results=sparse_results,
            graph_results=None,
            k=settings.rrf_k,
            id_field="memory_id",
        )
        logging.info("🧩 [MEMORY-RRF] 融合后去重 %d 条", len(fused_memories))

        fused_memories = fused_memories[:top_k]
        formatted = []
        for idx, item in enumerate(fused_memories, start=1):
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue
            formatted.append(f"[记忆摘要{idx}] {summary}")

        return formatted

    def retrieve_global(self, query: str, top_k: int = 3) -> List[str]:
        """
        【宏观轨道 (GLOBAL)】：查社区摘要！
        直接做 Dense 召回，它与原文切片维度不同，不参与微观切片的 RRF。
        """
        logging.info("🔭 [GLOBAL 模式] 正在检索全局社区摘要...")
        query_vector = self._encode_query_vector(query)

        search_res = self.summary_collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["summary", "level"]
        )

        summaries = []
        for hit in search_res[0]:
            lvl = hit.entity.get("level")
            text = hit.entity.get("summary")
            summaries.append(f"[社区层级 Level-{lvl} 摘要]：\n{text}")

        return summaries

    @traceable(name="hybrid_retriever_execute", run_type="retriever")
    def execute_retrieval(self, mode: str, rewritten_query: str, entities: List[str] = None) -> str:
        """总控调度器"""
        context_parts = []
        normalized_mode = (mode or "").upper()
        entities = entities or []
        has_current_user = any((e or "").strip() in {"当前用户", "[当前用户]"} for e in entities)

        if normalized_mode == "DIRECT":
            logging.info("💬 [DIRECT 模式] 跳过所有检索，直接返回空上下文")
            return ""

        if normalized_mode == "LOCAL":
            # 将 Router 提取出的 entities 传给 Local 检索
            chunks = self.retrieve_local(rewritten_query, entities, top_k=settings.top_k_rerank)
            if chunks:
                context_parts.append("【检索到的原著片段】：\n" + "\n---\n".join(chunks))

        elif normalized_mode == "GLOBAL":
            # 1. 宏观轨道：检索全局社区摘要（不参与底层 RRF，独立作为上帝视角）
            summaries = self.retrieve_global(rewritten_query, top_k=settings.global_summary_top_k)
            if summaries:
                context_parts.append("【核心宏观剧情概括】：\n" + "\n---\n".join(summaries))

            # 2. 微观轨道：【响应你的优化】依然保留原始切片的向量召回与全文检索！
            # 虽然摘要不参与 RRF，但底层切片本身依然需要通过 RRF+精排 选出最相关的几条作为细节补充。
            logging.info("🌍 [GLOBAL 模式] 正在补充底层原文切片的向量召回...")
            aux_chunks = self.retrieve_local(
                rewritten_query,
                entities=entities,
                top_k=settings.global_detail_chunk_top_k,
            )
            if aux_chunks:
                context_parts.append("【原著细节辅助补充】：\n" + "\n---\n".join(aux_chunks))

        elif normalized_mode == "MEMORY":
            memory_records = self.retrieve_memory_records(
                rewritten_query,
                entities=entities,
                top_k=settings.memory_summary_top_k,
            )
            if memory_records:
                context_parts.append("【用户专属对话记忆】：\n" + "\n---\n".join(memory_records))

            chunks = self.retrieve_local(
                rewritten_query,
                entities=entities,
                top_k=settings.memory_detail_chunk_top_k,
            )
            if chunks:
                context_parts.append("【原著相关切片】：\n" + "\n---\n".join(chunks))
        else:
            logging.warning("⚠️ 未知路由模式: %s，返回空上下文", mode)
            return ""

        # 只在 LOCAL / GLOBAL 下追加用户图谱记忆；MEMORY 模式默认不追加，避免同库子集重复注入。
        if has_current_user and normalized_mode in {"LOCAL", "GLOBAL"}:
            user_graph_mem = self._retrieve_user_graph_memory_summaries(
                rewritten_query=rewritten_query,
                top_k=settings.user_graph_memory_top_k,
            )
            if user_graph_mem:
                context_parts.append("【当前用户图谱记忆召回】：\n" + "\n---\n".join(user_graph_mem))

        return "\n\n====================\n\n".join(context_parts)
