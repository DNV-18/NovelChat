import logging
from typing import List, Dict, Any
from pymilvus import Collection, connections
from neo4j import GraphDatabase

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
        connections.connect("default", uri=settings.milvus_uri)
        self.chunk_collection = Collection(settings.milvus_collection_name)       # 原文表
        self.summary_collection = Collection(settings.milvus_community_summary_collection) # 摘要表

        # 3. 挂载 Neo4j (用于 LOCAL 模式的图谱线索召回)
        self.neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri, 
            auth=(settings.neo4j_username, settings.neo4j_password)
        )

    def close(self):
        self.neo4j_driver.close()

    def _rrf(self, dense_results: List[Dict], sparse_results: List[Dict], graph_results: List[Dict] = None, k: int = 60) -> List[Dict]:
        """
        【倒数秩融合算法 (Reciprocal Rank Fusion) - 三路版】
        将语义、关键字、图谱三路召回的结果打分融合。
        """
        rrf_scores = {}
        chunk_data_map = {} # 用于保存实际内容
        
        # 定义一个内部闭包处理每一路数据
        def process_results(results, weight=1.0):
            if not results: return
            for rank, item in enumerate(results):
                cid = item["chunk_id"]
                # weight 允许我们微调某一路的权重，通常设为 1.0
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight / (k + rank + 1)
                chunk_data_map[cid] = item

        # 处理三路召回
        process_results(dense_results)
        process_results(sparse_results)
        if graph_results:
            process_results(graph_results) # 将图谱召回的切片同样加入 RRF 擂台！

        # 按 RRF 分数倒序排序
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [chunk_data_map[cid] for cid in sorted_cids]

    def _get_graph_chunks(self, entities: List[str]) -> List[Dict]:
        """
        【核心逻辑】：从 Neo4j 中查找涉及指定实体的边，提取 source_chunk_ids，
        然后去 Milvus 中把对应的原文拉出来。
        """
        if not entities:
            return []
            
        logging.info(f"🕸️ [图谱轨道] 正在 Neo4j 中探索实体: {entities}")
        
        # 1. 使用 Cypher 查出图谱中与这些实体相关的原切片 ID
        cypher = """
        UNWIND $entities AS ent_name
        MATCH (n:Entity {id: ent_name})-[r]-(m:Entity)
        UNWIND r.source_chunk_ids AS chunk_id
        RETURN chunk_id, count(chunk_id) AS freq
        ORDER BY freq DESC
        LIMIT 60
        """
        
        chunk_ids = []
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, entities=entities)
            for record in result:
                if record["chunk_id"]:
                    chunk_ids.append(record["chunk_id"])

        if not chunk_ids:
            return []

        # 2. 根据找出的 chunk_ids，去 Milvus 反查原文文本
        # 注意 expr 的语法: chunk_id in ["chunk_01", "chunk_02"]
        id_list_str = "[" + ",".join([f"'{cid}'" for cid in chunk_ids]) + "]"
        query_res = self.chunk_collection.query(
            expr=f"chunk_id in {id_list_str}",
            output_fields=["chunk_id", "text", "pian", "zhang"]
        )
        
        # 为了保证 RRF 中 rank 的顺序（图谱中连接越紧密的排越前）
        # 我们按照 Neo4j 返回的频率顺序来重新排列查出来的文本
        text_map = {hit["chunk_id"]: hit["text"] for hit in query_res}
        
        graph_results = []
        for cid in chunk_ids:
            if cid in text_map:
                graph_results.append({"chunk_id": cid, "text": text_map[cid]})
                
        return graph_results

    def retrieve_local(self, query: str, entities: List[str], top_k: int = 6) -> List[str]:
        """
        【微观轨道 (LOCAL)】：查原文！
        流程：Dense + BM25 + Graph -> RRF 融合 -> Reranker 二次精排
        """
        logging.info("🔍 [LOCAL 模式] 正在进行三路召回与重排...")
        
        # 1. 语义向量召回 (Dense)
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True).tolist()[0]
        dense_req = self.chunk_collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=settings.top_k_retrieval, # 假设配置里是 60
            output_fields=["chunk_id", "text"]
        )
        dense_results = [{"chunk_id": hit.id, "text": hit.entity.get("text")} for hit in dense_req[0]]
        
        # 2. 关键字全文召回 (Sparse/BM25)
        # 留给 Copilot: 补充 Milvus 全文检索 search 语法
        sparse_results = dense_results # 此处模拟 BM25 结果
        
        # 3. 图谱实体召回 (Graph) 
        graph_results = self._get_graph_chunks(entities)
        
        # 4. RRF 倒数秩融合 (三路全开)
        # 此时 rrf_fused_chunks 里面可能有 100~180 个去重后的切片
        rrf_fused_chunks = self._rrf(dense_results, sparse_results, graph_results, k=settings.rrf_k)
        
        # 【你的神级优化：精排前的截断漏斗】
        # 绝对不能把 100 多条全送给昂贵的 Cross-Encoder，我们只取 RRF 得分最高的前 60 条！
        rrf_fused_chunks = rrf_fused_chunks[:settings.top_k_retrieval]
        
        # 5. Cross-Encoder 终极精排
        if not rrf_fused_chunks:
            return []
            
        logging.info(f"⚖️ [精排阶段] 对 RRF 粗筛出的 Top {len(rrf_fused_chunks)} 个切片进行 Cross-Encoder 昂贵打分...")
        pairs = [[query, chunk["text"]] for chunk in rrf_fused_chunks]
        scores = self.reranker.predict(pairs)
        
        for chunk, score in zip(rrf_fused_chunks, scores):
            chunk["rerank_score"] = score
            
        rrf_fused_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return [c["text"] for c in rrf_fused_chunks[:top_k]]

    def retrieve_global(self, query: str, top_k: int = 3) -> List[str]:
        """
        【宏观轨道 (GLOBAL)】：查社区摘要！
        直接做 Dense 召回，它与原文切片维度不同，不参与微观切片的 RRF。
        """
        logging.info("🔭 [GLOBAL 模式] 正在检索全局社区摘要...")
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True).tolist()[0]
        
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

    def execute_retrieval(self, mode: str, rewritten_query: str, entities: List[str] = None) -> str:
        """总控调度器"""
        context_parts = []
        
        if mode == "LOCAL":
            # 将 Router 提取出的 entities 传给 Local 检索
            chunks = self.retrieve_local(rewritten_query, entities or [], top_k=settings.top_k_rerank)
            if chunks:
                context_parts.append("【检索到的原著片段】：\n" + "\n---\n".join(chunks))

        elif mode == "GLOBAL":
            # 1. 宏观轨道：检索全局社区摘要（不参与底层 RRF，独立作为上帝视角）
            summaries = self.retrieve_global(rewritten_query, top_k=3)
            if summaries:
                context_parts.append("【核心宏观剧情概括】：\n" + "\n---\n".join(summaries))
            
            # 2. 微观轨道：【响应你的优化】依然保留原始切片的向量召回与全文检索！
            # 虽然摘要不参与 RRF，但底层切片本身依然需要通过 RRF+精排 选出最相关的几条作为细节补充。
            logging.info("🌍 [GLOBAL 模式] 正在补充底层原文切片的向量召回...")
            aux_chunks = self.retrieve_local(rewritten_query, entities=[], top_k=3)
            if aux_chunks:
                context_parts.append("【原著细节辅助补充】：\n" + "\n---\n".join(aux_chunks))

        elif mode == "MEMORY":
            # TODO: 查 long_term_events 表
            context_parts.append("【用户长期记忆提取中...】")
            
        return "\n\n====================\n\n".join(context_parts)