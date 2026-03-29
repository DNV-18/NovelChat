import json
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from neo4j import GraphDatabase

from src.config import BASE_DIR, settings
from src.utils.model_factory import ModelFactory
from src.utils.prompts import (
    GRAPH_EXTRACTOR_SYSTEM_PROMPT,
    build_graph_extractor_user_prompt,
)

class GraphExtractor:
    """
    Phase 2: GraphRAG 核心建图器
    负责让 LLM 从 Chunk 中提取实体网，并带入 chunk_id 安全合并到 Neo4j 图数据库。
    """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pwd):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))

    def close(self):
        self.driver.close()

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """兼容多种 OpenAI 兼容返回格式，提取文本内容。"""
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
    def _extract_json_block(text: str) -> str:
        """从模型回复中提取 JSON 文本（支持 ```json fenced block）。"""
        if not text:
            return ""

        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        # 尝试抓取第一个顶层大括号对象
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()
        return text.strip()

    @classmethod
    def _parse_graph_json(cls, raw_text: str) -> Dict[str, List[Dict[str, str]]]:
        """解析并清洗 LLM 返回的图谱 JSON。"""
        default = {"nodes": [], "edges": []}
        json_text = cls._extract_json_block(raw_text)
        if not json_text:
            return default

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return default

        nodes = data.get("nodes", []) if isinstance(data, dict) else []
        edges = data.get("edges", []) if isinstance(data, dict) else []

        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []

        clean_nodes: List[Dict[str, str]] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            node_id = str(n.get("id", "")).strip()
            if not node_id:
                continue
            clean_nodes.append(
                {
                    "id": node_id,
                    "type": str(n.get("type", "未知")).strip() or "未知",
                    "description": str(n.get("description", "")).strip(),
                }
            )

        clean_edges: List[Dict[str, str]] = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            source = str(e.get("source", "")).strip()
            target = str(e.get("target", "")).strip()
            relation = str(e.get("relation", "")).strip()
            if not source or not target or not relation:
                continue
            clean_edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "description": str(e.get("description", "")).strip(),
                }
            )

        return {"nodes": clean_nodes, "edges": clean_edges}

    async def _call_llm_extract(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """调用 LLM 抽取图谱并返回标准化 JSON。"""
        prompt = build_graph_extractor_user_prompt(text)

        response = await ModelFactory.chat_completion_async(
            messages=[
                {"role": "system", "content": GRAPH_EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model_tier=settings.graph_extract_model_tier,
            temperature=0.1,
            max_tokens=16384,
        )

        raw_text = self._extract_response_text(response)
        return self._parse_graph_json(raw_text)

    @staticmethod
    def _format_edges_with_node_meta(graph_data: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """将 nodes 元信息补到 edges，供 Cypher 批量写入。"""
        node_map = {n["id"]: n for n in graph_data.get("nodes", [])}
        formatted: List[Dict[str, str]] = []

        for edge in graph_data.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "")
            if not source or not target or not relation:
                continue

            source_node = node_map.get(source, {"type": "未知", "description": ""})
            target_node = node_map.get(target, {"type": "未知", "description": ""})
            formatted.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "description": edge.get("description", ""),
                    "source_type": source_node.get("type", "未知"),
                    "source_desc": source_node.get("description", ""),
                    "target_type": target_node.get("type", "未知"),
                    "target_desc": target_node.get("description", ""),
                }
            )
        return formatted

    def _save_to_neo4j(self, graph_data: dict, chunk_id: str):
        """
        【核心基调 Cypher】：使用 MERGE 进行安全写入，并挂载外键 chunk_id。
        让 Copilot 参考这个 Cypher 语句去补全逻辑。
        """
        # 这个 Cypher 语句是无价之宝，它保证了节点不会重复，并且会把多个 chunk_id 收集到一个数组里！
        cypher_query = """
        // 1. 合并（不存在则创建，存在则匹配）源节点和目标节点
        UNWIND $edges AS edge
        MERGE (n1:Entity {id: edge.source})
        ON CREATE SET n1.type = edge.source_type, n1.description = edge.source_desc
        
        MERGE (n2:Entity {id: edge.target})
        ON CREATE SET n2.type = edge.target_type, n2.description = edge.target_desc
        
        // 2. 合并它们之间的关系
        WITH n1, n2, edge
        CALL apoc.merge.relationship(n1, edge.relation, 
            {}, // 关系的属性键值对（这里为空，后续可扩展）
            {}, 
            n2, 
            {}
        ) YIELD rel
        
        // 3. 将当前来源文本的 chunk_id 追加到关系的 source_chunk_ids 列表中（去重追加）
        // 这一步是打通局部检索命脉的关键！
        SET rel.description = edge.description
        SET rel.source_chunk_ids = 
            CASE 
                WHEN $chunk_id IN rel.source_chunk_ids THEN rel.source_chunk_ids 
                ELSE coalesce(rel.source_chunk_ids, []) + $chunk_id 
            END
        """
        
        # 整理要传给 Cypher 的参数格式
        formatted_edges = self._format_edges_with_node_meta(graph_data)
        if not formatted_edges:
            return
        
        with self.driver.session() as session:
            session.run(cypher_query, edges=formatted_edges, chunk_id=chunk_id)

    async def _process_single_chunk(self, chunk: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """处理单个 chunk：LLM 抽取 + 图数据入库。"""
        chunk_id = str(chunk.get("chunk_id", ""))
        text = str(chunk.get("enriched_text") or chunk.get("original_text") or "").strip()

        if not chunk_id or not text:
            return {"chunk_id": chunk_id, "ok": False, "reason": "missing_chunk_id_or_text", "nodes": 0, "edges": 0}

        async with semaphore:
            try:
                graph_data = await self._call_llm_extract(text)
                self._save_to_neo4j(graph_data, chunk_id=chunk_id)
                return {
                    "chunk_id": chunk_id,
                    "ok": True,
                    "nodes": len(graph_data.get("nodes", [])),
                    "edges": len(graph_data.get("edges", [])),
                }
            except Exception as e:
                return {
                    "chunk_id": chunk_id,
                    "ok": False,
                    "reason": str(e),
                    "nodes": 0,
                    "edges": 0,
                }

    async def process_all_chunks(self, json_file_path: str, max_concurrency: int = 10) -> Dict[str, Any]:
        """
        读取注入后的 chunk JSON，并发调用 LLM 抽取图谱并写入 Neo4j。
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list):
            raise ValueError("输入 JSON 格式错误：应为 chunk 列表")

        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [asyncio.create_task(self._process_single_chunk(chunk, semaphore)) for chunk in chunks]

        pbar = tqdm(total=len(tasks), desc="Graph 抽取入库", unit="chunk")
        for task in tasks:
            task.add_done_callback(lambda _: pbar.update(1))

        results = await asyncio.gather(*tasks)
        pbar.close()

        success = sum(1 for r in results if r.get("ok"))
        failed = len(results) - success
        total_nodes = sum(int(r.get("nodes", 0)) for r in results)
        total_edges = sum(int(r.get("edges", 0)) for r in results)

        return {
            "total_chunks": len(results),
            "success_chunks": success,
            "failed_chunks": failed,
            "total_nodes_extracted": total_nodes,
            "total_edges_extracted": total_edges,
            "failures": [r for r in results if not r.get("ok")],
        }


if __name__ == "__main__":
    input_file = BASE_DIR / "data" / "processed" / "enriched_chunks.json"

    # 默认认证为 neo4j/neo4j，可由调用方改造为从安全存储读取
    extractor = GraphExtractor(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user="neo4j",
        neo4j_pwd="neo4j",
    )
    try:
        summary = asyncio.run(extractor.process_all_chunks(str(input_file), max_concurrency=8))
        print("Graph 抽取完成:", json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        extractor.close()