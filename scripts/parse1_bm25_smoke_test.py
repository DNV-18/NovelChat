import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from pymilvus import Collection, connections, db, utility

from src.config import BASE_DIR, settings
from src.ingestion.context_injector import ContextInjector
from src.ingestion.document_parser import NovelParser
from src.ingestion.milvus_indexer import MilvusIndexer
from src.ingestion.semantic_chunker import SemanticChunker
from src.utils.model_factory import ModelFactory


class Parse1Bm25SmokeTest:
    """抽样执行 Parse1，并验证 Milvus 分词/BM25 能力，最后清理测试数据。"""

    def __init__(self):
        self.run_id = f"parse1_smoke_{uuid.uuid4().hex[:10]}"
        self.test_collection = f"{settings.milvus_collection_name}_{self.run_id}"
        self.temp_chunks_file = BASE_DIR / "data" / "processed" / f"parse1_smoke_chunks_{self.run_id}.json"

        self._original_collection_name = settings.milvus_collection_name

    async def _run_parse1_pre_milvus(self) -> str:
        """参考 context_injector main：解析->切分->上下文注入，抽样前两章。"""
        raw_novel_path = Path(settings.novel_raw_file_path)
        if not raw_novel_path.exists():
            raise FileNotFoundError(f"找不到原始小说文件: {raw_novel_path}")

        print("\n[Stage 1/4] 结构解析 (document_parser) ...")
        parser = NovelParser(
            file_path=str(raw_novel_path),
            encoding=settings.novel_file_encoding,
        )
        chapters = parser.parse(show_progress=True)
        sample_chapters = chapters[:2]
        if not sample_chapters:
            raise RuntimeError("章节解析结果为空，无法继续。")
        print(f"✅ 章节抽样完成: total={len(chapters)}, sampled={len(sample_chapters)}")

        print("\n[Stage 2/4] 语义切分 (semantic_chunker) ...")
        embedding_model = ModelFactory.get_embedding_model()
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            max_tokens=settings.chunk_max_tokens,
            tokenizer_model=settings.local_tokenizer_path,
        )
        micro_chunks = chunker.process_chapters(sample_chapters, show_progress=True)
        if not micro_chunks:
            raise RuntimeError("语义切分结果为空，无法继续。")
        print(f"✅ 语义切分完成: chunks={len(micro_chunks)}")

        print("\n[Stage 3/4] 上下文注入 (context_injector) ...")
        injector = ContextInjector.from_settings()
        enriched_chunks = await injector.inject_context_async(
            micro_chunks=micro_chunks,
            original_chapters=sample_chapters,
            show_progress=True,
        )
        if not enriched_chunks:
            raise RuntimeError("上下文注入结果为空，无法继续。")
        print(f"✅ 上下文注入完成: enriched_chunks={len(enriched_chunks)}")

        print("\n[Stage 4/4] 写入临时 JSON ...")
        self.temp_chunks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.temp_chunks_file, "w", encoding="utf-8") as f:
            json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ 已写入临时数据文件: {self.temp_chunks_file}")

        return str(self.temp_chunks_file)

    @staticmethod
    def _extract_chinese_keyword(text: str) -> str:
        """尽量从文本中提取一个可用于全文检索的关键词。"""
        matches = re.findall(r"[\u4e00-\u9fff]{2,8}", text or "")
        if matches:
            return matches[1]
        return (text or "").strip()[:6]

    @staticmethod
    def _safe_schema_dict(collection: Collection) -> dict[str, Any]:
        schema = collection.schema
        to_dict = getattr(schema, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        # 兜底：手工转 dict
        fields = []
        for f in schema.fields:
            field_dict = {
                "name": f.name,
                "dtype": str(f.dtype),
            }
            if hasattr(f, "params"):
                field_dict["params"] = getattr(f, "params")
            fields.append(field_dict)
        return {"fields": fields}

    @staticmethod
    def _try_text_match_query(
        collection: Collection,
        keyword: str,
    ) -> tuple[bool, str, list, list[str], list[str]]:
        """尝试不同版本表达式，返回可用/不可用表达式清单。"""
        expr_candidates = [
            f'TEXT_MATCH(text, "{keyword}")',
            f'text_match(text, "{keyword}")',
        ]

        supported_exprs: list[str] = []
        unsupported_exprs: list[str] = []
        first_success_expr = ""
        first_success_rows = []

        for expr in expr_candidates:
            success = False
            last_error = ""

            # 1) 旧版 query(expr=...)
            try:
                rows = collection.query(expr=expr, output_fields=["chunk_id", "text"], limit=5)
                success = True
                if not first_success_expr:
                    first_success_expr = expr
                    first_success_rows = rows
            except Exception as e:
                last_error = str(e)
                pass

            # 2) 新版 query(filter=...)
            if not success:
                try:
                    rows = collection.query(filter=expr, output_fields=["chunk_id", "text"], limit=5)
                    success = True
                    if not first_success_expr:
                        first_success_expr = expr
                        first_success_rows = rows
                except Exception as e:
                    last_error = str(e)
                    pass

            if success:
                supported_exprs.append(expr)
            else:
                reason = last_error.replace("\n", " ")[:200] if last_error else "unknown_error"
                unsupported_exprs.append(f"{expr} -> {reason}")

        return bool(first_success_expr), first_success_expr, first_success_rows, supported_exprs, unsupported_exprs

    def _validate_milvus_bm25(self) -> None:
        print("\n🔎 开始验证 Milvus 分词器与 BM25/全文检索能力...")

        connections.connect("default", uri=settings.milvus_uri)
        existing_dbs = db.list_database()
        if settings.milvus_db_name not in existing_dbs:
            raise RuntimeError(f"Milvus 数据库不存在: {settings.milvus_db_name}")

        connections.connect("default", uri=settings.milvus_uri, db_name=settings.milvus_db_name)

        if not utility.has_collection(self.test_collection):
            raise RuntimeError(f"测试集合不存在: {self.test_collection}")

        collection = Collection(self.test_collection)
        collection.load()

        # 1) 基础入库验证
        if collection.num_entities <= 0:
            raise RuntimeError("Parse1 已执行，但 Milvus 集合为空。")
        print(f"✅ 集合入库验证通过: num_entities={collection.num_entities}")

        # 2) Schema 验证：检查 text 字段是否声明了 analyzer
        schema_dict = self._safe_schema_dict(collection)
        text_field = None
        for field in schema_dict.get("fields", []):
            if field.get("name") == "text":
                text_field = field
                break

        if not text_field:
            raise RuntimeError("Schema 中未找到 text 字段。")

        text_params = text_field.get("params", {}) or {}
        enable_analyzer = bool(text_params.get("enable_analyzer", False))
        analyzer_params = text_params.get("analyzer_params")

        if not enable_analyzer:
            raise RuntimeError("text 字段未开启 enable_analyzer，无法进行中文分词全文检索。")

        print(f"✅ text 字段已开启分词器: enable_analyzer={enable_analyzer}, analyzer_params={analyzer_params}")

        # 3) 检索能力验证：尝试 TEXT_MATCH / PHRASE_MATCH 表达式
        sample_rows = collection.query(expr="chunk_id != ''", output_fields=["text"], limit=1)
        if not sample_rows:
            raise RuntimeError("集合中无文本样本，无法执行 BM25 验证。")

        sample_text = str(sample_rows[0].get("text", ""))
        keyword = self._extract_chinese_keyword(sample_text)
        if not keyword:
            raise RuntimeError("无法从样本文本提取关键词，无法验证 BM25。")

        ok, used_expr, hit_rows, supported_exprs, unsupported_exprs = self._try_text_match_query(collection, keyword)
        print(f"ℹ️ 可用表达式: {supported_exprs if supported_exprs else '无'}")
        print(f"ℹ️ 不可用表达式: {unsupported_exprs if unsupported_exprs else '无'}")

        if not ok:
            raise RuntimeError(
                "未能执行 TEXT_MATCH/PHRASE_MATCH 查询。"
                "这通常表示 Milvus 版本或部署未启用全文检索能力。"
            )

        if not hit_rows:
            raise RuntimeError(
                f"全文检索表达式可执行但未命中结果: expr={used_expr}, keyword={keyword}"
            )

        print(f"✅ 全文检索能力验证通过: expr={used_expr}, keyword={keyword}, hits={len(hit_rows)}")

    async def run(self) -> None:
        try:
            print("\n🚀 开始执行 Parse1 抽样流程（前两章）...")
            chunks_file = await self._run_parse1_pre_milvus()

            print("\n📥 开始写入 Milvus 临时测试集合...")
            settings.milvus_collection_name = self.test_collection
            indexer = MilvusIndexer(
                collection_name=self.test_collection,
                dim=settings.milvus_vector_dim,
                milvus_uri=settings.milvus_uri,
            )
            embedding_model = ModelFactory.get_embedding_model()
            indexer.insert_data(
                json_file_path=chunks_file,
                embedding_model=embedding_model,
                batch_size=settings.milvus_insert_batch_size,
            )

            print("✅ 临时集合入库完成")

            self._validate_milvus_bm25()
            print("\n🎉 Parse1 + Milvus BM25 smoke test 全部通过！")
        finally:
            settings.milvus_collection_name = self._original_collection_name
            self.cleanup()

    def cleanup(self) -> None:
        print("\n🧹 开始清理测试数据...")

        try:
            connections.connect("default", uri=settings.milvus_uri)
            existing_dbs = db.list_database()
            if settings.milvus_db_name in existing_dbs:
                connections.connect(
                    "default",
                    uri=settings.milvus_uri,
                    db_name=settings.milvus_db_name,
                )
            if utility.has_collection(self.test_collection):
                utility.drop_collection(self.test_collection)
                print(f"✅ 已删除测试集合: {self.test_collection}")
            else:
                print(f"ℹ️ 测试集合不存在，跳过删除: {self.test_collection}")
        except Exception as e:
            print(f"⚠️ 清理 Milvus 测试集合失败: {e}")

        if self.temp_chunks_file.exists():
            self.temp_chunks_file.unlink()
            print(f"✅ 已删除临时 chunk 文件: {self.temp_chunks_file}")


if __name__ == "__main__":
    runner = Parse1Bm25SmokeTest()
    asyncio.run(runner.run())
