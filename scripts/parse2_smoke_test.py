import argparse
import asyncio
import json
import uuid

from neo4j import GraphDatabase
from pymilvus import connections, db, utility

from src.config import BASE_DIR, settings
from src.graphrag.community_summarizer import CommunitySummarizer
from src.graphrag.graph_extractor import GraphExtractor
from src.utils.model_factory import ModelFactory


class Parse2SmokeTest:
    """抽样执行 Parse2 全流程，并在结束后清理测试数据。"""

    def __init__(self, sample_size: int, extract_concurrency: int, summary_concurrency: int):
        self.sample_size = sample_size
        self.extract_concurrency = extract_concurrency
        self.summary_concurrency = summary_concurrency
        self.run_id = f"smoke_{uuid.uuid4().hex[:10]}"
        self.temp_json_path = (
            BASE_DIR / "data" / "processed" / f"parse2_smoke_chunks_{self.run_id}.json"
        )
        self.original_summary_collection = settings.milvus_community_summary_collection
        self.test_summary_collection = f"{self.original_summary_collection}_{self.run_id}"

    @staticmethod
    def _neo4j_auth() -> tuple[str, str]:
        return settings.neo4j_username, settings.neo4j_password

    def _assert_neo4j_is_clean(self):
        """为避免误删业务数据，测试前要求图数据库中没有 Entity/Community。"""
        user, pwd = self._neo4j_auth()
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(user, pwd))
        try:
            with driver.session() as session:
                row = session.run(
                    "MATCH (n) WHERE n:Entity OR n:Community RETURN count(n) AS cnt"
                ).single()
                count = int(row["cnt"]) if row and row.get("cnt") is not None else 0

            if count > 0:
                raise RuntimeError(
                    f"Neo4j 中已有 {count} 个 Entity/Community 节点。"
                    "为防止误删，smoke test 已中止。请先在测试环境清空图数据后再运行。"
                )
        finally:
            driver.close()

    def _build_sample_file(self):
        source_file = BASE_DIR / "data" / "processed" / "enriched_chunks.json"
        if not source_file.exists():
            raise FileNotFoundError(f"找不到输入文件: {source_file}")

        with open(source_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("输入 chunk 文件为空或格式错误。")

        sampled = chunks[: self.sample_size]
        output = []
        for idx, chunk in enumerate(sampled, start=1):
            row = dict(chunk)
            origin_id = str(row.get("chunk_id", f"chunk_{idx}"))
            row["chunk_id"] = f"{self.run_id}_{origin_id}"
            output.append(row)

        self.temp_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.temp_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✅ 已生成测试样本: {self.temp_json_path} (共 {len(output)} 条)")

    async def run(self):
        self._assert_neo4j_is_clean()
        self._build_sample_file()

        user, pwd = self._neo4j_auth()
        settings.milvus_community_summary_collection = self.test_summary_collection

        extractor = None
        summarizer = None

        try:
            extractor = GraphExtractor(
                neo4j_uri=settings.neo4j_uri,
                neo4j_user=user,
                neo4j_pwd=pwd,
            )

            print("\n[1/2] 开始执行 GraphExtractor ...")
            summary = await extractor.process_all_chunks(
                json_file_path=str(self.temp_json_path),
                max_concurrency=self.extract_concurrency,
            )
            print("✅ GraphExtractor 完成:", json.dumps(summary, ensure_ascii=False))

            print("\n[2/2] 开始执行 CommunitySummarizer ...")
            embedding_model = ModelFactory.get_embedding_model()
            summarizer = CommunitySummarizer(
                neo4j_uri=settings.neo4j_uri,
                neo4j_user=user,
                neo4j_pwd=pwd,
                embedding_model=embedding_model,
                milvus_uri=settings.milvus_uri,
                summary_max_concurrency=self.summary_concurrency,
            )
            await summarizer.process_and_summarize()
            print("✅ CommunitySummarizer 完成")
        finally:
            settings.milvus_community_summary_collection = self.original_summary_collection
            if extractor is not None:
                extractor.close()
            if summarizer is not None:
                summarizer.driver.close()
            try:
                self._cleanup()
            except Exception as e:
                print(f"⚠️ 清理阶段出现异常，请手动检查: {e}")

    def _cleanup(self):
        """清理本次测试产生的 Milvus 集合、Neo4j 图数据和临时文件。"""
        print("\n🧹 开始清理测试数据...")

        connections.connect("default", uri=settings.milvus_uri)
        existing_dbs = db.list_database()
        if settings.milvus_db_name in existing_dbs:
            connections.connect(
                "default",
                uri=settings.milvus_uri,
                db_name=settings.milvus_db_name,
            )
        if utility.has_collection(self.test_summary_collection):
            utility.drop_collection(self.test_summary_collection)
            print(f"✅ 已删除 Milvus 测试集合: {self.test_summary_collection}")
        else:
            print(f"ℹ️ Milvus 测试集合不存在，跳过: {self.test_summary_collection}")

        user, pwd = self._neo4j_auth()
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(user, pwd))
        try:
            with driver.session() as session:
                session.run("MATCH (n) WHERE n:Entity OR n:Community DETACH DELETE n")
            print("✅ 已清理 Neo4j 测试图数据 (Entity/Community)")
        finally:
            driver.close()

        if self.temp_json_path.exists():
            self.temp_json_path.unlink()
            print(f"✅ 已删除临时样本文件: {self.temp_json_path}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("参数必须 >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse2 全流程 smoke test")
    parser.add_argument("--sample-size", type=_positive_int, default=8, help="抽样 chunk 数量")
    parser.add_argument("--extract-concurrency", type=_positive_int, default=2, help="GraphExtractor 并发")
    parser.add_argument("--summary-concurrency", type=_positive_int, default=2, help="CommunitySummarizer 并发")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = Parse2SmokeTest(
        sample_size=args.sample_size,
        extract_concurrency=args.extract_concurrency,
        summary_concurrency=args.summary_concurrency,
    )
    asyncio.run(runner.run())
