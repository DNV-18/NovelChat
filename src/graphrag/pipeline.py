import asyncio

# 导入你的配置和模型工厂
from src.config import BASE_DIR, settings
from src.utils.model_factory import ModelFactory

# 导入 Phase 1 总控流水线（用于自动衔接）
from src.ingestion.pipeline import IngestionPipeline

# 导入 Phase 2 的两个核心工作组件
from src.graphrag.graph_extractor import GraphExtractor
from src.graphrag.node_profiler import NodeProfiler
from src.graphrag.community_summarizer import CommunitySummarizer

class GraphRAGPipeline:
    """
    Phase 2: GraphRAG 离线建图总控流水线
    Step 1: 读取 Phase 1 的 JSON，抽取实体与关系，打入 Neo4j。
    Step 1.5: 汇总 Entity.all_descriptions，生成节点画像与向量索引字段。
    Step 2: 在 Neo4j 中运行 Leiden 社区划分，生成摘要，双写 Neo4j 与 Milvus。
    """
    def __init__(self):
        # 依赖 Phase 1 产出的黄金存档点
        self.input_file = BASE_DIR / "data" / "processed" / "enriched_chunks.json"
        self.phase1_pipeline = IngestionPipeline(raw_file_path=str(settings.novel_raw_file_path))

    async def run(self):
        print("🚀 开始执行 Phase 2: GraphRAG 深度建图流水线...\n" + "="*50)

        # ---------------------------------------------------------
        # 预检：如果缺少 Phase 1 输出，自动补跑 Phase 1
        # ---------------------------------------------------------
        if not self.input_file.exists():
            print(f"⚠️ 未检测到 Phase 1 输出：{self.input_file}")
            print("🔁 自动触发 Phase 1 流水线以生成输入数据...")
            await self.phase1_pipeline.run()

        if not self.input_file.exists():
            raise FileNotFoundError(f"❌ 找不到输入数据！请先运行 Phase 1 确保 {self.input_file} 存在。")

        # ---------------------------------------------------------
        # 准备依赖：加载 Embedding 模型，解析 Neo4j 认证
        # ---------------------------------------------------------
        print("🤖 正在预热 Embedding 模型...")
        # 摘要向量化需要 Embedding 模型
        embedding_model = ModelFactory.get_embedding_model()
        neo4j_user = settings.neo4j_username
        neo4j_pwd = settings.neo4j_password

        # ---------------------------------------------------------
        # Step 1: 实体与关系提取 (Graph Extraction)
        # ---------------------------------------------------------
        print("\n🕸️ [Step 1] 开始从微观 Chunk 中提取实体与拓扑网...")
        extractor = GraphExtractor(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_pwd=neo4j_pwd,
        )
        
        try:
            await extractor.process_all_chunks(str(self.input_file))
        except Exception as e:
            print(f"❌ 实体提取阶段发生错误: {e}")
            raise e
        finally:
            extractor.close()
            print("✅ 实体提取完毕，底层知识图谱已写入 Neo4j！")

        # ---------------------------------------------------------
        # Step 1.5: 节点画像总结与 Entity 向量索引字段写入
        # ---------------------------------------------------------
        print("\n🧬 [Step 1.5] 开始生成 Entity 节点画像与向量...")
        profiler = NodeProfiler(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_pwd=neo4j_pwd,
            embedding_model=embedding_model,
            batch_size=settings.node_profile_batch_size,
            max_concurrency=settings.node_profile_max_concurrency,
            vector_dim=settings.milvus_vector_dim,
        )

        try:
            await profiler.process_all_nodes()
        except Exception as e:
            print(f"❌ 节点画像阶段发生错误: {e}")
            raise e
        finally:
            profiler.close()
            print("✅ 节点画像与 Entity 向量索引字段已就绪！")

        # ---------------------------------------------------------
        # Step 2: 层次化社区划分与摘要生成 (Community Summarization)
        # ---------------------------------------------------------
        print("\n🏙️ [Step 2] 开始运行 Leiden 算法与社区摘要生成...")
        summarizer = CommunitySummarizer(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_pwd=neo4j_pwd,
            embedding_model=embedding_model,
            milvus_uri=settings.milvus_uri,
            summary_max_concurrency=settings.community_summary_max_concurrency,
        )
        
        try:
            # 运行你刚才完美写完的主控流程
            await summarizer.process_and_summarize()
        except Exception as e:
            print(f"❌ 社区摘要生成阶段发生错误: {e}")
            raise e
        finally:
            summarizer.close()
            print("✅ 社区抽象完毕，图谱垂直树与全局摘要向量已构建！")

        print("\n" + "="*50)
        print("🎉 Phase 2 完美收官！你的高级 GraphRAG 底座已彻底建成！")

# ==========================================
# 独立运行入口
# ==========================================
if __name__ == "__main__":
    pipeline = GraphRAGPipeline()
    # 启动异步流水线
    asyncio.run(pipeline.run())
