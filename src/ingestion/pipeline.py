import os
import json
import asyncio
from typing import List, Dict
from tqdm import tqdm

from src.ingestion.document_parser import NovelParser
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.context_injector import ContextInjector
from src.utils.model_factory import ModelFactory
from src.config import BASE_DIR, settings

class IngestionPipeline:
    """
    Phase 1: 离线数据处理总控流水线
    负责将原始 txt 小说，经过物理切分、语义微切分、上下文注入，最终落盘为黄金存档点 JSON。
    """
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path
        # 定义输出路径：/data/processed/enriched_chunks.json
        self.processed_dir = BASE_DIR / "data" / "processed"
        self.output_file = self.processed_dir / "enriched_chunks.json"
        
        # 确保输出目录存在
        os.makedirs(self.processed_dir, exist_ok=True)

    async def run(self):
        print(f"🚀 开始执行 Phase 1 数据注入流水线...\n" + "="*50)
        stage_bar = tqdm(total=4, desc="Pipeline 阶段", unit="stage")

        # ---------------------------------------------------------
        # Step 1: 物理宏观切片 (按“篇-章”切分)
        # ---------------------------------------------------------
        parser = NovelParser(
            file_path=self.raw_file_path,
            encoding=settings.novel_file_encoding,
        )
        macro_chapters = parser.parse(show_progress=True)
        if not macro_chapters:
            print("❌ 解析失败，未提取到任何章节。")
            stage_bar.close()
            return
        stage_bar.update(1)

        # ---------------------------------------------------------
        # Step 2: 语义微切分 (按 Embedding 相似度断崖切分)
        # ---------------------------------------------------------
        # 从模型工厂单例获取 Embedding 模型，防止重复加载
        embedding_model = ModelFactory.get_embedding_model()
        chunker = SemanticChunker(
            embedding_model=embedding_model, 
            max_tokens=settings.chunk_max_tokens,
            tokenizer_model=settings.local_tokenizer_path,
        )
        micro_chunks = chunker.process_chapters(macro_chapters, show_progress=True)
        stage_bar.update(1)

        # ---------------------------------------------------------
        # Step 3: 上下文注入 (调用大模型补充前置背景)
        # ---------------------------------------------------------
        # ContextInjector 内部通过 ModelFactory 统一调度 LLM
        injector = ContextInjector.from_settings(
            model_name=settings.cheap_llm_model,
            max_concurrency=settings.context_inject_max_concurrency,
            model_tier= "smart"
        )
        
        # 执行异步高并发请求
        enriched_chunks = await injector.inject_context_async(
            micro_chunks=micro_chunks, 
            original_chapters=macro_chapters,
            show_progress=True,
        )
        stage_bar.update(1)

        # ---------------------------------------------------------
        # Step 4: 黄金存档点落盘 (存入 /data/processed)
        # ---------------------------------------------------------
        print("\n💾 正在将处理结果持久化保存到本地文件系统...")
        self._save_to_disk(enriched_chunks)
        stage_bar.update(1)
        stage_bar.close()
        
        print("\n" + "="*50)
        print("🎉 Phase 1 完美收官！")
        print(f"👉 下一步你可以:")
        print(f"   1. 读取 {self.output_file.name} 灌入 Milvus 数据库。")
        print(f"   2. 读取 {self.output_file.name} 传给 Phase 2 提取 GraphRAG 图谱。")

    def _save_to_disk(self, data: List[Dict]):
        """将最终结果保存为 JSON 文件"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 数据已安全保存至: {self.output_file}")


# ==========================================
# 独立运行入口
# ==========================================
if __name__ == "__main__":
    # 默认走配置中的原始小说路径
    raw_novel_path = str(settings.novel_raw_file_path)
    
    pipeline = IngestionPipeline(raw_file_path=raw_novel_path)
    
    # 启动异步事件循环
    asyncio.run(pipeline.run())