import asyncio
from typing import Dict, List
from tqdm import tqdm

from src.config import settings
from src.utils.model_factory import ModelFactory
from src.utils.prompts import (
    CONTEXT_INJECTOR_SYSTEM_PROMPT,
    build_context_injector_user_prompt,
)

class ContextInjector:
    """
    上下文注入器 (Contextual Retrieval 核心环节)
    利用大模型为脱离了上下文的微观 Chunk 补充前置背景（指代消解、时间地点人物归位）。
    由于 Chunk 数量庞大，采用异步高并发方式请求大模型 API。
    """
    def __init__(
        self,
        model_tier: str | None = None,
        model_name: str | None = None,
        max_concurrency: int | None = None,
    ):
        self.model_tier = model_tier or settings.context_inject_model_tier
        if model_name is not None:
            self.model_name = model_name
        elif self.model_tier == "smart":
            self.model_name = settings.smart_llm_model
        else:
            self.model_name = settings.cheap_llm_model
        # 限制并发量
        resolved_concurrency = max_concurrency or settings.context_inject_max_concurrency
        self.semaphore = asyncio.Semaphore(resolved_concurrency)

    @classmethod
    def from_settings(
        cls,
        model_tier: str | None = None,
        model_name: str | None = None,
        max_concurrency: int | None = None,
    ) -> "ContextInjector":
        """使用项目配置创建注入器实例。"""
        return cls(
            model_tier=model_tier or settings.context_inject_model_tier,
            model_name=model_name,
            max_concurrency=max_concurrency or settings.context_inject_max_concurrency,
        )

    @staticmethod
    def _extract_response_text(response) -> str:
        """从 OpenAI 兼容响应中尽可能提取可用文本。"""
        try:
            message = response.choices[0].message
        except Exception:
            return ""

        content = getattr(message, "content", None)
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        elif isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                    if txt:
                        parts.append(str(txt))
                else:
                    txt = getattr(item, "text", "")
                    if txt:
                        parts.append(str(txt))
            text = "".join(parts).strip()
            if text:
                return text

        # 某些本地模型会把可见输出放在 reasoning 字段
        for field in ("reasoning_content", "reasoning"):
            val = getattr(message, field, None)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return ""

    async def _process_single_chunk(self, chunk: Dict[str, str], chapter_content: str) -> Dict[str, str]:
        """
        处理单个 Chunk：调用大模型获取 Context，并拼接到 enriched_text 中。
        """
        original_text = chunk.get("original_text", "")
        if not original_text.strip():
            chunk["enriched_text"] = ""
            chunk["context_prefix"] = "空文本"
            return chunk

        prompt = build_context_injector_user_prompt(chapter_content, original_text)
        
        async with self.semaphore:
            try:
                response = await ModelFactory.chat_completion_async(
                    messages=[
                        {"role": "system", "content": CONTEXT_INJECTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    model_tier=self.model_tier,
                    model_name=self.model_name,
                    temperature=0.1,
                    max_tokens=16384,
                )
                context_prefix = self._extract_response_text(response)

                if not context_prefix:
                    finish_reason = "unknown"
                    try:
                        finish_reason = response.choices[0].finish_reason
                    except Exception:
                        pass
                    print(
                        f"⚠️ Chunk {chunk.get('chunk_id', 'N/A')} 首次响应为空，"
                        f"finish_reason={finish_reason}，执行一次降级重试..."
                    )

                    retry_prompt = (
                        f"{prompt}\n"
                        "强制要求：必须输出一行中文背景信息；"
                        "若信息不足，请输出“人物与场景不明，待补全文。”"
                    )
                    retry_resp = await ModelFactory.chat_completion_async(
                        messages=[
                            {"role": "system", "content": CONTEXT_INJECTOR_SYSTEM_PROMPT},
                            {"role": "user", "content": retry_prompt},
                        ],
                        model_tier=self.model_tier,
                        model_name=self.model_name,
                        temperature=0.0,
                        max_tokens=16384,
                    )
                    context_prefix = self._extract_response_text(retry_resp)

                if not context_prefix:
                    context_prefix = "人物与场景不明，待补全文。"
                
                # 组装：[补充的上下文] + 原文
                enriched_text = f"【上下文背景】：{context_prefix}\n{original_text}"
                
                # 更新 Chunk 数据
                chunk["enriched_text"] = enriched_text
                chunk["context_prefix"] = context_prefix # 顺便单独存一份方便 debug
                
            except Exception as e:
                print(f"❌ Chunk {chunk['chunk_id']} 处理失败: {e}")
                # 失败降级：如果大模型报错，保留原文，防止流程中断
                chunk["enriched_text"] = original_text
                chunk["context_prefix"] = "提取失败"

        return chunk

    async def inject_context_async(
        self,
        micro_chunks: List[Dict[str, str]],
        original_chapters: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> List[Dict[str, str]]:
        """
        并发处理所有的 Chunk
        :param micro_chunks: Phase 1 Step 2 输出的微观 Chunk 列表
        :param original_chapters: Phase 1 Step 1 输出的宏观章节列表（为了给大模型提供整章视野）
        """
        print(f"💉 开始为 {len(micro_chunks)} 个 Chunk 异步注入上下文...")
        
        # 建立一个快速查找表：通过篇-集-章的名字，快速找到这一章的完整正文
        chapter_map = {}
        for ch in original_chapters:
            key = f"{ch['pian']}_{ch['ji']}_{ch['zhang']}"
            chapter_map[key] = ch.get("content", "")

        tasks = []
        for chunk in micro_chunks:
            key = f"{chunk['pian']}_{chunk['ji']}_{chunk['zhang']}"
            # 找到这个 Chunk 属于哪一章，把那一整章的内容也传过去
            chapter_content = chapter_map.get(key, "")
            
            # 创建异步任务
            task = asyncio.create_task(self._process_single_chunk(chunk, chapter_content))
            tasks.append(task)

        pbar = None
        if show_progress:
            pbar = tqdm(total=len(tasks), desc="Stage3/4 上下文注入", unit="chunk")

            def _on_done(_):
                if pbar is not None:
                    pbar.update(1)

            for task in tasks:
                task.add_done_callback(_on_done)

        # 挂起等待所有 API 请求并发完成
        enriched_chunks = await asyncio.gather(*tasks)
        if pbar is not None:
            pbar.close()
        print("✅ 上下文注入全部完成！")
        return list(enriched_chunks)


# ==========================================
# Phase 1 整体流水线测试入口
# ==========================================
if __name__ == "__main__":
    from src.ingestion.document_parser import NovelParser
    from src.ingestion.semantic_chunker import SemanticChunker
    from src.utils.model_factory import ModelFactory
    
    async def run_pipeline():
        print("=" * 70)
        print("[Pipeline] 上下文注入联调开始")
        print("=" * 70)
        print(f"[Config] LLM Model: {settings.cheap_llm_model}")
        print(f"[Config] LLM Base URL: {settings.openai_base_url}")
        print(f"[Config] Context Max Concurrency: {settings.context_inject_max_concurrency}")
        print(f"[Config] Chunk Max Tokens: {settings.chunk_max_tokens}")

        print("\n[Stage 1/3] 结构解析 (document_parser) ...")
        parser = NovelParser(
            file_path=str(settings.novel_raw_file_path),
            encoding=settings.novel_file_encoding,
        )
        chapters = parser.parse()
        sample_chapters = chapters[:1]
        print(f"[Stage 1/3] 完成: 总章节 {len(chapters)}，测试章节 {len(sample_chapters)}")
        if sample_chapters:
            first = sample_chapters[0]
            print(f"[Stage 1/3] 示例章节: {first.get('pian', '')} / {first.get('ji', '')} / {first.get('zhang', '')}")

        print("\n[Stage 2/3] 语义切分 (semantic_chunker) ...")
        embedding_model = ModelFactory.get_embedding_model()
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            max_tokens=settings.chunk_max_tokens,
            tokenizer_model=settings.local_tokenizer_path,
        )
        micro_chunks = chunker.process_chapters(sample_chapters)
        print(f"[Stage 2/3] 完成: 生成微观切片 {len(micro_chunks)}")
        if micro_chunks:
            first_chunk = micro_chunks[0]
            print(
                "[Stage 2/3] 示例切片: "
                f"{first_chunk.get('chunk_id', 'N/A')} | "
                f"{first_chunk.get('pian', '')} / {first_chunk.get('ji', '')} / {first_chunk.get('zhang', '')}"
            )

        print("\n[Stage 3/3] 上下文注入 (context_injector) ...")
        injector = ContextInjector.from_settings()
        results = await injector.inject_context_async(micro_chunks, sample_chapters)
        print(f"[Stage 3/3] 完成: 注入结果 {len(results)}")
        
        print("\n[Summary] 注入结果展示 (前5条)")
        print("-" * 70)
        for r in results[:5]:
            print(f"ID: {r['chunk_id']}")
            print(f"原始文本: {r['original_text']}")
            print(f"上下文前缀: {r.get('context_prefix', '')}")
            print(f"注入后文本:\n{r['enriched_text']}")
            print("-" * 70)

        print("[Pipeline] 全部完成")
        print("=" * 70)

    # 运行异步测试
    asyncio.run(run_pipeline())