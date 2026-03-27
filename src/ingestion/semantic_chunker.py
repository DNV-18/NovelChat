import re
import numpy as np
from typing import Dict, List
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.utils.model_factory import ModelFactory
from src.config import settings

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None

class SemanticChunker:
    """
    语义微切分器
    将“章”级别的长文本，利用 Embedding 模型的语义相似度，智能切分为 500-800 Token 的微观 Chunk。
    """
    def __init__(
        self,
        embedding_model,
        max_tokens: int = 800,
        tokenizer_model: str | None = None,
    ):
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.tokenizer = self._build_tokenizer(tokenizer_model=tokenizer_model)

    def _build_tokenizer(self, tokenizer_model: str | None):
        """
        始终优先使用本地 Qwen tokenizer，失败后回退到 tiktoken。
        """
        local_path = tokenizer_model or settings.local_tokenizer_path
        model_ref = settings.smart_llm_model

        if AutoTokenizer is not None:
            # 先走本地路径，避免触发外网访问
            try:
                tok = AutoTokenizer.from_pretrained(
                    local_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                print(f"✅ 使用 transformers tokenizer(本地路径): {local_path}")
                return tok
            except Exception:
                pass

        # 回退策略：优先尝试模型名映射，再用通用编码
        try:
            enc = tiktoken.encoding_for_model(model_ref)
            print(f"⚠️ 回退到 tiktoken encoding_for_model: {model_ref}")
            return enc
        except Exception:
            print("⚠️ 回退到 tiktoken cl100k_base")
            return tiktoken.get_encoding("cl100k_base")

    def _token_len(self, text: str) -> int:
        """兼容 transformers tokenizer 与 tiktoken 的 token 计数。"""
        if hasattr(self.tokenizer, "encode"):
            try:
                # tiktoken.encode 返回 list[int]，transformers 也可返回 list[int]
                return len(self.tokenizer.encode(text))
            except TypeError:
                # 部分 HF tokenizer encode 需要显式参数
                return len(self.tokenizer.encode(text, add_special_tokens=False))
        if callable(self.tokenizer):
            out = self.tokenizer(text)
            return len(out.get("input_ids", []))
        raise RuntimeError("Unsupported tokenizer type")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        按照中文常见断句符把段落切成句子，并保留标点符号。
        """
        # 使用正则在 句号、问号、叹号、省略号、换行符 处切分
        sentences = re.split(r'([。！？\n]+)', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            # 将句子和后面的标点拼起来
            s = sentences[i] + sentences[i+1]
            if s.strip():
                result.append(s.strip())
        # 处理可能剩下的最后一句没有标点的话
        if len(sentences) % 2 != 0 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        return result

    def chunk_chapter(self, chapter_text: str, percentile_threshold: int | None = None) -> List[str]:
        """
        核心算法：对一章的内容进行语义切分
        :param percentile_threshold: 断崖阈值百分位（默认 90，表示只在差异最大的前 10% 处砍一刀）
        """
        if percentile_threshold is None:
            percentile_threshold = settings.semantic_breakpoint_percentile

        sentences = self._split_into_sentences(chapter_text)
        if not sentences:
            return []
        
        # 1. 如果整章特别短，直接返回
        if self._token_len(" ".join(sentences)) <= self.max_tokens:
            return [" ".join(sentences)]

        # 2. 计算所有句子的 Embedding
        # print("正在计算句子向量...")
        embeddings = self.embedding_model.encode(sentences)

        # 3. 计算相邻句子的余弦相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)

        # 4. 寻找切分点 (相似度极度下跌的“断崖”)
        # 巧妙做法：不设固定值，而是取这章文本里，相似度变化幅度的第 90 百分位的低值
        if similarities:
            threshold = np.percentile(similarities, 100 - percentile_threshold)
        else:
            threshold = 1.0

        print("相似度断崖阈值: {:.4f}".format(threshold))
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i)

        # 5. 根据切分点组装 Chunk
        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = self._token_len(sentence)
            
            # 强制熔断机制：如果加上这句话超过了最大 Token 限制，不管语义断没断，强制砍一刀
            if current_length + sentence_length > self.max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
                continue

            current_chunk.append(sentence)
            current_length += sentence_length

            # 如果当前句子是一个语义断点，顺势砍一刀
            if i in breakpoints:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_chapters(
        self,
        chapters: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> List[Dict[str, str]]:
        """
        接收 Phase 1 的章节列表，返回处理后的微观 Chunk 列表
        """
        micro_chunks = []
        global_chunk_id = 0

        chapter_iter = chapters
        if show_progress:
            chapter_iter = tqdm(chapters, desc="Stage2/4 语义切分", unit="chapter")

        for chapter in chapter_iter:
            print(f"🔪 正在语义切分: [{chapter['pian']}]-[{chapter['zhang']}]")
            chunks_text = self.chunk_chapter(chapter["content"])
            
            for i, text in enumerate(chunks_text):
                micro_chunks.append({
                    "chunk_id": f"chunk_{global_chunk_id:06d}",
                    "pian": chapter["pian"],
                    "ji": chapter["ji"],
                    "zhang": chapter["zhang"],
                    "chunk_index": i, # 记录这是本章的第几个切片
                    "original_text": text,
                    "enriched_text": "", # 留给 Step 3 注入上下文
                    "context_prefix": "", # 留给 Step 3 填充上下文摘要
                })
                global_chunk_id += 1
                
        print(f"✅ 语义切分完成！共生成 {len(micro_chunks)} 个微观 Chunk。")
        return micro_chunks


if __name__ == "__main__":
    from src.ingestion.document_parser import NovelParser

    TEST_SAMPLE_CHAPTERS = 2

    print("=" * 60)
    print("Phase1 + Phase2 联调测试")
    print("=" * 60)

    parser = NovelParser(
        file_path=str(settings.novel_raw_file_path),
        encoding=settings.novel_file_encoding,
    )
    chapters = parser.parse()
    sample_chapters = chapters[:TEST_SAMPLE_CHAPTERS]

    embedding_model = ModelFactory.get_embedding_model()

    tokenizer_model = settings.local_tokenizer_path
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        max_tokens=settings.chunk_max_tokens,
        tokenizer_model=tokenizer_model,
    )

    results = chunker.process_chapters(sample_chapters)
    print("-" * 60)
    print(f"测试章节数: {len(sample_chapters)}")
    print(f"生成切片数: {len(results)}")

    for r in results[:5]:
        tok = chunker._token_len(r["original_text"])
        print(f"[{r['chunk_id']}] tokens={tok} {r['pian']} -> {r['ji']} -> {r['zhang']}")