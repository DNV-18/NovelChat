import re
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.config import settings


CH_NUM = "0-9零一二三四五六七八九十百千万两〇"
MARKER_RE = re.compile(rf"第[{CH_NUM}]+(?P<mark>[篇集章])")
KEYWORD_RE = re.compile(rf"第[{CH_NUM}]+[篇集章]")
TITLE_LINE_PREFIX_RE = re.compile(r'^[\s\u3000"“”‘’《》【】\[\]()（）-]*$')

# 标题锚点后允许的起始字符，避免把“第一篇中”“第3章里”判成标题
HEADING_NEXT_ALLOWED_CHARS = set(" \t\u3000\n\r-—~·•|/（）()[]【】《》<>:：,，.。!?！？;；\"“”'‘’第")


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def is_heading_line(line: str, max_line_chars: int | None = None) -> bool:
    if max_line_chars is None:
        max_line_chars = settings.heading_max_line_chars

    stripped = normalize_line(line)
    if not stripped:
        return False
    if len(stripped) > max_line_chars:
        return False
    if "第" not in stripped:
        return False

    matches = list(MARKER_RE.finditer(stripped))
    if not matches:
        return False

    # 校验每个“第X篇/集/章”后紧邻字符，防止正文短行误匹配
    for m in matches:
        if m.end() < len(stripped):
            ch = stripped[m.end()]
            if ch not in HEADING_NEXT_ALLOWED_CHARS:
                return False

    prefix = stripped[: matches[0].start()]
    if not TITLE_LINE_PREFIX_RE.match(prefix):
        return False

    return True


def split_segments_from_line(line: str) -> List[Tuple[str, str]]:
    """将一行中的多个“第X篇/集/章”按锚点拆分成片段。"""
    stripped = normalize_line(line)
    matches = list(MARKER_RE.finditer(stripped))
    segments: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(stripped)
        seg = stripped[start:end].strip(" \u3000-—~·•|/（）()[]【】《》<>")
        segments.append((seg, m.group("mark")))
    return segments


class NovelParser:
    """
    小说物理结构解析器
    专门针对《吞噬星空》的三级结构：篇 -> (集) -> 章 进行宏观切分
    """
    def __init__(self, file_path: str, encoding: str | None = None):
        self.file_path = Path(file_path)
        self.encoding = encoding or settings.novel_file_encoding

    def parse(self, show_progress: bool = True) -> List[Dict[str, str]]:
        """
        解析整本小说，返回结构化的章节列表
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"找不到小说文件: {self.file_path}")

        print(f"📖 开始解析小说宏观结构: {self.file_path}")

        text = self.file_path.read_bytes().decode(self.encoding, errors="replace")
        replacement_count = text.count("\ufffd")
        if replacement_count:
            print(f"⚠️ 解码替换字符数: {replacement_count}，建议确认编码是否正确")

        lines = text.splitlines()

        chapters = []

        current_pian = "未命名篇"
        current_ji = ""
        current_zhang = ""
        current_content: List[str] = []

        def save_current_chapter():
            """当遇到新章节或文件结束时，保存上一个章节。"""
            if current_zhang:
                chapters.append({
                    "pian": current_pian,
                    "ji": current_ji,
                    "zhang": current_zhang,
                    "content": "\n".join(current_content).strip()
                })
                current_content.clear()

        line_iter = lines
        if show_progress:
            line_iter = tqdm(lines, desc="Stage1/4 结构切分", unit="line")

        for line in line_iter:
            cleaned = normalize_line(line)
            if not cleaned:
                continue

            if KEYWORD_RE.search(cleaned) and is_heading_line(cleaned):
                segments = split_segments_from_line(cleaned)
                for seg_text, mark in segments:
                    save_current_chapter()
                    if mark == "篇":
                        current_pian = seg_text
                        current_ji = ""
                        current_zhang = ""
                    elif mark == "集":
                        current_ji = seg_text
                        current_zhang = ""
                    else:
                        current_zhang = seg_text
                continue

            if current_zhang:
                current_content.append(cleaned)

        save_current_chapter()

        print(f"✅ 宏观解析完成！共提取出 {len(chapters)} 个物理章节。")
        return chapters

# ==========================================
# 独立测试入口 (供你本地 Debug 验证)
# ==========================================
if __name__ == "__main__":
    parser = NovelParser(
        file_path=str(settings.novel_raw_file_path),
        encoding=settings.novel_file_encoding,
    )
    parsed_data = parser.parse()
    
    # 打印前 3 章看看结构对不对
    for i, chapter in enumerate(parsed_data[:3]):
        print(f"\n--- 提取切片 {i+1} ---")
        print(f"层级: [{chapter['pian']}] -> [{chapter['ji']}] -> [{chapter['zhang']}]")
        print(f"正文字数: {len(chapter['content'])} 字")
        print(f"正文预览: {chapter['content'][:50]}...")