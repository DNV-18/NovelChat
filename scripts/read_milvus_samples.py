import argparse
import json
from typing import List

from pymilvus import Collection, DataType, connections

from src.config import settings


TARGET_COLLECTIONS = [
    # settings.milvus_collection_name,
    settings.milvus_community_summary_collection,
    # settings.milvus_chat_memory_collection,
]


VECTOR_DTYPES = {DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.SPARSE_FLOAT_VECTOR}


def _all_non_vector_field_names(collection: Collection) -> List[str]:
    return [field.name for field in collection.schema.fields if field.dtype not in VECTOR_DTYPES]


def _build_match_all_expr(collection: Collection) -> str:
    """构建一个尽量通用的筛选表达式，用于抽样读取。"""
    scalar_fields = [
        f
        for f in collection.schema.fields
        if f.dtype not in VECTOR_DTYPES
    ]
    if not scalar_fields:
        raise RuntimeError(f"集合 {collection.name} 不包含可用于 query 的标量字段")

    # 优先使用主键字段构造表达式。
    primary = next((f for f in scalar_fields if getattr(f, "is_primary", False)), scalar_fields[0])
    name = primary.name
    dtype = primary.dtype

    if dtype in {DataType.VARCHAR, DataType.STRING}:
        return f'{name} != ""'
    if dtype in {DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64}:
        return f"{name} >= -9223372036854775808"
    if dtype in {DataType.FLOAT, DataType.DOUBLE}:
        return f"{name} >= -1e308"
    if dtype == DataType.BOOL:
        return f"{name} in [true, false]"

    # 其他类型兜底
    return f"{name} != ''"


def dump_collection_samples(collection_name: str, limit: int) -> None:
    print(f"\n========== Collection: {collection_name} ==========")
    collection = Collection(collection_name, using="default")
    collection.load()
    # print(collection.schema)

    output_fields = _all_non_vector_field_names(collection)
    expr = _build_match_all_expr(collection)

    rows = collection.query(
        expr=expr,
        output_fields=output_fields,
        limit=limit,
    )

    print(f"num_entities={collection.num_entities}")
    print(f"returned_rows={len(rows)}")
    print("fields=", output_fields)
    
    # 【新增】：检查是否有生成失败的无效摘要数据
    failed_keywords = ["该社区摘要暂不可用", "该社区暂不可用", "暂无稳定可总结信息", "该社区暂无可总结内容", "暂无可总结内容"]
    valid_rows = []
    failed_rows = []
    
    for row in rows:
        summary_text = row.get("summary", "")
        # 简单匹配几个生成失败时候可能回填的占位符文本
        is_failed = any(kw in summary_text for kw in failed_keywords) or not summary_text.strip()
        
        if is_failed:
            failed_rows.append(row)
        else:
            valid_rows.append(row)
            
    print(f"\n✅ 有效摘要数: {len(valid_rows)}")
    print(f"❌ 失败/无效摘要数: {len(failed_rows)}")
    
    if failed_rows:
        print("\n⚠️ 以下为部分生成失败的异常数据样本：")
        print(json.dumps(failed_rows[:5], ensure_ascii=False, indent=2, default=str))

    print("\n📄 部分有效数据样本：")
    print(json.dumps(valid_rows[:limit], ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 Milvus 三个集合各 N 条样本（返回全部字段）")
    parser.add_argument("--limit", type=int, default=100, help="每个集合读取条数，默认 10")
    args = parser.parse_args()

    connections.connect(
        alias="default",
        uri=settings.milvus_uri,
        db_name=settings.milvus_db_name,
    )

    for name in TARGET_COLLECTIONS:
        try:
            dump_collection_samples(name, limit=max(1, args.limit))
        except Exception as e:
            print(f"[ERROR] 读取集合失败: {name} -> {e}")


if __name__ == "__main__":
    main()
