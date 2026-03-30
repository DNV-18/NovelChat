import json
from pathlib import Path

from tqdm import tqdm
from pymilvus import (
    connections,
    db,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Function, 
    FunctionType
)

from src.config import BASE_DIR, settings
from src.utils.model_factory import ModelFactory

class MilvusIndexer:
    """
    Milvus 混合检索建库与数据注入器
    核心特性：不仅保存 Dense 向量，同时为文本字段开启 Analyzer 倒排索引，支持后续的 BM25 检索。
    """
    def __init__(
        self,
        collection_name: str | None = None,
        dim: int | None = None,
        milvus_uri: str | None = None,
    ):
        self._milvus_uri = milvus_uri or settings.milvus_uri
        self._ensure_milvus_db(settings.milvus_db_name)
        self.collection_name = collection_name or settings.milvus_collection_name
        self.dim = dim or settings.milvus_vector_dim
        self.collection = None

    def _ensure_milvus_db(self, db_name: str):
        """确保目标 Milvus 数据库存在，不存在则自动创建。"""
        # 先连接默认数据库，才能执行数据库管理操作。
        connections.connect("default", uri=self._milvus_uri)
        existing_dbs = db.list_database()
        if db_name not in existing_dbs:
            print(f"🛠️ Milvus 数据库 '{db_name}' 不存在，正在自动创建...")
            db.create_database(db_name)
        # 切换到目标数据库供后续 collection 操作使用。
        connections.connect("default", uri=self._milvus_uri, db_name=db_name)

    def create_collection(self):
        """定义表结构 (Schema) 并创建 Collection"""
        if utility.has_collection(self.collection_name):
            print(f"⚠️ 集合 {self.collection_name} 已存在，准备清空重建...")
            utility.drop_collection(self.collection_name)

        # 1. 定义字段
        fields = [
            # 主键，强制要求用我们自己生成的 chunk_id，方便后续和 Neo4j 对应
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="pian", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="ji", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="zhang", dtype=DataType.VARCHAR, max_length=100),
            
            # 【核心基调】：为原文文本开启分词器和倒排索引，支持 BM25 全文检索！
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, 
                        enable_analyzer=True, # 开启分词
                        enable_match=True, # 开启全文匹配能力（TEXT_MATCH/PHRASE_MATCH）
                        analyzer_params={"type": "chinese"}), # 使用中文分词器
            
            # 稠密向量字段
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )

        schema = CollectionSchema(fields, functions=[bm25_function], description="小说混合检索底座 (Dense + BM25)",  enable_dynamic_field=True)
        self.collection = Collection(self.collection_name, schema)

        # 2. 为稠密向量创建 HNSW 索引 (查询速度极快)
        index_params = {
            "metric_type": "COSINE", # 余弦相似度
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index(field_name="dense_vector", index_params=index_params)

        # 3. 为文本字段创建倒排索引，支持 BM25/全文检索
        text_index_params = {
            "index_type": "INVERTED"
        }
        self.collection.create_index(field_name="text", index_params=text_index_params)

        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25"
        }
        self.collection.create_index(field_name="sparse_vector", index_params=sparse_index_params)

        print(f"✅ Milvus 集合 '{self.collection_name}' 创建成功，双路索引已就绪！")

    def insert_data(self, json_file_path: str, embedding_model, batch_size: int | None = None):
        """读取 Phase 1 的 JSON 文件，计算向量并批量入库"""
        print(f"📦 正在加载数据: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        resolved_batch_size = batch_size or settings.milvus_insert_batch_size

        if not chunks:
            print("⚠️ 没有可入库的数据，已跳过。")
            return

        # 确保 collection 已可用
        if self.collection is None:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                self.create_collection()

        print(f"🧮 开始计算向量并分批写入 Milvus，batch_size={resolved_batch_size}")
        inserted = 0
        total = len(chunks)

        for start in tqdm(range(0, total, resolved_batch_size), desc="Milvus 入库", unit="batch"):
            batch = chunks[start:start + resolved_batch_size]

            chunk_ids, pians, jis, zhangs, texts = [], [], [], [], []
            for chunk in batch:
                chunk_ids.append(chunk["chunk_id"])
                pians.append(chunk.get("pian", ""))
                jis.append(chunk.get("ji", ""))
                zhangs.append(chunk.get("zhang", ""))
                # 优先使用注入上下文后的文本，缺失时回退原文
                texts.append(chunk.get("enriched_text") or chunk.get("original_text", ""))

            vectors = embedding_model.encode(texts)
            if vectors and len(vectors[0]) != self.dim:
                raise ValueError(
                    f"向量维度不匹配: collection dim={self.dim}, embedding dim={len(vectors[0])}"
                )

            data = [chunk_ids, pians, jis, zhangs, texts, vectors]
            self.collection.insert(data)
            inserted += len(batch)

        self.collection.flush()  # 强制落盘
        print(f"✅ 成功将 {inserted} / {total} 条数据打入 Milvus 向量引擎！")

if __name__ == "__main__":
    json_file = BASE_DIR / "data" / "processed" / "enriched_chunks.json"

    if not Path(json_file).exists():
        raise FileNotFoundError(f"找不到待入库文件: {json_file}")

    embed_model = ModelFactory.get_embedding_model()
    indexer = MilvusIndexer(
            collection_name=settings.milvus_collection_name,
            dim=settings.milvus_vector_dim,
            milvus_uri=settings.milvus_uri,
        )
    indexer.create_collection()
    indexer.insert_data(str(json_file), embedding_model=embed_model, batch_size=settings.milvus_insert_batch_size)