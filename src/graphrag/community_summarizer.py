import asyncio
from collections import defaultdict
from typing import Dict, List
import networkx as nx
import igraph as ig
import leidenalg
from tqdm import tqdm
from neo4j import GraphDatabase
from pymilvus import connections, db, utility, FieldSchema, CollectionSchema, DataType, Collection
from src.config import settings
from src.utils.model_factory import ModelFactory
from src.utils.prompts import (
    COMMUNITY_SUMMARY_SYSTEM_PROMPT,
    build_community_summary_user_prompt,
)


class CommunitySummarizer:
    """
    Phase 2: 层次化社区摘要生成与双写引擎
    1. 使用 Leiden 算法进行多层级 (Hierarchical) 社区划分。
    2. 调用 LLM 自底向上生成摘要。
    3. 将树状拓扑写入 Neo4j (仅保留垂直连线)。
    4. 将摘要向量化写入 Milvus (用于极速 Global Search)。
    """
    def __init__(
        self,
        neo4j_uri,
        neo4j_user,
        neo4j_pwd,
        embedding_model,
        milvus_uri=None,
        summary_max_concurrency: int = 5,
    ):
        # 1. 挂载 Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        # 2. 挂载模型
        self.embedding_model = embedding_model
        self.summary_semaphore = asyncio.Semaphore(summary_max_concurrency)
        # 3. 挂载 Milvus 并初始化摘要专属集合
        self._milvus_uri = milvus_uri or settings.milvus_uri
        self._ensure_milvus_db(settings.milvus_db_name)
        self.summary_collection_name = settings.milvus_community_summary_collection
        self._init_milvus_collection()

    def _ensure_milvus_db(self, db_name: str):
        """确保目标 Milvus 数据库存在，不存在则自动创建。"""
        connections.connect("default", uri=self._milvus_uri)
        existing_dbs = db.list_database()
        if db_name not in existing_dbs:
            print(f"🛠️ Milvus 数据库 '{db_name}' 不存在，正在自动创建...")
            db.create_database(db_name)
        connections.connect("default", uri=self._milvus_uri, db_name=db_name)

    def _init_milvus_collection(self):
        """在 Milvus 中初始化一个专门存全局摘要的表"""
        if utility.has_collection(self.summary_collection_name):
            print(f"⚠️ Milvus 集合 {self.summary_collection_name} 已存在，准备清空重建...")
            utility.drop_collection(self.summary_collection_name)

        fields = [
            FieldSchema(name="community_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="level", dtype=DataType.INT64), # 记录这是第几层的摘要
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus_vector_dim) # 替换为你的模型维度
        ]
        schema = CollectionSchema(fields, description="GraphRAG 社区全局摘要库")
        self.collection = Collection(self.summary_collection_name, schema)
        self.collection.create_index(
            field_name="dense_vector", 
            index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
        )

    def _run_hierarchical_leiden(self, G: nx.Graph) -> dict:
        """
        【核心算法基调】：迭代运行 Leiden 算法，直到网络收敛，形成严格的层级树。
        交由 Copilot 补全使用 igraph 转换和循环迭代的具体逻辑。
        """
        # 返回数据结构约定：
        # {
        #    0: [{"community_id": "C_0_1", "nodes": ["罗峰", "洪"]}, ...],
        #    1: [{"community_id": "C_1_1", "sub_communities": ["C_0_1", "C_0_2"]}, ...]
        # }
        print("🧠 正在运行 Leiden 层次化社区发现算法...")
        
        # current_graph = G
        # level = 0
        # hierarchy = {}
        # while True:
        #     1. nx 转 igraph
        #     2. partition = leidenalg.find_partition(igraph_obj, leidenalg.ModularityVertexPartition)
        #     3. 记录当前 level 的 communities，存入 hierarchy[level]
        #     4. 将每个 community 缩点（变成超级节点），构建下一层的 current_graph
        #     5. 如果 current_graph 节点数 == 1，或者 modularity 不再提升，break
        #     level += 1
        
        # return hierarchy
        hierarchy: Dict[int, List[Dict[str, list]]] = {}

        if G.number_of_nodes() == 0:
            print("⚠️ 输入图为空，无法进行社区划分。")
            return hierarchy

        current_graph = G.copy()
        level = 0

        while True:
            if current_graph.number_of_nodes() == 0:
                break

            node_names = list(current_graph.nodes())
            if len(node_names) == 1:
                cid = f"C_{level}_1"
                if level == 0:
                    hierarchy[level] = [{"community_id": cid, "nodes": node_names, "content": node_names}]
                else:
                    hierarchy[level] = [{"community_id": cid, "sub_communities": node_names, "content": node_names}]
                break

            # 1) networkx -> igraph
            name_to_idx = {name: i for i, name in enumerate(node_names)}
            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(node_names))
            ig_graph.vs["name"] = node_names

            edges = []
            weights = []
            for u, v, data in current_graph.edges(data=True):
                if u == v:
                    continue
                edges.append((name_to_idx[u], name_to_idx[v]))
                weights.append(float(data.get("weight", 1.0)))

            if edges:
                ig_graph.add_edges(edges)
                ig_graph.es["weight"] = weights

            # 2) Leiden 分区
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                weights=ig_graph.es["weight"] if ig_graph.ecount() > 0 else None,
            )

            # 3) 记录当前层社区
            level_communities: List[Dict[str, list]] = []
            node_to_comm: Dict[str, str] = {}

            for i, comm in enumerate(partition):
                members = [ig_graph.vs[idx]["name"] for idx in comm]
                community_id = f"C_{level}_{i + 1}"
                for member in members:
                    node_to_comm[member] = community_id

                if level == 0:
                    level_communities.append(
                        {
                            "community_id": community_id,
                            "nodes": members,
                            "content": members,
                        }
                    )
                else:
                    level_communities.append(
                        {
                            "community_id": community_id,
                            "sub_communities": members,
                            "content": members,
                        }
                    )

            hierarchy[level] = level_communities

            # 收敛条件：该层已经只剩一个社区，结束
            if len(level_communities) <= 1:
                break

            # 4) 缩点生成下一层图
            next_graph = nx.Graph()
            for comm in level_communities:
                next_graph.add_node(comm["community_id"])

            edge_weights = defaultdict(float)
            for u, v, data in current_graph.edges(data=True):
                cu = node_to_comm[u]
                cv = node_to_comm[v]
                if cu == cv:
                    continue
                key = tuple(sorted((cu, cv)))
                edge_weights[key] += float(data.get("weight", 1.0))

            for (cu, cv), w in edge_weights.items():
                next_graph.add_edge(cu, cv, weight=w)

            # 如果下一层节点数没有减少，说明无法继续抽象，停止
            if next_graph.number_of_nodes() >= current_graph.number_of_nodes():
                break

            current_graph = next_graph
            level += 1

        return hierarchy

    async def _generate_summary(self, level: int, content_data: list) -> str:
        """根据不同层级，构建不同的 Prompt 调用大模型"""
        if not content_data:
            return "该社区暂无可总结内容。"

        prompt = build_community_summary_user_prompt(level, content_data)

        try:
            async with self.summary_semaphore:
                response = await ModelFactory.chat_completion_async(
                    messages=[
                        {"role": "system", "content": COMMUNITY_SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model_tier=settings.community_summary_model_tier,
                    temperature=0.2,
                    max_tokens=10240,
                )

            message = response.choices[0].message
            content = getattr(message, "content", "")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = "".join(
                    item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                    for item in content
                ).strip()
            else:
                text = ""

            return text or "该社区暂无稳定可总结信息。"
        except Exception as e:
            return f"摘要生成失败：{e}"

    async def _summarize_level_parallel(
        self,
        level: int,
        jobs: list,
        community_pbar,
    ) -> list:
        """在同一层内并发生成社区摘要，层与层之间由外层循环保证串行。"""
        async def _worker(job: dict):
            summary_text = await self._generate_summary(level, job["summary_input"])
            return {
                "community_id": job["community_id"],
                "child_ids": job["child_ids"],
                "summary_text": summary_text,
            }

        tasks = [asyncio.create_task(_worker(job)) for job in jobs]
        results = []
        for task in asyncio.as_completed(tasks):
            results.append(await task)
            community_pbar.update(1)
        return results

    def _save_to_neo4j(self, community_id: str, level: int, child_ids: list, summary_text: str):
        """
        【图谱写入基调】：严格遵守无横向连线原则，仅建立垂直的 BELONGS_TO 树状关系。
        """
        with self.driver.session() as session:
            # 创建当前层的超级节点
            session.run(
                "MERGE (c:Community {id: $cid}) "
                "SET c.level = $lvl, c.summary = $summary",
                cid=community_id,
                lvl=level,
                summary=summary_text,
            )
            
            # 建立下层节点指向上层节点的垂直连线
            if level == 0:
                # Level 0 的孩子是原始 Entity
                cypher = """
                UNWIND $child_ids AS child_id
                MATCH (e:Entity {id: child_id}), (c:Community {id: $cid})
                MERGE (e)-[:BELONGS_TO]->(c)
                """
            else:
                # Level > 0 的孩子是下一层的 Community
                cypher = """
                UNWIND $child_ids AS child_id
                MATCH (sub_c:Community {id: child_id}), (c:Community {id: $cid})
                MERGE (sub_c)-[:SUBCOMMUNITY_OF]->(c)
                """
            if child_ids:
                session.run(cypher, child_ids=child_ids, cid=community_id)

    async def process_and_summarize(self):
        """【主控流水线】"""
        print("📥 正在从 Neo4j 拉取全图拓扑...")
        # 1. 从 Neo4j 拉取所有实体和边，组装成 NetworkX Graph
        G = nx.Graph()
        with self.driver.session() as session:
            node_rows = session.run(
                "MATCH (e:Entity) "
                "RETURN e.id AS id, coalesce(e.type, '未知') AS type, coalesce(e.description, '') AS description"
            )
            for row in node_rows:
                node_id = row["id"]
                if node_id:
                    G.add_node(node_id, type=row["type"], description=row["description"])

            edge_rows = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.id AS source, b.id AS target, type(r) AS relation, coalesce(r.description, '') AS description"
            )
            for row in edge_rows:
                source = row["source"]
                target = row["target"]
                if not source or not target or source == target:
                    continue

                if G.has_edge(source, target):
                    G[source][target]["weight"] += 1.0
                    G[source][target]["relations"].add(row["relation"])
                    if row["description"]:
                        G[source][target]["descriptions"].append(row["description"])
                else:
                    G.add_edge(
                        source,
                        target,
                        weight=1.0,
                        relations={row["relation"]},
                        descriptions=[row["description"]] if row["description"] else [],
                    )

        if G.number_of_nodes() == 0:
            print("⚠️ Neo4j 中未找到 Entity 节点，流程结束。")
            return
        
        # 2. 运行层次化聚类
        hierarchy = self._run_hierarchical_leiden(G)
        if not hierarchy:
            print("⚠️ 未形成有效社区层级，流程结束。")
            return
        
        total_levels = len(hierarchy)
        total_communities = sum(len(comms) for comms in hierarchy.values())
        print("\n📊 Leiden 算法收敛完毕！【层级结构报告】:")
        print(f"   ▶ 共生成 {total_levels} 个抽象层级。")
        for lvl, comms in hierarchy.items():
            print(f"   ▶ Level {lvl}: 共划分出 {len(comms)} 个超级社区。")
        print("="*50)

        # 3. 自底向上 (Bottom-Up) 生成摘要并双写入库
        milvus_data_to_insert = {"community_ids": [], "levels": [], "summaries": []}
        community_summary_map: Dict[str, str] = {}
        level_pbar = tqdm(range(total_levels), desc="社区层级处理", unit="level")
        community_pbar = tqdm(total=total_communities, desc="社区摘要生成与入库", unit="community")

        for level in level_pbar:
            print(f"\n✍️ 正在处理 Level {level} 的摘要生成与入库...")
            communities_in_level = hierarchy[level]

            jobs = []
            for comm in communities_in_level:
                cid = comm["community_id"]

                # a. 生成摘要
                if level == 0:
                    node_ids = comm.get("nodes", [])
                    subgraph = G.subgraph(node_ids)

                    entities = [
                        {
                            "id": n,
                            "type": subgraph.nodes[n].get("type", "未知"),
                            "description": subgraph.nodes[n].get("description", ""),
                        }
                        for n in subgraph.nodes
                    ]
                    relations = []
                    for u, v, data in subgraph.edges(data=True):
                        relations.append(
                            {
                                "source": u,
                                "target": v,
                                "weight": data.get("weight", 1.0),
                                "relation_types": sorted(list(data.get("relations", set()))),
                                "descriptions": data.get("descriptions", [])[:3],
                            }
                        )
                    summary_input = {"entities": entities, "relations": relations}
                    child_ids = node_ids
                else:
                    child_ids = comm.get("sub_communities", [])
                    summary_input = [
                        {
                            "community_id": child_id,
                            "summary": community_summary_map.get(child_id, ""),
                        }
                        for child_id in child_ids
                    ]

                jobs.append(
                    {
                        "community_id": cid,
                        "child_ids": child_ids,
                        "summary_input": summary_input,
                    }
                )

            level_results = await self._summarize_level_parallel(
                level=level,
                jobs=jobs,
                community_pbar=community_pbar,
            )

            for result in level_results:
                cid = result["community_id"]
                summary_text = result["summary_text"]
                child_ids = result["child_ids"]
                community_summary_map[cid] = summary_text

                # b. 写入 Neo4j (建立垂直连线)
                self._save_to_neo4j(cid, level, child_ids, summary_text)

                # c. 收集 Milvus 数据
                milvus_data_to_insert["community_ids"].append(cid)
                milvus_data_to_insert["levels"].append(level)
                milvus_data_to_insert["summaries"].append(summary_text)

        level_pbar.close()
        community_pbar.close()

        # 4. 批量向量化并打入 Milvus
        if not milvus_data_to_insert["summaries"]:
            print("⚠️ 无社区摘要可入库，流程结束。")
            return

        print("\n🧮 正在将所有社区摘要向量化...")
        vectors = self.embedding_model.encode(milvus_data_to_insert["summaries"])

        if vectors and len(vectors[0]) != settings.milvus_vector_dim:
            raise ValueError(
                f"向量维度不匹配: collection dim={settings.milvus_vector_dim}, embedding dim={len(vectors[0])}"
            )
        
        print(f"💾 正在向 Milvus ({self.summary_collection_name}) 写入 {len(vectors)} 条全局摘要...")
        self.collection.insert([
            milvus_data_to_insert["community_ids"],
            milvus_data_to_insert["levels"],
            milvus_data_to_insert["summaries"],
            vectors
        ])
        self.collection.flush()
        
        print("✅ GraphRAG 社区划分、LLM 摘要生成、Neo4j/Milvus 双写任务完美收官！")