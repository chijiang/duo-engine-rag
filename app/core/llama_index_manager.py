import os
from typing import List, Dict, Any, Optional, Tuple
import uuid
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core import Settings
import logging
from datetime import datetime

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.lmstudio import LMStudio
from typing import Any, List, Dict, Optional, Tuple


from zhipuai import ZhipuAI

from app.config.settings import (
    DEEPSEEK_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MILVUS_DIMENSION,
    MILVUS_HOST,
    MILVUS_PORT,
    TEXT_SPLITTER,
    DEFAULT_TOP_K,
    SIMILARITY_TOP_K,
    GRAPH_TOP_K,
    NEBULA_USER,
    NEBULA_PASSWORD,
    NEBULA_HOSTS,
    ZHIPU_API_KEY,
)
from app.core.retrievers import PropertyGraphIndexImpl, NebulaPropertyGraphStoreImpl


class ZhipuAIEmbedding(BaseEmbedding):
    api_key: Optional[str] = None
    client: Optional[ZhipuAI] = None

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.client = ZhipuAI(api_key=self.api_key)

    def _get_embedding(self, text):
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=MILVUS_DIMENSION,
        )
        return response.data[0].embedding

    def _get_text_embedding(self, text):
        return self._get_embedding(text)

    def _get_query_embedding(self, query):
        return self._get_embedding(query)

    async def _aget_text_embedding(self, texts):
        return self._get_embedding(texts)

    async def _aget_query_embedding(self, queries):
        return self._get_embedding(queries)


class LlamaIndexManager:
    """LlamaIndex管理器，处理文档的索引和检索"""

    def __init__(self):
        # NebulaGraph自动化集成
        self.graph_store_dict = {}  # 按用户隔离
        self.vec_store_dict = {}
        self.user_index_dict: Dict[str, PropertyGraphIndexImpl] = {}

        # 配置LLM和Embedding模型
        self.llm = LMStudio(
            model_name="qwen2.5-7b-instruct-1m",
            base_url="http://127.0.0.1:1234/v1",
            context_window=128000,
            temperature=0.,
            timeout=1000,
        )

        self.conclusion_llm = LMStudio(
            model_name="qwen3-30b-a3b",
            base_url="http://127.0.0.1:1234/v1",
            context_window=128000,
            temperature=0.1,
            timeout=1000,
        )
        # self.llm = DeepSeek(model=LLM_MODEL, api_key=DEEPSEEK_API_KEY)
        self.embed_model = ZhipuAIEmbedding(api_key=ZHIPU_API_KEY)
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # 配置文本分块器
        if TEXT_SPLITTER == "sentence":
            self.text_splitter = SentenceSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n"
            )
        else:
            # 如需其他分块器可在此添加
            self.text_splitter = SentenceSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n"
            )

    def _get_user_index(self, user_id: str):
        if user_id not in self.user_index_dict:
            try:
                self.user_index_dict[user_id] = PropertyGraphIndexImpl.from_existing(
                    property_graph_store=self._get_graph_store(user_id),
                    vector_store=self._get_vec_store(user_id),
                )
            except Exception as e:
                logging.error(f"加载索引失败: {str(e)}")
                return None
        return self.user_index_dict[user_id]

    def _get_graph_store(self, user_id: str) -> NebulaPropertyGraphStoreImpl:
        # 返回一个共享的 NebulaPropertyGraphStoreImpl 实例，指向 "llamaindex_nebula_property_graph" space
        # 隔离将通过节点/关系上的 "user_id" 属性实现
        if user_id not in self.graph_store_dict:
            logging.info(
                "Creating a new shared NebulaPropertyGraphStoreImpl for 'llamaindex_nebula_property_graph' space."
            )
            self.graph_store_dict[user_id] = NebulaPropertyGraphStoreImpl(
                space="llamaindex_nebula_property_graph",
                username=NEBULA_USER,
                password=NEBULA_PASSWORD,
                url=NEBULA_HOSTS,
                props_schema=(
                    "`file_path` STRING, "
                    "`file_name` STRING, "
                    "`file_type` STRING, "
                    "`file_size` INT, "
                    "`creation_date` STRING, "
                    "`last_modified_date` STRING, "
                    "`_node_content` STRING, "
                    "`_node_type` STRING, "
                    "`document_id` STRING, "
                    "`ref_doc_id` STRING, "
                    "`triplet_source_id` STRING, "
                    "`user_id` STRING, "
                    "`created_at` STRING, "
                    "`doc_id` STRING, "
                    "`updated_at` STRING"
                ),
                # NebulaPropertyGraphStoreImpl 会使用默认的 tag/edge 名称
                # e.g., entity_label="entity", relation_label="relationship", chunk_label="chunk"
            )
        return self.graph_store_dict[user_id]

    def _get_vec_store(self, user_id: str):
        if user_id not in self.vec_store_dict:
            self.vec_store_dict[user_id] = MilvusVectorStore(
                uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
                collection_name="llamaindex_milvus_vector_store",
                dim=MILVUS_DIMENSION,
                embedding_field="embedding",
            )
        return self.vec_store_dict[user_id]

    async def process_document(
        self, file_path: str, user_id: str, doc_metadata: Dict[str, Any]
    ) -> str:
        """处理文档，将其分块、计算嵌入、更新数据库"""
        try:
            doc_id = str(uuid.uuid4())
            # 确保 doc_metadata 包含 user_id，它将被传播到每个 Node
            base_metadata = doc_metadata.copy()  # 创建副本以避免修改原始字典
            base_metadata["document_id"] = doc_id
            base_metadata["created_at"] = datetime.now().isoformat()

            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

            # 2. 用 SentenceSplitter 拆分成 chunk
            documents = []
            all_nodes = []
            for doc_obj in docs:
                doc_obj.metadata = base_metadata.copy()
                nodes = self.text_splitter.get_nodes_from_documents([doc_obj])
                all_nodes.extend(nodes)
                for node in nodes:
                    # 每个 chunk 变成一个新的 Document
                    chunk_doc = Document(text=node.text, metadata=node.metadata.copy())
                    documents.append(chunk_doc)

            user_index = self._get_user_index(user_id)
            if user_index is None:
                # 2. 构建 NebulaPropertyGraphStoreImpl 和 SimpleVectorStore
                graph_store = self._get_graph_store(user_id)
                vec_store = self._get_vec_store(user_id)

                # 3. 用 from_documents 构建索引
                storage_context = StorageContext.from_defaults(
                    vector_store=vec_store, graph_store=graph_store
                )
                # vector_index = VectorStoreIndex.from_documents(
                #     documents,
                #     storage_context=storage_context,
                #     show_progress=True,
                # )

                graph_index = PropertyGraphIndexImpl.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True,
                    embed_kg_nodes=False,  # 根据您的设置
                )

                graph_index.storage_context.persist(
                    f"./data/llama_index_storage_{user_id}/"
                )

                self.user_index_dict[user_id] = graph_index
            else:
                user_index.insert_nodes(all_nodes)
            # vec_store.persist(f"./data/nebula_vec_store_{user_id}.json")

            # Milvus 部分 (使用 documents)
            # nodes_for_milvus = self.text_splitter.get_nodes_from_documents(documents) # 重复分块？应该用现有的 nodes/documents
            # contents = [doc.text for doc in docs]
            # doc_ids_for_milvus = [doc.metadata["doc_id"] for doc in docs]
            # metadatas_for_milvus = [doc.metadata for doc in docs]

            # embeddings = self.embed_model.get_text_embedding_batch(contents)

            # self.milvus_client.insert_documents(
            #     user_id=user_id, # Milvus 集合名会用 user_id
            #     doc_ids=doc_ids_for_milvus,
            #     contents=contents,
            #     embeddings=embeddings,
            #     metadatas=metadatas_for_milvus,
            # )
            return doc_id
        except Exception as e:
            logging.error(f"处理文档时出错：{str(e)}")
            return None

    async def query(
        self,
        user_id: str,
        query_text: str,
        similarity_top_k: int = None,
        graph_top_k: int = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """进行查询，组合向量检索和图检索结果"""
        # 设置默认检索数量
        similarity_top_k = similarity_top_k or SIMILARITY_TOP_K
        graph_top_k = graph_top_k or GRAPH_TOP_K

        search_index = self._get_user_index(user_id)
        if search_index is None:
            return "没有找到索引", []

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="user_id", operator=FilterOperator.EQ, value=user_id
                ),
            ]
        )

        retriever = search_index.as_retriever(
            similarity_top_k=similarity_top_k,
            graph_top_k=graph_top_k,
            filters=filters,
        )

        all_results = await retriever.aretrieve(
            query_text, properties={"user_id": user_id}
        )

        # 构建上下文
        context_text = "\n\n".join([result.get_content() for result in all_results])

        # 生成回答
        response = self.conclusion_llm.complete(
            f'''\
## 请仅基于以下`相关信息`回答问题，如未提供任何信息，则提示用户缺少相关信息，无法回答问题。
## 相关信息：
```{context_text}```

## 问题: 
{query_text}

## 回答:/no_think''',
            timeout=1000
        )

        # 处理源文档信息
        sources = []
        for result in all_results:
            source = {
                "doc_id": result.metadata["document_id"],
                "content": result.get_content()[:200] + "...",  # 截断长文本
                "score": result.get_score(),
            }
            if "metadata" in result and result.metadata:
                source["file_name"] = result.metadata.get("file_name")
                source["file_type"] = result.metadata.get("file_type")

            sources.append(source)

        return response.text, sources

    def _vector_search(
        self, user_id: str, query_text: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """进行向量相似度检索"""
        try:
            # 计算查询的嵌入向量
            query_embedding = self.embed_model.get_text_embedding(query_text)

            # 使用Milvus客户端进行搜索
            results = self.milvus_client.search_similar_documents(
                user_id=user_id, query_embedding=query_embedding, top_k=top_k
            )

            # 标记来源类型
            for result in results:
                result["source_type"] = "vector"

            return results

        except Exception as e:
            logging.error(f"向量检索出错：{str(e)}")
            return []

    def _graph_search(
        self, user_id: str, query_text: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """用LlamaIndex自动化接口进行图谱检索，并根据user_id过滤"""
        try:
            graph_store = self._get_graph_store(user_id)  # 获取共享的 graph_store
            vec_store = self._get_vec_store(user_id)  # PGI 使用的 SimpleVectorStore

            # PropertyGraphIndex.from_existing 只是重新加载索引结构，
            # 它不会重新构建或修改存储在 graph_store 中的数据。
            index = PropertyGraphIndexImpl.from_existing(
                property_graph_store=graph_store,
                vector_store=vec_store,  # 如果 PGI 用了向量组件
                # 如果在构建时指定了 kg_extractors 或其他参数，这里也需要匹配
            )

            # 获取检索器
            # retriever = index.as_retriever(include_text=True)
            # 我们需要传递 prop_filters 给底层的 get_rel_map
            # PropertyGraphRetriever 的构造函数或者 as_retriever 可能没有直接暴露这个

            # 方案: 自定义 Retriever 或直接调用更底层的方法
            # 让我们检查 PropertyGraphRetriever 的 _retrieve 方法和它如何调用 get_rel_map

            # PropertyGraphRetriever 的构造函数中有一个 `graph_traversal_depth` 和 `max_knowledge_sequence`
            # 但没有直接的 prop_filters。

            # 尝试直接使用 PropertyGraphIndex 中的方法，或者更深入地看 Retriever
            # 如果标准 retriever 不支持，一个更简单（但不完全通过 LlamaIndex Retriever API）的方式：
            # 1. 用 LLM 或其他方式从 query_text 中提取关键词/实体。
            # 2. 使用 graph_store.get() 或类似的（如果支持 Cypher/属性过滤）来查找这些实体（属于该 user_id）。
            # 3. 然后使用 graph_store.get_rel_map() 传入这些实体和 prop_filters。

            # LlamaIndex (v0.10.x) PropertyGraphRetriever 使用 _graph_store.get_rel_map
            # 查看 PropertyGraphIndex 的 as_retriever 是否可以将参数传递下去。
            # class PropertyGraphRetriever(BaseRetriever):
            #    def __init__(
            #        self,
            #        index: "PropertyGraphIndex",
            #        ...,
            #        graph_traversal_depth: int = 2,
            #        prop_filter_fn: Optional[Callable[[Dict], Dict]] = None, <--- 这个看起来很有希望！
            #        **kwargs: Any,
            #    ) -> None:

            # prop_filter_fn: A function that takes a dictionary of properties
            #                 and returns a dictionary of properties to filter by.
            # 这不是我们想要的。我们需要的是在 Cypher WHERE 子句中添加过滤。

            # 回到 NebulaPropertyGraphStoreImpl 的 get_rel_map(self, entities, depth, limit, prop_filters)
            # prop_filters (Optional[PropertyFilters]): Property filters for nodes/relationships.
            # class PropertyFilters:
            #    properties: Optional[List[List[Tuple[str, Any]]]] = Field(
            #        default=None,
            #        description="List of properties to filter by. Applied with OR, and within Applied with AND.",
            #    )
            #    labels: Optional[List[str]] = Field(
            #        default=None, description="List of labels to filter by. Applied with OR."
            #    )

            # 这意味着我们不能简单地通过 retriever 来做。
            # 我们需要一种方式让 get_rel_map 被调用时包含 user_id 过滤。

            # **一个更实际的方案，虽然不是最理想的 LlamaIndex 方式：**
            # 由于 LlamaIndex 的 PropertyGraphRetriever 可能不直接暴露我们需要的属性过滤机制到 Nebula 的 get_rel_map，
            # 我们可以在检索后对结果进行过滤。这不是最高效的，因为 Nebula 会返回更多数据。

            retriever = index.as_retriever(include_text=True)  # 标准检索器
            # 这个nodes列表是 llama_index.core.schema.NodeWithScore 的列表
            retrieved_nodes_with_score = retriever.retrieve(query_text)

            results = []
            for node_with_score in retrieved_nodes_with_score:
                node_obj = (
                    node_with_score.node
                )  # 这是 TextNode 或类似的 LlamaIndex Node
                # 我们在 process_document 中确保了 user_id 在 metadata 中
                if node_obj.metadata.get("user_id") == user_id:
                    results.append(
                        {
                            "doc_id": node_obj.metadata.get("doc_id", ""),
                            "content": node_obj.get_content(),  # 使用 get_content()
                            "metadata": node_obj.metadata,
                            "score": (
                                node_with_score.score
                                if node_with_score.score is not None
                                else 0.0
                            ),
                            "source_type": "graph",
                        }
                    )
                if len(results) >= top_k:  # 尽早停止
                    break

            # 如果检索到的节点太少，这里的 top_k 可能无法满足
            # 这种后过滤方式可能导致返回的节点数少于 top_k
            # 为了缓解，可以考虑在 retriever.retrieve 时请求更多节点，例如 top_k * 2 或更高
            # retrieved_nodes_with_score = retriever.retrieve(query_text, top_k_multiplier=2) # 假设有这样的参数
            # LlamaIndex Retriever 通常有一个 similarity_top_k 或类似参数在构建时设置，或在 retrieve 时。
            # PropertyGraphRetriever 的 retrieve 方法没有直接的 top_k 参数，它依赖于构建时的设置。

            # logging.debug(f"Graph search for user {user_id} with query '{query_text}' found {len(results)} results after filtering.")
            return results[:top_k]  # 确保最终结果不超过 top_k

        except Exception as e:
            logging.error(f"图检索出错 for user {user_id}: {str(e)}")
            return []

    def _merge_search_results(
        self, vector_results: List[Dict[str, Any]], graph_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """合并向量检索和图检索结果，去重并按相关度排序"""
        # 使用doc_id+content组合作为去重依据
        unique_results = {}

        # 处理向量检索结果
        for result in vector_results:
            key = f"{result['doc_id']}_{result['content'][:50]}"  # 使用文档ID和内容前50个字符作为key
            if key not in unique_results or result.get("score", 0) > unique_results[
                key
            ].get("score", 0):
                unique_results[key] = result

        # 处理图检索结果，进行相似度重新排序
        for result in graph_results:
            key = f"{result['doc_id']}_{result['content'][:50]}"  # 使用文档ID和内容前50个字符作为key

            # 如果结果已存在，我们给予图检索一个加权因子
            if key in unique_results:
                # 如果向量结果分数高于当前图结果，保留但修改来源类型
                if unique_results[key].get("score", 0) > result.get("score", 0):
                    unique_results[key]["source_type"] = "hybrid"
            else:
                # 否则直接添加图检索结果
                unique_results[key] = result

        # 将结果转换回列表并按相关度排序
        merged_results = list(unique_results.values())
        merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return merged_results[:DEFAULT_TOP_K]  # 限制返回结果数量

    def close(self):
        """关闭连接"""
        self.milvus_client.close()
