from pymilvus import connections, Collection, utility
from typing import Optional, List, Dict, Any
from app.config.settings import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_PREFIX, MILVUS_DIMENSION

@DeprecationWarning("MilvusClient is deprecated.")
class MilvusClient:
    """Milvus向量数据库客户端"""
    
    def __init__(self):
        # 连接到Milvus服务器
        self.connection = connections.connect(
            alias="default", 
            host=MILVUS_HOST, 
            port=MILVUS_PORT
        )
        self.dimension = MILVUS_DIMENSION
        self.collection_prefix = MILVUS_COLLECTION_PREFIX
    
    def get_collection_name(self, user_id: str) -> str:
        """根据用户ID获取对应的collection名称"""
        return f"{self.collection_prefix}{user_id}"
    
    def collection_exists(self, user_id: str) -> bool:
        """检查用户的collection是否存在"""
        collection_name = self.get_collection_name(user_id)
        return utility.has_collection(collection_name)
    
    def create_user_collection(self, user_id: str) -> None:
        """为用户创建一个新的collection"""
        from pymilvus import FieldSchema, CollectionSchema, DataType
        
        collection_name = self.get_collection_name(user_id)
        
        # 如果collection已存在，则直接返回
        if self.collection_exists(user_id):
            return
        
        # 定义collection的schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(fields)
        
        # 创建collection
        collection = Collection(name=collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    
    def get_collection(self, user_id: str) -> Optional[Collection]:
        """获取用户的collection"""
        if not self.collection_exists(user_id):
            self.create_user_collection(user_id)
        
        collection_name = self.get_collection_name(user_id)
        collection = Collection(name=collection_name)
        collection.load()
        return collection
    
    def insert_documents(self, user_id: str, doc_ids: List[str], contents: List[str], 
                        embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[str]:
        """向用户的collection中插入文档"""
        collection = self.get_collection(user_id)
        
        # 准备插入数据
        entities = []
        ids = []
        
        for i in range(len(contents)):
            # 为每个分块生成唯一ID
            chunk_id = f"{doc_ids[i]}_{i}"
            ids.append(chunk_id)
            
            entities.append({
                "id": chunk_id,
                "doc_id": doc_ids[i],
                "user_id": user_id,
                "content": contents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i]
            })
        
        # 执行插入操作
        collection.insert(entities)
        collection.flush()
        
        return ids
    
    def search_similar_documents(self, user_id: str, query_embedding: List[float], 
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """在用户的collection中搜索与查询向量最相似的文档"""
        collection = self.get_collection(user_id)
        
        # 设置搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        # 执行搜索操作
        results = collection.search(
            data=[query_embedding], 
            anns_field="embedding", 
            param=search_params,
            limit=top_k,
            output_fields=["id", "doc_id", "content", "metadata"]
        )
        
        # 处理搜索结果
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "id": hit.entity.get("id"),
                    "doc_id": hit.entity.get("doc_id"),
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score
                })
        
        return search_results
    
    def delete_document(self, user_id: str, doc_id: str) -> None:
        """从用户的collection中删除指定文档的所有块"""
        collection = self.get_collection(user_id)
        
        # 执行删除操作
        collection.delete(f"doc_id == '{doc_id}'")
    
    def close(self) -> None:
        """关闭Milvus连接"""
        connections.disconnect("default") 