from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as NebulaConfig
from nebula3.Exception import IOErrorException

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import uuid

from app.config.settings import NEBULA_HOSTS, NEBULA_USER, NEBULA_PASSWORD
from app.utils.entity_extractor import EntityExtractor


class NebulaClient:
    """NebulaGraph图数据库客户端"""
    
    def __init__(self):
        # 配置Nebula连接
        nebula_configs = []
        for host_port in NEBULA_HOSTS:
            host, port = host_port.split(":")
            config = NebulaConfig()
            config.max_connection_pool_size = 10
            nebula_configs.append((host, int(port), config))
        
        # 创建连接池
        self.connection_pool = ConnectionPool()
        self.connection_pool.init(nebula_configs, NEBULA_USER, NEBULA_PASSWORD)
        
        # 实体提取器
        self.entity_extractor = EntityExtractor()
    
    def get_entity_tag_name(self, user_id: str) -> str:
        """根据用户ID生成实体标签名称"""
        return f"entity_{user_id}"
    
    def get_chunk_tag_name(self, user_id: str) -> str:
        """根据用户ID生成文本块标签名称"""
        return f"chunk_{user_id}"
    
    def get_has_chunk_edge_name(self, user_id: str) -> str:
        """根据用户ID生成has_chunk边类型名称"""
        return f"has_chunk_{user_id}"
    
    def get_related_to_edge_name(self, user_id: str) -> str:
        """根据用户ID生成related_to边类型名称"""
        return f"related_to_{user_id}"
    
    def create_user_schema(self, user_id: str) -> bool:
        """为用户创建必要的标签和边类型"""
        with self.connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
            # 创建实体标签
            entity_tag = self.get_entity_tag_name(user_id)
            create_entity_tag = f"""
            CREATE TAG IF NOT EXISTS {entity_tag}(
                name string,
                type string,
                properties string
            )
            """
            result = session.execute(create_entity_tag)
            if not result.is_succeeded():
                logging.error(f"创建实体标签失败: {result.error_msg()}")
                return False
            
            # 创建文本块标签
            chunk_tag = self.get_chunk_tag_name(user_id)
            create_chunk_tag = f"""
            CREATE TAG IF NOT EXISTS {chunk_tag}(
                content string,
                doc_id string,
                metadata string
            )
            """
            result = session.execute(create_chunk_tag)
            if not result.is_succeeded():
                logging.error(f"创建文本块标签失败: {result.error_msg()}")
                return False
            
            # 创建边类型：实体与文本块的关系
            has_chunk_edge = self.get_has_chunk_edge_name(user_id)
            create_has_chunk_edge = f"""
            CREATE EDGE IF NOT EXISTS {has_chunk_edge}(
                weight double
            )
            """
            result = session.execute(create_has_chunk_edge)
            if not result.is_succeeded():
                logging.error(f"创建{has_chunk_edge}边类型失败: {result.error_msg()}")
                return False
            
            # 创建边类型：实体与实体的关系
            related_to_edge = self.get_related_to_edge_name(user_id)
            create_related_to_edge = f"""
            CREATE EDGE IF NOT EXISTS {related_to_edge}(
                relation string,
                weight double
            )
            """
            result = session.execute(create_related_to_edge)
            if not result.is_succeeded():
                logging.error(f"创建{related_to_edge}边类型失败: {result.error_msg()}")
                return False
            
            # 创建索引
            create_indexes = [
                f"CREATE TAG INDEX IF NOT EXISTS {entity_tag}_name_idx ON {entity_tag}(name(64))",
                f"CREATE TAG INDEX IF NOT EXISTS {chunk_tag}_doc_idx ON {chunk_tag}(doc_id(32))",
                f"CREATE EDGE INDEX IF NOT EXISTS {related_to_edge}_idx ON {related_to_edge}(relation(32))"
            ]
            
            for index_stmt in create_indexes:
                result = session.execute(index_stmt)
                if not result.is_succeeded():
                    logging.error(f"创建索引失败: {result.error_msg()}")
                    return False
            
            # 重建索引
            rebuild_indexes = [
                f"REBUILD TAG INDEX {entity_tag}_name_idx",
                f"REBUILD TAG INDEX {chunk_tag}_doc_idx",
                f"REBUILD EDGE INDEX {related_to_edge}_idx"
            ]
            
            for rebuild_stmt in rebuild_indexes:
                result = session.execute(rebuild_stmt)
                if not result.is_succeeded():
                    logging.error(f"重建索引失败: {result.error_msg()}")
                    return False
            
            return True
    
    def add_entity(self, user_id: str, entity_id: str, entity_name: str, 
                 entity_type: str, properties: Dict[str, Any] = None) -> bool:
        """添加实体节点"""
        entity_tag = self.get_entity_tag_name(user_id)
        
        with self.connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
            # 序列化属性
            props_json = "{}" if properties is None else json.dumps(properties)
            
            # 插入实体
            insert_stmt = f"""
            INSERT VERTEX {entity_tag}(name, type, properties) 
            VALUES '{entity_id}':('{entity_name}', '{entity_type}', '{props_json}')
            """
            
            result = session.execute(insert_stmt)
            return result.is_succeeded()
    
    def add_chunk(self, user_id: str, chunk_id: str, content: str, 
                doc_id: str, metadata: Dict[str, Any] = None) -> bool:
        """添加文本块节点"""
        chunk_tag = self.get_chunk_tag_name(user_id)
        
        with self.connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
            # 序列化元数据
            metadata_json = "{}" if metadata is None else json.dumps(metadata)
            
            # 清理内容中的引号和换行符
            content = content.replace("'", "\\'").replace("\n", " ")
            
            # 插入文本块
            insert_stmt = f"""
            INSERT VERTEX {chunk_tag}(content, doc_id, metadata) 
            VALUES '{chunk_id}':('{content}', '{doc_id}', '{metadata_json}')
            """
            
            result = session.execute(insert_stmt)
            return result.is_succeeded()
    
    def add_relationship(self, user_id: str, source_id: str, target_id: str, 
                       relation_type: str, relation: str = "", weight: float = 1.0) -> bool:
        """添加关系边"""
        with self.connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
            # 根据关系类型选择边类型
            if relation_type == "has_chunk":
                edge_name = self.get_has_chunk_edge_name(user_id)
                insert_stmt = f"""
                INSERT EDGE {edge_name}(weight) 
                VALUES '{source_id}'->'{target_id}':(${weight})
                """
            elif relation_type == "related_to":
                edge_name = self.get_related_to_edge_name(user_id)
                insert_stmt = f"""
                INSERT EDGE {edge_name}(relation, weight) 
                VALUES '{source_id}'->'${target_id}':('{relation}', ${weight})
                """
            else:
                return False
            
            result = session.execute(insert_stmt)
            return result.is_succeeded()
    
    def search_entities(self, user_id: str, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索实体节点"""
        entity_tag = self.get_entity_tag_name(user_id)
        
        with self.connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
            # 查询实体
            query_stmt = f"""
            MATCH (e:{entity_tag}) 
            WHERE e.{entity_tag}.name CONTAINS '{entity_name}' 
            RETURN e.{entity_tag}.name AS name, e.{entity_tag}.type AS type, 
                   e.{entity_tag}.properties AS properties, id(e) AS id 
            LIMIT {limit}
            """
            
            result = session.execute(query_stmt)
            if not result.is_succeeded():
                return []
            
            entities = []
            for record in result.records():
                entity = {
                    "id": record.values()[3].as_string(),
                    "name": record.values()[0].as_string(),
                    "type": record.values()[1].as_string(),
                    "properties": json.loads(record.values()[2].as_string())
                }
                entities.append(entity)
            
            return entities
    
    def search_related_chunks(self, user_id: str, entity_ids: List[str], 
                            limit: int = 5) -> List[Dict[str, Any]]:
        """搜索与实体相关的文本块"""
        entity_tag = self.get_entity_tag_name(user_id)
        chunk_tag = self.get_chunk_tag_name(user_id)
        has_chunk_edge = self.get_has_chunk_edge_name(user_id)
        
        with self.connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
            # 构建ID列表
            id_list = ", ".join([f"'{id}'" for id in entity_ids])
            
            # 查询相关文本块
            query_stmt = f"""
            MATCH (e:{entity_tag})-[r:{has_chunk_edge}]->(c:{chunk_tag}) 
            WHERE id(e) IN [{id_list}] 
            RETURN c.{chunk_tag}.content AS content, c.{chunk_tag}.doc_id AS doc_id, 
                   c.{chunk_tag}.metadata AS metadata, id(c) AS id 
            LIMIT {limit}
            """
            
            result = session.execute(query_stmt)
            if not result.is_succeeded():
                return []
            
            chunks = []
            for record in result.records():
                chunk = {
                    "id": record.values()[3].as_string(),
                    "content": record.values()[0].as_string(),
                    "doc_id": record.values()[1].as_string(),
                    "metadata": json.loads(record.values()[2].as_string())
                }
                chunks.append(chunk)
            
            return chunks
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体，使用EntityExtractor"""
        return self.entity_extractor.extract_entities(text)
    
    def build_knowledge_graph_from_text(self, user_id: str, text: str, doc_id: str) -> Tuple[List[str], List[str]]:
        """从文本构建知识图谱，包括实体和关系"""
        # 创建用户schema
        if not self.create_user_schema(user_id):
            return [], []
            
        # 提取实体
        entities = self.entity_extractor.extract_entities(text)
        
        # 提取三元组
        triples = self.entity_extractor.extract_triples(text)
        
        # 为文本块生成ID
        chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
        
        # 添加文本块到图数据库
        self.add_chunk(
            user_id=user_id,
            chunk_id=chunk_id,
            content=text,
            doc_id=doc_id
        )
        
        # 添加实体及与文本块的关系
        entity_ids = []
        for entity in entities:
            # 添加实体
            entity_id = entity["id"]
            entity_ids.append(entity_id)
            
            self.add_entity(
                user_id=user_id,
                entity_id=entity_id,
                entity_name=entity["name"],
                entity_type=entity["type"],
                properties=entity.get("properties", {})
            )
            
            # 添加实体与文本块的关系
            self.add_relationship(
                user_id=user_id,
                source_id=entity_id,
                target_id=chunk_id,
                relation_type="has_chunk",
                weight=1.0
            )
        
        # 添加实体之间的关系（基于三元组）
        relation_ids = []
        for triple in triples:
            # 查找或创建主体实体
            subject_entities = [e for e in entities if e["name"].lower() == triple["subject"].lower()]
            if not subject_entities:
                subject_id = f"entity_{uuid.uuid4().hex[:8]}"
                subject_entity = {
                    "id": subject_id,
                    "name": triple["subject"],
                    "type": "concept"
                }
                entities.append(subject_entity)
                entity_ids.append(subject_id)
                
                self.add_entity(
                    user_id=user_id,
                    entity_id=subject_id,
                    entity_name=triple["subject"],
                    entity_type="concept"
                )
                
                self.add_relationship(
                    user_id=user_id,
                    source_id=subject_id,
                    target_id=chunk_id,
                    relation_type="has_chunk",
                    weight=1.0
                )
            else:
                subject_id = subject_entities[0]["id"]
            
            # 查找或创建客体实体
            object_entities = [e for e in entities if e["name"].lower() == triple["object"].lower()]
            if not object_entities:
                object_id = f"entity_{uuid.uuid4().hex[:8]}"
                object_entity = {
                    "id": object_id,
                    "name": triple["object"],
                    "type": "concept"
                }
                entities.append(object_entity)
                entity_ids.append(object_id)
                
                self.add_entity(
                    user_id=user_id,
                    entity_id=object_id,
                    entity_name=triple["object"],
                    entity_type="concept"
                )
                
                self.add_relationship(
                    user_id=user_id,
                    source_id=object_id,
                    target_id=chunk_id,
                    relation_type="has_chunk",
                    weight=1.0
                )
            else:
                object_id = object_entities[0]["id"]
            
            # 添加关系
            relation_id = f"relation_{uuid.uuid4().hex[:8]}"
            relation_ids.append(relation_id)
            
            self.add_relationship(
                user_id=user_id,
                source_id=subject_id,
                target_id=object_id,
                relation_type="related_to",
                relation=triple["relation"],
                weight=0.8
            )
        
        return entity_ids, relation_ids
    
    def close(self):
        """关闭连接池"""
        self.connection_pool.close() 