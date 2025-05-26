import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import aiosqlite
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as NebulaConfig
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from app.core.llama_index_manager import LlamaIndexManager
from app.models.schemas import DocumentMetadata
from app.config.settings import NEBULA_HOSTS, NEBULA_USER, NEBULA_PASSWORD


class DocumentService:
    """文档服务，处理文档上传和处理"""
    
    DB_PATH = "metadata.db"
    NEBULA_SPACE_NAME = "llamaindex_nebula_property_graph"

    def __init__(self):
        self.llama_index_manager = LlamaIndexManager()
        self.upload_dir = "uploads"
        
        # 确保上传目录存在
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def _create_nebula_connection_pool(self):
        """创建NebulaGraph连接池"""
        nebula_configs = []
        # 处理NEBULA_HOSTS可能是字符串或列表的情况
        hosts = NEBULA_HOSTS if isinstance(NEBULA_HOSTS, list) else [NEBULA_HOSTS]
        
        for host_port in hosts:
            host, port = host_port.split(":")
            config = NebulaConfig()
            config.max_connection_pool_size = 10
            nebula_configs.append((host, int(port), config))
        
        connection_pool = ConnectionPool()
        connection_pool.init(nebula_configs, NEBULA_USER, NEBULA_PASSWORD)
        return connection_pool
    
    def _create_nebula_space(self) -> bool:
        """创建NebulaGraph空间"""
        connection_pool = None
        try:
            connection_pool = self._create_nebula_connection_pool()
            
            with connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
                # 创建空间（如果不存在）
                create_space_stmt = f"""
                CREATE SPACE IF NOT EXISTS {self.NEBULA_SPACE_NAME} (
                    partition_num = 10,
                    replica_factor = 1,
                    vid_type = FIXED_STRING(64)
                )
                """
                
                result = session.execute(create_space_stmt)
                if not result.is_succeeded():
                    logging.error(f"创建NebulaGraph空间失败: {result.error_msg()}")
                    return False
                
                logging.info(f"NebulaGraph空间 '{self.NEBULA_SPACE_NAME}' 创建成功或已存在")
                
                # 使用空间
                use_space_stmt = f"USE {self.NEBULA_SPACE_NAME}"
                result = session.execute(use_space_stmt)
                if not result.is_succeeded():
                    logging.error(f"使用NebulaGraph空间失败: {result.error_msg()}")
                    return False
                
                # 创建基本的标签和边类型（LlamaIndex需要的）
                # 这些是NebulaPropertyGraphStore使用的默认标签和边类型
                create_statements = [
                    """
                    CREATE TAG IF NOT EXISTS entity(
                        `file_path` STRING,
                        `file_name` STRING,
                        `file_type` STRING,
                        `file_size` INT,
                        `creation_date` STRING,
                        `last_modified_date` STRING,
                        `_node_content` STRING,
                        `_node_type` STRING,
                        `document_id` STRING,
                        `ref_doc_id` STRING,
                        `triplet_source_id` STRING,
                        `user_id` STRING,
                        `created_at` STRING,
                        `doc_id` STRING,
                        `updated_at` STRING
                    )
                    """,
                    """
                    CREATE TAG IF NOT EXISTS chunk(
                        `file_path` STRING,
                        `file_name` STRING,
                        `file_type` STRING,
                        `file_size` INT,
                        `creation_date` STRING,
                        `last_modified_date` STRING,
                        `_node_content` STRING,
                        `_node_type` STRING,
                        `document_id` STRING,
                        `ref_doc_id` STRING,
                        `triplet_source_id` STRING,
                        `user_id` STRING,
                        `created_at` STRING,
                        `doc_id` STRING,
                        `updated_at` STRING
                    )
                    """,
                    """
                    CREATE EDGE IF NOT EXISTS relationship(
                        `file_path` STRING,
                        `file_name` STRING,
                        `file_type` STRING,
                        `file_size` INT,
                        `creation_date` STRING,
                        `last_modified_date` STRING,
                        `_node_content` STRING,
                        `_node_type` STRING,
                        `document_id` STRING,
                        `ref_doc_id` STRING,
                        `triplet_source_id` STRING,
                        `user_id` STRING,
                        `created_at` STRING,
                        `doc_id` STRING,
                        `updated_at` STRING
                    )
                    """
                ]
                
                for stmt in create_statements:
                    result = session.execute(stmt)
                    if not result.is_succeeded():
                        logging.warning(f"创建标签/边类型时出现警告: {result.error_msg()}")
                        # 不返回False，因为这些可能已经存在
                
                logging.info("NebulaGraph基本schema创建完成")
                return True
                
        except Exception as e:
            logging.error(f"创建NebulaGraph空间时出错: {str(e)}")
            return False
        finally:
            if connection_pool:
                connection_pool.close()

    async def initialize_db(self):
        """初始化数据库并创建表（如果不存在）"""
        # 1. 初始化SQLite数据库
        async with aiosqlite.connect(self.DB_PATH) as db:
            await db.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                doc_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                doc_type TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
            await db.commit()
            logging.info("数据库表 'document_metadata' 初始化完成。")
        
        # 2. 初始化NebulaGraph空间
        try:
            if self._create_nebula_space():
                logging.info("NebulaGraph空间初始化完成。")
            else:
                logging.error("NebulaGraph空间初始化失败，但系统将继续运行。")
        except Exception as e:
            logging.error(f"NebulaGraph初始化过程中出现异常: {str(e)}")
            logging.warning("NebulaGraph初始化失败，但系统将继续运行。请检查NebulaGraph连接配置。")

    def save_uploaded_file(self, file, user_id: str) -> str:
        """保存上传的文件"""
        # 创建用户目录
        user_dir = os.path.join(self.upload_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # 生成文件名
        filename = file.filename
        file_ext = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(user_dir, unique_filename)
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        return file_path
    
    async def process_document(self, file_path: str, user_id: str, 
                        doc_name: Optional[str] = None,
                        doc_type: Optional[str] = None,) -> Optional[str]:
        """处理文档并将元数据保存到SQLite"""
        try:
            # 如果没有提供文档名，则使用原始文件名
            if not doc_name:
                doc_name = os.path.basename(file_path)
            
            # 如果没有提供文档类型，则根据文件扩展名推断
            if not doc_type:
                doc_type = os.path.splitext(file_path)[1].lstrip('.')
            
            # 准备LlamaIndex的元数据（可能与SQLite存储的略有不同或子集）
            llama_index_specific_metadata = {
                "file_name": doc_name, # Original filename for LlamaIndex
                "file_type": doc_type,
                "user_id": user_id, 
                # Add any other metadata LlamaIndex specifically needs for processing
            }

            # Pass necessary metadata to LlamaIndex, potentially including our generated_doc_id
            # if LlamaIndex can use it (e.g., as a top-level document ID).
            # The metadata dict for LlamaIndex might be different from what we store in SQLite.
            generated_doc_id = await self.llama_index_manager.process_document(
                file_path=file_path,
                user_id=user_id, # For LlamaIndex's internal user separation if any
                doc_metadata={**llama_index_specific_metadata} # Example
            )

            # 元数据存入SQLite
            current_time_utc = datetime.now(timezone.utc).isoformat()
            
            async with aiosqlite.connect(self.DB_PATH) as db:
                await db.execute("""
                INSERT INTO document_metadata (doc_id, user_id, doc_name, doc_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (generated_doc_id, user_id, doc_name, doc_type, current_time_utc, current_time_utc))
                await db.commit()
            
            logging.info(f"文档元数据已保存到SQLite: doc_id={generated_doc_id}, user_id={user_id}")
            return generated_doc_id # Return our generated doc_id

        except Exception as e:
            logging.error(f"处理文档或保存元数据出错: {str(e)}")
            # Consider more specific error handling or re-raising
            return None
    
    def delete_temp_file(self, file_path: str) -> bool:
        """删除临时文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logging.error(f"删除文件出错: {str(e)}")
            return False

    async def get_documents_by_user_id(self, user_id: str) -> List[DocumentMetadata]:
        """根据用户ID从SQLite获取该用户上传的文档元数据列表"""
        documents = []
        try:
            async with aiosqlite.connect(self.DB_PATH) as db:
                # Set row_factory to aiosqlite.Row to access columns by name
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT doc_id, user_id, doc_name, doc_type, created_at, updated_at FROM document_metadata WHERE user_id = ? ORDER BY created_at DESC", 
                    (user_id,)
                )
                rows = await cursor.fetchall()
                
                if rows:
                    for row in rows:
                        # Convert row object to dict or directly access fields
                        documents.append(DocumentMetadata(
                            doc_id=row["doc_id"],
                            user_id=row["user_id"],
                            doc_name=row["doc_name"],
                            doc_type=row["doc_type"],
                            created_at=row["created_at"],
                            updated_at=row["updated_at"]
                        ))
            
            if not documents:
                logging.info(f"No documents found in SQLite for user_id: {user_id}")
            else:
                logging.info(f"Retrieved {len(documents)} documents from SQLite for user_id: {user_id}")
            
            return documents
        
        except Exception as e:
            logging.error(f"Error fetching documents from SQLite for user_id {user_id}: {str(e)}")
            # Depending on desired behavior, re-raise or return empty list/raise HTTPException
            return [] # Return empty list on error for now 

    async def delete_document(self, user_id: str, doc_id: str) -> bool:
        """删除文档及其相关数据
        
        Args:
            user_id: 用户ID
            doc_id: 文档ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # 1. 从SQLite删除元数据
            async with aiosqlite.connect(self.DB_PATH) as db:
                await db.execute(
                    "DELETE FROM document_metadata WHERE doc_id = ? AND user_id = ?",
                    (doc_id, user_id)
                )
                await db.commit()
            
            # 2. 从LlamaIndex管理的存储中删除（包括Milvus和NebulaGraph）
            # 获取用户的索引
            user_index = self.llama_index_manager._get_user_index(user_id)
            if user_index:
                # # 创建过滤器以匹配文档ID
                # filters = MetadataFilters(
                #     filters=[
                #         MetadataFilter(
                #             key="document_id",
                #             operator=FilterOperator.EQ,
                #             value=doc_id
                #         ),
                #         MetadataFilter(
                #             key="user_id",
                #             operator=FilterOperator.EQ,
                #             value=user_id
                #         )
                #     ]
                # )
                
                # 从向量存储中删除
                vec_store = self.llama_index_manager._get_vec_store(user_id)
                if vec_store:
                    vec_store.delete(ref_doc_id=doc_id)
                
                # 从图存储中删除
                graph_store = self.llama_index_manager._get_graph_store(user_id)
                if graph_store:
                    # 删除相关的节点和边
                    graph_store.delete(properties={"document_id": doc_id, "user_id": user_id})
                    # graph_store.delete_nodes(filters=filters)
                    # graph_store.delete_edges(filters=filters)
            
            logging.info(f"成功删除文档: user_id={user_id}, doc_id={doc_id}")
            return True
            
        except Exception as e:
            logging.error(f"删除文档时出错: {str(e)}")
            return False 