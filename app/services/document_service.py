import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from app.core.llama_index_manager import LlamaIndexManager
from app.models.schemas import DocumentMetadata


class DocumentService:
    """文档服务，处理文档上传和处理"""
    
    def __init__(self):
        self.llama_index_manager = LlamaIndexManager()
        self.upload_dir = "uploads"
        
        # 确保上传目录存在
        os.makedirs(self.upload_dir, exist_ok=True)
    
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
        """处理文档"""
        try:
            # 如果没有提供文档名，则使用原始文件名
            if not doc_name:
                doc_name = os.path.basename(file_path)
            
            # 如果没有提供文档类型，则根据文件扩展名推断
            if not doc_type:
                doc_type = os.path.splitext(file_path)[1].lstrip('.')
            
            # 准备文档元数据
            metadata = {
                "file_name": doc_name,
                "file_type": doc_type,
                "user_id": user_id,
            }
            
            # 处理文档
            doc_id = await self.llama_index_manager.process_document(
                file_path=file_path,
                user_id=user_id,
                doc_metadata=metadata
            )
            
            return doc_id
        
        except Exception as e:
            logging.error(f"处理文档出错: {str(e)}")
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