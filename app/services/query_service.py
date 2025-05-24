import logging
from typing import Dict, List, Any, Tuple, Optional

from app.core.llama_index_manager import LlamaIndexManager
from app.models.schemas import QueryRequest, QueryResponse, DocumentSource


class QueryService:
    """查询服务，处理用户查询"""
    
    def __init__(self):
        self.llama_index_manager = LlamaIndexManager()
    
    async def process_query(self, query_request: QueryRequest) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        """处理用户查询"""
        try:
            # 从请求中提取参数
            user_id = query_request.user_id
            query_text = query_request.query
            similarity_top_k = query_request.similarity_top_k
            graph_top_k = query_request.graph_top_k
            include_sources = query_request.include_sources
            
            # 进行查询
            answer, sources = await self.llama_index_manager.query(
                user_id=user_id,
                query_text=query_text,
                similarity_top_k=similarity_top_k,
                graph_top_k=graph_top_k
            )
            
            if not include_sources:
                sources = None
            
            return answer, sources
        
        except Exception as e:
            logging.error(f"处理查询出错: {str(e)}")
            return None, None 