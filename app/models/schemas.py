from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    """用户信息模型"""
    user_id: str = Field(..., description="用户唯一ID")
    username: Optional[str] = Field(None, description="用户名")
    
    
class DocumentMetadata(BaseModel):
    """文档元数据模型"""
    doc_id: str = Field(..., description="文档唯一ID")
    doc_name: str = Field(..., description="文档名称")
    doc_type: str = Field(..., description="文档类型")
    user_id: str = Field(..., description="所属用户ID")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")


class DocumentUploadRequest(BaseModel):
    """文档上传请求模型"""
    user_id: str = Field(..., description="用户ID")
    doc_name: Optional[str] = Field(None, description="文档名称")
    doc_type: Optional[str] = Field(None, description="文档类型")


class QueryRequest(BaseModel):
    """查询请求模型"""
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="用户查询内容")
    top_k: Optional[int] = Field(None, description="返回的结果数量")
    include_sources: Optional[bool] = Field(True, description="是否包含来源文档")
    similarity_top_k: Optional[int] = Field(None, description="向量检索返回数量")
    graph_top_k: Optional[int] = Field(None, description="图检索返回数量")


class DocumentSource(BaseModel):
    """文档来源模型"""
    doc_id: str = Field(..., description="文档ID")
    doc_name: Optional[str] = Field(None, description="文档名称")
    content: str = Field(..., description="文档内容片段")
    score: Optional[float] = Field(None, description="相关性分数")
    # source_type: str = Field(..., description="来源类型: vector/graph")


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="生成的回答")
    sources: Optional[List[DocumentSource]] = Field(None, description="来源文档")
    

class EntityNode(BaseModel):
    """实体节点模型，用于图数据库"""
    node_id: str = Field(..., description="节点ID")
    node_type: str = Field(..., description="节点类型: entity/chunk")
    name: str = Field(..., description="节点名称")
    properties: Optional[Dict[str, Any]] = Field(None, description="节点属性")
    

class Relationship(BaseModel):
    """关系模型，用于图数据库"""
    source_id: str = Field(..., description="源节点ID")
    target_id: str = Field(..., description="目标节点ID")
    rel_type: str = Field(..., description="关系类型")
    properties: Optional[Dict[str, Any]] = Field(None, description="关系属性") 