from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Body
from typing import Optional, Dict, Any, List

from app.models.schemas import (
    DocumentUploadRequest,
    QueryRequest,
    QueryResponse,
    DocumentSource,
    UserDocumentsResponse,
    DeleteDocumentRequest
)
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

router = APIRouter()

# 服务实例
document_service = DocumentService()
query_service = QueryService()


@router.post("/documents/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    doc_name: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
):
    """上传文档接口"""
    # 保存上传的文件
    file_path = document_service.save_uploaded_file(file, user_id)
    
    # 处理文档
    doc_id = await document_service.process_document(
        file_path=file_path,
        user_id=user_id,
        doc_name=doc_name or file.filename,
        doc_type=doc_type,
    )
    
    # 删除临时文件
    document_service.delete_temp_file(file_path)
    
    if not doc_id:
        raise HTTPException(status_code=500, detail="文档处理失败")
    
    return {"message": "文档上传成功", "doc_id": doc_id}


@router.get("/documents/user/{user_id}", response_model=UserDocumentsResponse)
async def list_user_documents(user_id: str):
    """获取指定用户上传的所有文档元数据"""
    documents = await document_service.get_documents_by_user_id(user_id)
    if not documents: # Optional: could also return empty list directly from service
        # Depending on preference, either return an empty list in the response model
        # or raise an HTTPException for 404 Not Found.
        # The current service method returns an empty list, which is fine.
        pass 
    return UserDocumentsResponse(documents=documents)


@router.post("/query", response_model=QueryResponse)
async def query(query_request: QueryRequest):
    """查询接口"""
    # 处理查询
    answer, sources = await query_service.process_query(query_request)
    
    if not answer:
        raise HTTPException(status_code=500, detail="查询处理失败")
    
    # 转换源文档
    source_docs = None
    if sources:
        source_docs = []
        for source in sources:
            source_docs.append(DocumentSource(
                doc_id=source["doc_id"],
                doc_name=source.get("file_name"),
                content=source["content"],
                score=source.get("score"),
            ))
    
    # 构建响应
    response = QueryResponse(
        answer=answer,
        sources=source_docs
    )
    
    return response 


@router.delete("/documents/delete", status_code=200)
async def delete_document(request: DeleteDocumentRequest):
    """删除文档接口"""
    success = await document_service.delete_document(
        user_id=request.user_id,
        doc_id=request.doc_id
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="文档删除失败")
    
    return {"message": "文档删除成功"} 