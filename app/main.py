import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.router import api_router
from app.config.settings import API_HOST, API_PORT
from app.api.endpoints import document_service

import nest_asyncio
nest_asyncio.apply()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logging.info("应用程序启动：正在初始化数据库...")
    await document_service.initialize_db()
    logging.info("数据库初始化完成。")
    yield
    logging.info("应用程序关闭。")

# 创建FastAPI应用
app = FastAPI(
    title="LlamaIndex双引擎RAG系统",
    description="基于LlamaIndex框架，集成Milvus向量数据库和NebulaGraph图数据库的RAG系统",
    version="0.1.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载API路由
app.include_router(api_router)


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "欢迎使用LlamaIndex双引擎RAG系统API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


# 如果作为主程序运行，启动Uvicorn服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        loop="asyncio"
    ) 