import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API相关配置
API_PORT = int(os.getenv("API_PORT", 8081))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# 模型配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

LLM_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")

# Milvus配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
MILVUS_COLLECTION_PREFIX = os.getenv("MILVUS_COLLECTION_PREFIX", "kg_collection_")
MILVUS_DIMENSION = int(os.getenv("MILVUS_DIMENSION", 1536))

# NebulaGraph配置
NEBULA_HOSTS = os.getenv("NEBULA_HOSTS", "localhost:9669")
NEBULA_USER = os.getenv("NEBULA_USER", "root")
NEBULA_PASSWORD = os.getenv("NEBULA_PASSWORD", "nebula")

# LlamaIndex配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TEXT_SPLITTER = os.getenv("TEXT_SPLITTER", "sentence")

# 系统默认配置
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))  # 默认检索文档数量
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))  # 向量相似度检索文档数
GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", 5))