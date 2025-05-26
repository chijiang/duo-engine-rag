# LlamaIndex Dual-Engine RAG System

[![Project Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://github.com/chijiang/duo-engine-rag.git)

This project implements a sophisticated Retrieval Augmented Generation (RAG) system leveraging the power of LlamaIndex. It uniquely combines a Milvus vector database for semantic search with a NebulaGraph graph database for knowledge graph-based retrieval, offering a dual-engine approach to information retrieval. The system also features multi-user knowledge base isolation, ensuring data privacy and organization.

## Key Features

*   **Advanced Document Processing**: Automatically chunks documents, extracts key entities, and generates vector embeddings upon upload.
*   **Dual-Engine Retrieval Power**: Enhances search accuracy by simultaneously querying both vector (Milvus) and graph (NebulaGraph) databases, then intelligently fusing the results.
*   **Automated Knowledge Graph Construction**: Builds a dynamic knowledge graph by identifying entities and their relationships within your documents.
*   **Multi-User Architecture**: Provides distinct Milvus collections and NebulaGraph spaces for each user, guaranteeing knowledge base separation and security.
*   **Developer-Friendly RESTful API**: Offers a standardized set of API endpoints (built with FastAPI) for seamless integration into other applications.

## System Architecture

The system is composed of the following core components:

1.  **LlamaIndex Core**: Drives document processing, indexing, retrieval, and generation.
2.  **Milvus**: Serves as the vector database, storing and querying document embeddings for semantic similarity.
3.  **NebulaGraph**: Acts as the graph database, storing and querying the knowledge graph.
4.  **FastAPI Backend**: Exposes the system's functionalities through RESTful APIs.
5.  **Multi-User Isolation Layer**: Manages separate data stores for different users.

## Getting Started

### Prerequisites

Ensure the following are installed and running:

*   Milvus (2.x)
*   NebulaGraph (3.x)
*   Python 3.8+

### Installation

1.  Clone the repository (if you haven't already):
    ```bash
    # git clone https://github.com/chijiang/duo-engine-rag.git
    # cd duo-engine-rag
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Environment Configuration

Create a `.env` file in the project root (you can copy `.env.example` if provided) and configure the following:

```env
# API Configuration
API_PORT=8108
API_HOST=0.0.0.0

# LLM Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key # for llm
ZHIPU_API_KEY=your_zhipu_api_key # for embedding
LLM_MODEL=deepseek-chat  # Or your preferred model
EMBEDDING_MODEL=embedding-3  # Or your preferred embedding model

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_PREFIX=user_ # Prefix for user-specific collections

# NebulaGraph Configuration
NEBULA_HOSTS=localhost:9669 # Comma-separated if multiple hosts
NEBULA_USER=root
NEBULA_PASSWORD=nebula
NEBULA_SPACE_PREFIX=user_knowledge_ # Prefix for user-specific graph spaces

# LlamaIndex Configuration (Adjust as needed)
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TEXT_SPLITTER=sentence # Options: 'word', 'sentence', 'token', etc.
```

### Running the Application

There are a couple of ways to run the application:

1.  **Using the script in `app/main.py` (if configured for production/simpler startup):**
    ```bash
    python -m app.main
    ```
    *Note: Ensure `app/main.py` is set up to run uvicorn with the correct parameters, including `--loop asyncio` if needed, as shown below.*

2.  **Directly with Uvicorn (recommended for development, matches debug configuration):**
    ```bash
    python -m uvicorn app.main:app --reload --host $API_HOST --port $API_PORT --loop asyncio
    ```
    Or, using the specific values from your debug configuration if environment variables are not set in your shell:
    ```bash
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8108 --loop asyncio
    ```

**Important Note on `asyncio`:** LlamaIndex extensively uses `asyncio` for its operations. To ensure compatibility and prevent potential issues, FastAPI (and Uvicorn) **must** also be configured to use the `asyncio` event loop. This is achieved by including the `--loop asyncio` flag when running Uvicorn. Your `app/main.py` also includes `nest_asyncio.apply()` which helps manage asyncio event loops, particularly in environments like Jupyter notebooks, but explicitly setting the loop for Uvicorn is crucial for FastAPI services.

The API will be accessible at `http://localhost:8108` (or your configured host/port), with interactive documentation available at `http://localhost:8108/docs`.

## API Usage Examples

### Upload a Document

```bash
curl -X POST "http://localhost:8108/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" \
  -F "user_id=user123" \
  -F "doc_name=Example Document" \
  -F "doc_type=pdf"
```

**Response:**
```json
{
  "message": "文档上传成功",
  "doc_id": "generated-uuid-here"
}
```

### Get User Documents

```bash
curl -X GET "http://localhost:8108/api/documents/user/user123" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "documents": [
    {
      "doc_id": "generated-uuid-here",
      "doc_name": "Example Document",
      "doc_type": "pdf",
      "user_id": "user123",
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
  ]
}
```

### Perform a Query

```bash
curl -X POST "http://localhost:8108/api/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "query": "What are the main findings of this document?",
    "include_sources": true,
    "similarity_top_k": 5,
    "graph_top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "Based on the documents, the main findings include...",
  "sources": [
    {
      "doc_id": "generated-uuid-here",
      "doc_name": "Example Document",
      "content": "Relevant content snippet from the document...",
      "score": 0.85
    }
  ]
}
```

### Delete a Document

```bash
curl -X DELETE "http://localhost:8108/api/documents/delete" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "doc_id": "generated-uuid-here"
  }'
```

**Response:**
```json
{
  "message": "文档删除成功"
}
```

### API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/documents/upload` | Upload and process a document |
| GET | `/api/documents/user/{user_id}` | Get all documents for a user |
| POST | `/api/query` | Query the knowledge base |
| DELETE | `/api/documents/delete` | Delete a specific document |
| GET | `/` | API information and health check |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/redoc` | Alternative API documentation (ReDoc) |

## System Workflow

1.  **Document Ingestion**:
    *   User uploads a document via the API.
    *   The document is split into manageable chunks.
    *   Entities and relationships are extracted.
    *   Vector embeddings are computed for chunks.
    *   Data is indexed and stored in both Milvus (embeddings) and NebulaGraph (entities/relationships).
2.  **Query Processing**:
    *   User submits a query via the API.
    *   The query is processed by both the vector search engine (Milvus) and the graph search engine (NebulaGraph).
    *   Results from both engines are retrieved and synthesized.
    *   A consolidated answer is generated by the LLM and returned to the user.

## Project Structure

```
.
├── app/                  # Main application code
│   ├── api/              # API endpoint definitions (FastAPI routers)
│   ├── config/           # Configuration management (e.g., settings, .env loading)
│   ├── core/             # Core logic (indexing, querying, RAG pipeline)
│   ├── db/               # Database interaction (Milvus, NebulaGraph connectors)
│   ├── models/           # Pydantic models for API requests/responses and data structures
│   ├── services/         # Business logic layer
│   ├── utils/            # Utility functions
│   └── main.py           # FastAPI application entry point
├── data/                 # (Optional) Persistent data, non-versioned (e.g. local Milvus/Nebula data if not dockerized)
├── uploads/              # Temporary storage for uploaded files
├── books/                # (Example content, if applicable for sample data)
├── .env                  # Local environment variables (ignored by Git)
├── .env.example          # Example environment file
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Technology Stack

*   **Core Framework**: LlamaIndex
*   **Web Framework**: FastAPI
*   **Vector Database**: Milvus
*   **Graph Database**: NebulaGraph
*   **LLM & Embeddings**: OpenAI (GPT models, Ada embeddings) - *Note: `transformers` and `torch` are in `requirements.txt`, suggesting flexibility for local models.*
*   **Programming Language**: Python

## Future Enhancements

*   Support for a wider range of document formats.
*   More sophisticated entity and relationship extraction.
*   Implementation of user authentication and granular permission management.
*   Advanced retrieval and result fusion strategies.
*   Development of a user-friendly web interface.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](http://creativecommons.org/licenses/by-nc/4.0/). 

You are free to:
*   **Share** — copy and redistribute the material in any medium or format
*   **Adapt** — remix, transform, and build upon the material

Under the following terms:
*   **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
*   **NonCommercial** — You may not use the material for commercial purposes.

See the `LICENSE` file in the root of the project for more details.

---

*This README was last updated on 2025-05-20.* 