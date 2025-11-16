# RAG (Retrieval-Augmented Generation) Implementation

This project now includes a full RAG implementation for the knowledge base, allowing users to upload documents and query them using AI.

## Features

- **Document Upload**: Support for PDF, TXT, MD, DOC, and DOCX files
- **Background Processing**: Asynchronous document ingestion using Celery
- **Vector Embeddings**: OpenAI embeddings stored in PostgreSQL with pgvector
- **Similarity Search**: Fast vector similarity search across user documents
- **LangGraph Integration**: RAG seamlessly integrated into the chat workflow

## Setup

1. **Environment Variables**
   Copy `.env.example` to `.env` and set:
   - `OPENAI_API_KEY`: Your OpenAI API key for embeddings
   - `SECRET_KEY`: JWT secret key for authentication

2. **Start Services**
   ```bash
   docker-compose up -d
   ```
   This starts:
   - PostgreSQL with pgvector extension
   - Redis for Celery broker
   - MinIO for document storage
   - FastAPI application
   - Celery worker for background processing

3. **Database Migration**
   The database tables will be created automatically on startup.

## Usage

### 1. Upload a Document
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

### 2. Trigger Document Ingestion
```bash
curl -X POST "http://localhost:8000/documents/{document_id}/ingest" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. Check Ingestion Status
```bash
curl -X GET "http://localhost:8000/documents/{document_id}/ingestion-status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Search Documents
```bash
curl -X POST "http://localhost:8000/documents/search" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 5,
    "score_threshold": 0.7
  }'
```

### 5. Chat with RAG
```bash
curl -X POST "http://localhost:8000/stream" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "thread-123",
    "message": "What does the document say about AI?",
    "use_rag": true,
    "rag_k": 5,
    "rag_threshold": 0.7
  }'
```

## Architecture

### Document Processing Pipeline

1. **Upload**: Document stored in MinIO
2. **Ingestion Trigger**: Celery task queued
3. **Text Extraction**: Based on file type (PDF, DOCX, etc.)
4. **Chunking**: Text split into ~1000 token chunks with 150 token overlap
5. **Embedding Generation**: OpenAI embeddings for each chunk
6. **Storage**: Chunks and embeddings stored in PostgreSQL

### RAG Flow

1. User sends query with `use_rag=true`
2. System searches for similar chunks in user's documents
3. Top-k relevant chunks retrieved based on cosine similarity
4. Context passed to LLM along with user query
5. LLM generates response using retrieved context

## API Endpoints

### Document Management
- `POST /documents/upload` - Upload a new document
- `GET /documents` - List user's documents
- `GET /documents/{id}` - Get document details
- `GET /documents/{id}/download` - Download document
- `DELETE /documents/{id}` - Delete document

### RAG Operations
- `POST /documents/{id}/ingest` - Start document ingestion
- `GET /documents/{id}/ingestion-status` - Check ingestion status
- `POST /documents/search` - Search across documents

### Chat
- `POST /stream` - Chat with optional RAG support

## Configuration

### Chunking Parameters
- **Chunk Size**: 1000 tokens (configurable in `services/ingestion.py`)
- **Chunk Overlap**: 150 tokens
- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)

### Search Parameters
- **Default k**: 5 results
- **Default Threshold**: 0.7 similarity score
- **Distance Metric**: Cosine similarity

## Monitoring

### Celery Tasks
Monitor Celery tasks:
```bash
docker logs fastapi-celery-worker -f
```

### Database
Check document chunks:
```sql
SELECT COUNT(*) FROM document_chunks WHERE user_id = 'USER_ID';
```

## Troubleshooting

1. **Ingestion Fails**: Check Celery worker logs and document `ingestion_error` field
2. **No Results**: Lower the similarity threshold or check if documents are ingested
3. **Slow Search**: Create vector index if not exists (done automatically)

## Performance Optimization

1. **Vector Index**: IVFFlat index created automatically for faster searches
2. **Batch Processing**: Embeddings generated in batches of 10
3. **Async Processing**: All ingestion done in background via Celery