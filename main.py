from fastapi import FastAPI, Depends, HTTPException, Query, Request, status, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dtos.chat_request import ChatRequest
from graph import s_graph
from utils import create_sse_stream
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from contextlib import asynccontextmanager
import os
from typing import List, Optional
from uuid import UUID
import json
import logging

logger = logging.getLogger(__name__)

# Local imports
from database import get_db, engine
from models import Base, User, Document, DocumentChunk, IngestionStatus
from schemas import (
    ThreadCreate, ThreadUpdate, ThreadResponse,
    UserCreate, UserLogin, UserUpdate, UserResponse, Token,
    DocumentUpload, DocumentResponse, DocumentListResponse,
    IngestionStatusResponse, DocumentSearchRequest, DocumentSearchResponse, DocumentSearchResult
)
from services import ThreadService, AuthService, DocumentService, minio_service, vector_store_service
from services.ingestion import ingest_document
from sqlalchemy.orm import Session
from sqlalchemy import func, text



@asynccontextmanager
async def lifespan(app: FastAPI):
    # First, ensure pgvector extension is enabled
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.error(f"Failed to create pgvector extension: {e}")
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Add missing columns to existing tables (migration)
    with engine.connect() as conn:
        try:
            # Check if ingestion_status column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'documents' 
                AND column_name = 'ingestion_status'
            """))
            
            if not result.fetchone():
                # Add missing columns to documents table
                logger.info("Adding missing columns to documents table")
                conn.execute(text("""
                    ALTER TABLE documents 
                    ADD COLUMN IF NOT EXISTS ingestion_status VARCHAR DEFAULT 'pending',
                    ADD COLUMN IF NOT EXISTS ingestion_error TEXT
                """))
                conn.commit()
                logger.info("Added ingestion columns to documents table")
        except Exception as e:
            logger.error(f"Error during migration: {e}")
    
    # Create vector index for better performance
    from services.vectorstore import vector_store_service
    db = next(get_db())
    try:
        vector_store_service.create_vector_index(db)
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
    finally:
        db.close()
    
    # ENTER lifespan: properly enter the async context manager
    async with AsyncPostgresSaver.from_conn_string(
        "postgresql://postgres:postgres@postgres:5432/postgres"
    ) as saver:
        await saver.setup()  # initialize tables if needed
        app.state.checkpointer = saver
        app.state.graph = s_graph.compile(checkpointer=saver)

        # yield control back to FastAPI; app is now ready
        yield

        # EXIT lifespan handled automatically by async with
        # No need to call aclose manually; context manager closes cleanly


app = FastAPI(
    title="Minimal FastAPI App",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)


@app.get("/")
async def root():
    return {"message": "Hello World", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fastapi-langgraph-agent"}


@app.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Comprehensive health check for all services."""
    health_status = {
        "status": "healthy",
        "service": "fastapi-langgraph-agent",
        "checks": {}
    }
    
    # Check database connection
    try:
        db.execute(text("SELECT 1"))
        db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        health_status["checks"]["database"] = {"status": "healthy", "type": "postgresql_with_pgvector"}
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check MinIO connection
    try:
        buckets = minio_service.client.list_buckets()
        health_status["checks"]["minio"] = {"status": "healthy", "buckets": len(buckets)}
    except Exception as e:
        health_status["checks"]["minio"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check Redis connection (through Celery)
    try:
        from celery_app import celery_app
        celery_stats = celery_app.control.inspect().stats()
        health_status["checks"]["redis"] = {"status": "healthy" if celery_stats else "degraded"}
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check OpenAI API key configuration
    health_status["checks"]["openai"] = {
        "status": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"
    }
    
    return health_status


@app.get("/session/{thread_id}")
async def get_session(thread_id: str):
    # Access the compiled graph from app.state
    graph = app.state.graph
    
    # Get the current state using the thread_id config
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get the current state from the graph
        state = await graph.aget_state(config)

        print("state", state)
        
        if state and state.values:
            # Get messages from the state
            raw_messages = state.values.get("messages", [])
            
            # Convert LangChain message objects to dictionaries
            messages = []
            for msg in raw_messages:
                messages.append({
                    "content": msg.content,
                    "type": msg.type,  # "human" or "ai"
                    "id": msg.id if hasattr(msg, 'id') else None,
                    "additional_kwargs": msg.additional_kwargs if hasattr(msg, 'additional_kwargs') else {},
                    "response_metadata": msg.response_metadata if hasattr(msg, 'response_metadata') else {}
                })
        else:
            messages = []
    except Exception as e:
        # Handle case where thread doesn't exist
        messages = []

    return {"thread_id": thread_id, "messages": messages}


# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Dependency to get current user from JWT token
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_payload = AuthService.decode_token(token)
    if token_payload is None or token_payload.type != "access":
        raise credentials_exception
    
    try:
        user_id = UUID(token_payload.sub)
    except ValueError:
        raise credentials_exception
    
    user = AuthService.get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception
    
    return user

# Helper function to get user ID from request
def get_current_user_id(request: Request) -> str:
    """Extract user ID from request headers."""
    return request.headers.get("X-User-ID", "default-user")


# Stream endpoint with authentication
@app.post("/stream")
async def stream(
    req: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    # access the compiled graph from app.state
    simple_graph = app.state.graph

    # Build input with user ID and RAG parameters
    input_data = {
        "user_query": req.message,
        "user_id": str(current_user.id),
        "use_rag": req.use_rag,
        "rag_k": req.rag_k,
        "rag_threshold": req.rag_threshold
    }

    return StreamingResponse(
        create_sse_stream(simple_graph, input_data, req.thread_id),
        media_type="text/event-stream"
    )


# Thread management endpoints
@app.post("/threads", response_model=ThreadResponse)
async def create_thread(
    thread: ThreadCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> ThreadResponse:
    """Create a new conversation thread."""
    user_id = str(current_user.id)
    
    db_thread = ThreadService.create_thread(
        db=db,
        user_id=user_id,
        thread_data=thread
    )
    
    return ThreadResponse(
        id=db_thread.id,
        user_id=db_thread.user_id,
        title=db_thread.title,
        metadata=json.loads(db_thread.thread_metadata) if db_thread.thread_metadata else None,
        created_at=db_thread.created_at,
        updated_at=db_thread.updated_at
    )


@app.get("/threads", response_model=List[ThreadResponse])
async def list_threads(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[ThreadResponse]:
    """List all threads for the authenticated user."""
    user_id = str(current_user.id)
    
    threads = ThreadService.get_user_threads(
        db=db,
        user_id=user_id,
        skip=skip,
        limit=limit
    )
    
    return [
        ThreadResponse(
            id=thread.id,
            user_id=thread.user_id,
            title=thread.title,
            metadata=json.loads(thread.thread_metadata) if thread.thread_metadata else None,
            created_at=thread.created_at,
            updated_at=thread.updated_at
        )
        for thread in threads
    ]


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> ThreadResponse:
    """Get a specific thread by ID."""
    user_id = str(current_user.id)
    
    thread = ThreadService.get_thread(
        db=db,
        thread_id=thread_id,
        user_id=user_id
    )
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return ThreadResponse(
        id=thread.id,
        user_id=thread.user_id,
        title=thread.title,
        metadata=json.loads(thread.thread_metadata) if thread.thread_metadata else None,
        created_at=thread.created_at,
        updated_at=thread.updated_at
    )


@app.patch("/threads/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: UUID,
    thread_update: ThreadUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> ThreadResponse:
    """Update a thread's title or metadata."""
    user_id = str(current_user.id)
    
    updated_thread = ThreadService.update_thread(
        db=db,
        thread_id=thread_id,
        user_id=user_id,
        thread_update=thread_update
    )
    
    if not updated_thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return ThreadResponse(
        id=updated_thread.id,
        user_id=updated_thread.user_id,
        title=updated_thread.title,
        metadata=json.loads(updated_thread.thread_metadata) if updated_thread.thread_metadata else None,
        created_at=updated_thread.created_at,
        updated_at=updated_thread.updated_at
    )


@app.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Delete a thread."""
    user_id = str(current_user.id)
    
    deleted = ThreadService.delete_thread(
        db=db,
        thread_id=thread_id,
        user_id=user_id
    )
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {"message": "Thread deleted successfully"}


# Authentication endpoints
@app.post("/auth/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    db: Session = Depends(get_db)
) -> UserResponse:
    """Register a new user."""
    # Check if user exists
    if AuthService.get_user_by_email(db, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if AuthService.get_user_by_username(db, user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    user = AuthService.create_user(db, user_data)
    
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@app.post("/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
) -> Token:
    """Login with username/email and password."""
    user = AuthService.authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token = AuthService.create_access_token(user.id)
    refresh_token = AuthService.create_refresh_token(user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )


@app.post("/auth/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
) -> Token:
    """Refresh access token using refresh token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_payload = AuthService.decode_token(refresh_token)
    if token_payload is None or token_payload.type != "refresh":
        raise credentials_exception
    
    try:
        user_id = UUID(token_payload.sub)
    except ValueError:
        raise credentials_exception
    
    user = AuthService.get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    
    access_token = AuthService.create_access_token(user.id)
    new_refresh_token = AuthService.create_refresh_token(user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )


@app.patch("/auth/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> UserResponse:
    """Update current user information."""
    # Check if email is being updated and already exists
    if user_update.email and user_update.email != current_user.email:
        if AuthService.get_user_by_email(db, user_update.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if username is being updated and already exists
    if user_update.username and user_update.username != current_user.username:
        if AuthService.get_user_by_username(db, user_update.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    updated_user = AuthService.update_user(db, current_user.id, user_update)
    
    return UserResponse(
        id=updated_user.id,
        email=updated_user.email,
        username=updated_user.username,
        full_name=updated_user.full_name,
        is_active=updated_user.is_active,
        is_superuser=updated_user.is_superuser,
        created_at=updated_user.created_at,
        updated_at=updated_user.updated_at
    )


# Document management endpoints
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DocumentResponse:
    """
    Upload a document to the knowledge base.
    
    Allowed file types: .pdf, .txt, .md, .doc, .docx
    Maximum file size: 10MB
    """
    # Read file content
    file_content = await file.read()
    file_size = len(file_content)
    
    # Reset file position for upload
    await file.seek(0)
    
    try:
        # Upload document
        document = DocumentService.upload_document(
            db=db,
            user_id=current_user.id,
            filename=file.filename,
            file_data=file.file,
            file_size=file_size,
            content_type=file.content_type or "application/octet-stream"
        )
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload document"
            )
        
        return DocumentResponse(
            id=document.id,
            user_id=document.user_id,
            filename=document.filename,
            file_type=document.file_type,
            file_size=document.file_size,
            content_type=document.content_type,
            bucket_name=document.bucket_name,
            metadata=json.loads(document.document_metadata) if document.document_metadata else None,
            ingestion_status=document.ingestion_status,
            ingestion_error=document.ingestion_error,
            created_at=document.created_at,
            updated_at=document.updated_at
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DocumentListResponse:
    """List all documents for the authenticated user."""
    documents = DocumentService.get_user_documents(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    
    total = DocumentService.count_user_documents(db, current_user.id)
    
    document_responses = [
        DocumentResponse(
            id=doc.id,
            user_id=doc.user_id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            content_type=doc.content_type,
            bucket_name=doc.bucket_name,
            metadata=json.loads(doc.document_metadata) if doc.document_metadata else None,
            ingestion_status=doc.ingestion_status,
            ingestion_error=doc.ingestion_error,
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )
        for doc in documents
    ]
    
    return DocumentListResponse(
        documents=document_responses,
        total=total,
        page=(skip // limit) + 1,
        page_size=limit
    )


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DocumentResponse:
    """Get metadata for a specific document."""
    document = DocumentService.get_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return DocumentResponse(
        id=document.id,
        user_id=document.user_id,
        filename=document.filename,
        file_type=document.file_type,
        file_size=document.file_size,
        content_type=document.content_type,
        bucket_name=document.bucket_name,
        metadata=json.loads(document.document_metadata) if document.document_metadata else None,
        ingestion_status=document.ingestion_status,
        ingestion_error=document.ingestion_error,
        created_at=document.created_at,
        updated_at=document.updated_at
    )


@app.get("/documents/{document_id}/download")
async def download_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download a document directly."""
    # Get document metadata
    document = DocumentService.get_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Download file content from MinIO
    file_content = minio_service.download_file(
        object_name=document.minio_object_name,
        bucket_name=document.bucket_name
    )
    
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download file"
        )
    
    # Return file as streaming response
    from io import BytesIO
    import urllib.parse
    
    # Encode filename for HTTP headers (handle Unicode characters)
    safe_filename = document.filename.encode('ascii', 'ignore').decode('ascii')
    if safe_filename != document.filename:
        # If filename contains non-ASCII characters, use RFC 5987 encoding
        encoded_filename = urllib.parse.quote(document.filename)
        content_disposition = f"attachment; filename=\"{safe_filename}\"; filename*=UTF-8''{encoded_filename}"
    else:
        content_disposition = f'attachment; filename="{document.filename}"'
    
    return StreamingResponse(
        BytesIO(file_content),
        media_type=document.content_type,
        headers={
            "Content-Disposition": content_disposition
        }
    )




@app.get("/documents/{document_id}/url")
async def get_document_url(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """
    Get document download URL.
    Note: Use /documents/{id}/download for direct download instead.
    """
    document = DocumentService.get_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Return the API endpoint for downloading
    download_url = f"/documents/{document_id}/download"
    
    return {
        "download_url": download_url,
        "filename": document.filename,
        "content_type": document.content_type,
        "note": "Use this URL with your authentication token to download the file"
    }



@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Delete a document from the knowledge base."""
    success = DocumentService.delete_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or deletion failed"
        )
    
    return {"message": "Document deleted successfully"}


# Document ingestion and RAG endpoints
@app.post("/documents/{document_id}/ingest")
async def trigger_document_ingestion(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Trigger background ingestion for a document."""
    # Get document and verify ownership
    document = DocumentService.get_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check if already processing or completed
    if document.ingestion_status == IngestionStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is already being processed"
        )
    
    if document.ingestion_status == IngestionStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has already been ingested. Delete and re-upload to process again."
        )
    
    # Trigger Celery task
    task = ingest_document.delay(str(document.id), str(current_user.id))
    
    return {
        "message": "Ingestion started",
        "document_id": str(document_id),
        "task_id": task.id
    }


@app.get("/documents/{document_id}/ingestion-status", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> IngestionStatusResponse:
    """Get the ingestion status of a document."""
    # Get document and verify ownership
    document = DocumentService.get_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get chunk statistics if ingested
    chunks_count = None
    total_tokens = None
    
    if document.ingestion_status == IngestionStatus.COMPLETED:
        chunks = db.query(
            func.count(DocumentChunk.id).label('count'),
            func.sum(DocumentChunk.token_count).label('total_tokens')
        ).filter(
            DocumentChunk.document_id == document.id
        ).first()
        
        if chunks:
            chunks_count = chunks.count
            total_tokens = chunks.total_tokens
    
    return IngestionStatusResponse(
        document_id=document.id,
        status=document.ingestion_status,
        error_message=document.ingestion_error,
        chunks_count=chunks_count,
        total_tokens=total_tokens
    )


@app.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    search_request: DocumentSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DocumentSearchResponse:
    """Search across user's documents using vector similarity."""
    # Search for similar chunks
    results = vector_store_service.search_similar_chunks(
        query=search_request.query,
        user_id=current_user.id,
        k=search_request.k,
        score_threshold=search_request.score_threshold,
        filter_document_ids=search_request.document_ids,
        db=db
    )
    
    # Get document information for results
    formatted_results = []
    for result in results:
        # Get document info
        document = db.query(Document).filter(
            Document.id == result["document_id"]
        ).first()
        
        if document:
            formatted_results.append(DocumentSearchResult(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                filename=document.filename,
                chunk_text=result["chunk_text"],
                chunk_index=result["chunk_index"],
                similarity_score=result["similarity_score"],
                metadata=result["metadata"]
            ))
    
    return DocumentSearchResponse(
        query=search_request.query,
        results=formatted_results,
        total_results=len(formatted_results)
    )


