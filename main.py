from fastapi import FastAPI, Depends, HTTPException, Query, Request, status
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

# Local imports
from database import get_db, engine
from models import Base, User
from schemas import ThreadCreate, ThreadUpdate, ThreadResponse, UserCreate, UserLogin, UserUpdate, UserResponse, Token
from services import ThreadService, AuthService
from sqlalchemy.orm import Session



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create threads table if it doesn't exist
    Base.metadata.create_all(bind=engine)
    
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
    return {"status": "healthy", "service": "fastapi-minimal"}

@app.post("/stream")
async def stream(req: ChatRequest):
    # access the compiled graph from app.state
    simple_graph = app.state.graph

    return StreamingResponse(
        create_sse_stream(simple_graph, {"user_query": req.message}, req.thread_id),
        media_type="text/event-stream"
    )

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


