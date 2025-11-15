"""Thread model for conversation management."""
from sqlalchemy import Column, String, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from uuid import uuid4

Base = declarative_base()


class Thread(Base):
    """
    SQLAlchemy model for conversation threads.
    
    Each thread represents a conversation between a user and the AI.
    The actual messages are stored by LangGraph using the thread_id.
    """
    __tablename__ = "threads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String(255), nullable=True, default="New Chat")
    thread_metadata = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())