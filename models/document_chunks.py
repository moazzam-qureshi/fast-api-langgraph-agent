"""Document chunk model for storing embeddings."""
from sqlalchemy import Column, String, Integer, Text, ForeignKey, func, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from uuid import uuid4
from .threads import Base


class DocumentChunk(Base):
    """
    SQLAlchemy model for document chunks with embeddings.
    
    Stores text chunks from documents with their vector embeddings for RAG.
    """
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)  # Sequential index of chunk in document
    chunk_text = Column(Text, nullable=False)  # The actual text content
    embedding = Column(Vector(1536), nullable=True)  # OpenAI embeddings are 1536 dimensions
    chunk_metadata = Column(JSON, nullable=True)  # Store page_number, section, etc.
    token_count = Column(Integer, nullable=False)  # Number of tokens in chunk
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Create indexes for efficient searching
    __table_args__ = (
        # Index for user-filtered searches
        {"schema": None}
    )