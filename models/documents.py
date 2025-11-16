"""Document model for knowledge base storage."""
from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey, func, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from .threads import Base
import enum


class IngestionStatus(enum.Enum):
    """Enum for document ingestion status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    """
    SQLAlchemy model for documents in the knowledge base.
    
    Stores metadata about uploaded documents with references to MinIO storage.
    """
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String(10), nullable=False)  # Extension: pdf, txt, md, doc, docx
    file_size = Column(Integer, nullable=False)  # Size in bytes
    minio_object_name = Column(String, unique=True, nullable=False)  # Unique object name in MinIO
    bucket_name = Column(String, nullable=False, default="documents")
    content_type = Column(String, nullable=False)  # MIME type
    document_metadata = Column(Text, nullable=True)  # JSON field for additional metadata
    ingestion_status = Column(Enum(IngestionStatus), default=IngestionStatus.PENDING, nullable=False)
    ingestion_error = Column(Text, nullable=True)  # Store error message if ingestion fails
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")