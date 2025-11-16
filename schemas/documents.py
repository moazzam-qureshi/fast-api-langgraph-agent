"""Document schemas for knowledge base requests and responses."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class DocumentBase(BaseModel):
    """Base document schema."""
    filename: str = Field(..., description="Original filename")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class DocumentUpload(BaseModel):
    """Schema for document upload metadata."""
    # File will be received as UploadFile, this is just for additional metadata if needed
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    """Schema for document responses."""
    id: UUID
    user_id: UUID
    file_type: str
    file_size: int
    content_type: str
    bucket_name: str
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Schema for paginated document list."""
    documents: List[DocumentResponse]
    total: int
    page: int = Field(default=1)
    page_size: int = Field(default=20)
    
    model_config = ConfigDict(from_attributes=True)


class DocumentDownloadResponse(BaseModel):
    """Schema for document download response."""
    download_url: str = Field(description="Pre-signed URL for downloading the document")
    expires_in: int = Field(description="URL expiration time in seconds")
    filename: str
    content_type: str


class AllowedFileTypes(BaseModel):
    """Schema for listing allowed file types."""
    extensions: List[str] = [".pdf", ".txt", ".md", ".doc", ".docx"]
    max_size_mb: int = 10
    max_size_bytes: int = 10 * 1024 * 1024  # 10MB in bytes