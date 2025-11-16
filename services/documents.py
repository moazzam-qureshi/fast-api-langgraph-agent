"""Document service for CRUD operations."""
from typing import Optional, List, BinaryIO, Tuple
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from sqlalchemy import desc
import json
import os
from datetime import timedelta

from models.documents import Document
from schemas.documents import DocumentUpload
from services.minio import minio_service


# Allowed file extensions and their MIME types
ALLOWED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class DocumentService:
    """Service class for document CRUD operations."""
    
    @staticmethod
    def validate_file(filename: str, file_size: int) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate file type and size.
        
        Returns:
            Tuple of (is_valid, error_message, file_extension)
        """
        # Check file size
        if file_size > MAX_FILE_SIZE:
            return False, f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // 1024 // 1024}MB", None
        
        # Check file extension
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return False, f"File type {file_extension} is not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS.keys())}", None
        
        return True, None, file_extension
    
    @staticmethod
    def generate_object_name(user_id: UUID, filename: str) -> str:
        """Generate a unique object name for MinIO storage."""
        file_extension = os.path.splitext(filename)[1]
        unique_id = uuid4()
        return f"{user_id}/{unique_id}{file_extension}"
    
    @staticmethod
    def upload_document(
        db: Session,
        user_id: UUID,
        filename: str,
        file_data: BinaryIO,
        file_size: int,
        content_type: str,
        metadata: Optional[dict] = None
    ) -> Optional[Document]:
        """
        Upload a document to MinIO and save metadata to database.
        
        Args:
            db: Database session
            user_id: User ID
            filename: Original filename
            file_data: File binary data
            file_size: File size in bytes
            content_type: MIME type
            metadata: Optional metadata
            
        Returns:
            Document object if successful, None otherwise
        """
        # Validate file
        is_valid, error_msg, file_extension = DocumentService.validate_file(filename, file_size)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Generate unique object name
        object_name = DocumentService.generate_object_name(user_id, filename)
        
        # Upload to MinIO
        success = minio_service.upload_file(
            file_data=file_data,
            object_name=object_name,
            content_type=content_type,
            file_size=file_size
        )
        
        if not success:
            return None
        
        # Save metadata to database
        db_document = Document(
            user_id=user_id,
            filename=filename,
            file_type=file_extension[1:],  # Remove the dot
            file_size=file_size,
            minio_object_name=object_name,
            bucket_name=minio_service.default_bucket,
            content_type=content_type,
            document_metadata=json.dumps(metadata) if metadata else None
        )
        
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        return db_document
    
    @staticmethod
    def get_document(db: Session, document_id: UUID, user_id: UUID) -> Optional[Document]:
        """Get a document by ID, ensuring user ownership."""
        return db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
    
    @staticmethod
    def get_user_documents(
        db: Session,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20
    ) -> List[Document]:
        """Get all documents for a user."""
        return db.query(Document).filter(
            Document.user_id == user_id
        ).order_by(
            desc(Document.created_at)
        ).offset(skip).limit(limit).all()
    
    @staticmethod
    def count_user_documents(db: Session, user_id: UUID) -> int:
        """Count total documents for a user."""
        return db.query(Document).filter(
            Document.user_id == user_id
        ).count()
    
    @staticmethod
    def get_download_url(
        db: Session,
        document_id: UUID,
        user_id: UUID,
        expires_hours: int = 1
    ) -> Optional[Tuple[str, Document]]:
        """
        Get a pre-signed download URL for a document.
        
        Returns:
            Tuple of (download_url, document) if successful, None otherwise
        """
        document = DocumentService.get_document(db, document_id, user_id)
        if not document:
            return None
        
        url = minio_service.get_presigned_url(
            object_name=document.minio_object_name,
            bucket_name=document.bucket_name,
            expires=timedelta(hours=expires_hours)
        )
        
        if not url:
            return None
        
        return url, document
    
    @staticmethod
    def delete_document(db: Session, document_id: UUID, user_id: UUID) -> bool:
        """
        Delete a document from both MinIO and database.
        
        Args:
            db: Database session
            document_id: Document ID
            user_id: User ID for ownership verification
            
        Returns:
            bool: True if successful, False otherwise
        """
        document = DocumentService.get_document(db, document_id, user_id)
        if not document:
            return False
        
        # Delete from MinIO
        success = minio_service.delete_file(
            object_name=document.minio_object_name,
            bucket_name=document.bucket_name
        )
        
        if not success:
            return False
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return True