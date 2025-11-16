"""Document ingestion service for extracting text, chunking, and generating embeddings."""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.orm import Session

# Document processing imports
import pypdf
from docx import Document as DocxDocument
import markdown

from celery_app import celery
from database import get_db_sync
from models import Document, DocumentChunk, IngestionStatus
from services.minio import minio_service

logger = logging.getLogger(__name__)

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Token counter for chunk sizing
encoding = tiktoken.encoding_for_model("gpt-4o-mini")


class DocumentProcessor:
    """Handles document text extraction based on file type."""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            import io
            pdf_file = io.BytesIO(file_content)
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
        
        return text.strip()
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file."""
        text = ""
        try:
            import io
            docx_file = io.BytesIO(file_content)
            doc = DocxDocument(docx_file)
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n\n"
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise
        
        return text.strip()
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes) -> str:
        """Extract text from TXT file."""
        try:
            return file_content.decode('utf-8', errors='ignore').strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            raise
    
    @staticmethod
    def extract_text_from_md(file_content: bytes) -> str:
        """Extract text from Markdown file."""
        try:
            md_text = file_content.decode('utf-8', errors='ignore')
            # Convert markdown to plain text (keeping structure)
            return md_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from MD: {e}")
            raise
    
    @staticmethod
    def extract_text(file_type: str, file_content: bytes) -> str:
        """Extract text based on file type."""
        extractors = {
            'pdf': DocumentProcessor.extract_text_from_pdf,
            'txt': DocumentProcessor.extract_text_from_txt,
            'md': DocumentProcessor.extract_text_from_md,
            'doc': DocumentProcessor.extract_text_from_docx,
            'docx': DocumentProcessor.extract_text_from_docx,
        }
        
        extractor = extractors.get(file_type)
        if not extractor:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return extractor(file_content)


class ChunkProcessor:
    """Handles text chunking and embedding generation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @staticmethod
    def _token_length(text: str) -> int:
        """Calculate token length of text."""
        return len(encoding.encode(text))
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        chunks = []
        text_chunks = self.text_splitter.split_text(text)
        
        for idx, chunk_text in enumerate(text_chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                "chunk_total": len(text_chunks)
            }
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "token_count": self._token_length(chunk_text)
            })
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            return embeddings_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


@celery.task(bind=True, name='ingest_document', max_retries=3)
def ingest_document(self: Task, document_id: str, user_id: str) -> Dict[str, Any]:
    """
    Celery task to ingest a document: extract text, chunk it, and generate embeddings.
    
    Args:
        document_id: UUID of the document to process
        user_id: UUID of the user who owns the document
    
    Returns:
        Dict with ingestion results
    """
    db = next(get_db_sync())
    
    try:
        # Get document from database
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        # Update status to processing
        document.ingestion_status = IngestionStatus.PROCESSING
        db.commit()
        
        logger.info(f"Starting ingestion for document {document_id}: {document.filename}")
        
        # Download file from MinIO
        file_content = minio_service.download_file(
            object_name=document.minio_object_name,
            bucket_name=document.bucket_name
        )
        
        if not file_content:
            raise ValueError("Failed to download file from storage")
        
        # Extract text
        text = DocumentProcessor.extract_text(document.file_type, file_content)
        
        if not text or len(text.strip()) == 0:
            raise ValueError("No text content extracted from document")
        
        logger.info(f"Extracted {len(text)} characters from {document.filename}")
        
        # Create chunks
        chunk_processor = ChunkProcessor()
        chunks_data = chunk_processor.create_chunks(
            text=text,
            metadata={
                "filename": document.filename,
                "file_type": document.file_type,
                "document_id": str(document.id)
            }
        )
        
        logger.info(f"Created {len(chunks_data)} chunks from {document.filename}")
        
        # Delete existing chunks for this document (in case of re-ingestion)
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).delete()
        
        # Generate embeddings in batches
        batch_size = 10
        all_chunks = []
        
        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            
            # Generate embeddings
            embeddings = chunk_processor.generate_embeddings(texts)
            
            # Create chunk records
            for chunk_data, embedding in zip(batch, embeddings):
                chunk = DocumentChunk(
                    document_id=document.id,
                    user_id=document.user_id,
                    chunk_index=chunk_data["metadata"]["chunk_index"],
                    chunk_text=chunk_data["text"],
                    embedding=embedding,
                    chunk_metadata=chunk_data["metadata"],
                    token_count=chunk_data["token_count"]
                )
                all_chunks.append(chunk)
        
        # Bulk insert all chunks
        db.bulk_save_objects(all_chunks)
        
        # Update document status
        document.ingestion_status = IngestionStatus.COMPLETED
        document.ingestion_error = None
        db.commit()
        
        logger.info(f"Successfully ingested document {document_id} with {len(all_chunks)} chunks")
        
        return {
            "document_id": str(document_id),
            "status": "completed",
            "chunks_created": len(all_chunks),
            "total_tokens": sum(chunk.token_count for chunk in all_chunks)
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task time limit exceeded for document {document_id}")
        document.ingestion_status = IngestionStatus.FAILED
        document.ingestion_error = "Processing time limit exceeded"
        db.commit()
        raise
        
    except Exception as e:
        logger.error(f"Error ingesting document {document_id}: {str(e)}")
        
        # Update document status
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.ingestion_status = IngestionStatus.FAILED
                document.ingestion_error = str(e)
                db.commit()
        except:
            pass
        
        # Retry the task
        raise self.retry(exc=e, countdown=60)
    
    finally:
        db.close()