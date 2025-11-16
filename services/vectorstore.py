"""Vector store service for RAG retrieval using PGVector."""
import os
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy import text, and_
from sqlalchemy.orm import Session
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from pgvector.sqlalchemy import Vector

from models import DocumentChunk
from database import engine

logger = logging.getLogger(__name__)

# Initialize embeddings model (same as in ingestion)
embeddings_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)


class VectorStoreService:
    """Service for vector similarity search and RAG retrieval."""
    
    def __init__(self):
        self.connection_string = "postgresql://postgres:postgres@postgres:5432/postgres"
        self.collection_name = "document_chunks"
        self.embeddings = embeddings_model
        
    def initialize_vectorstore(self) -> PGVector:
        """Initialize PGVector store."""
        try:
            vectorstore = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                distance_strategy=DistanceStrategy.COSINE
            )
            return vectorstore
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def search_similar_chunks(
        self,
        query: str,
        user_id: UUID,
        k: int = 5,
        score_threshold: float = 0.7,
        filter_document_ids: Optional[List[UUID]] = None,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on query embedding.
        
        Args:
            query: Search query text
            user_id: User ID to filter results
            k: Number of results to return
            score_threshold: Minimum similarity score
            filter_document_ids: Optional list of document IDs to search within
            db: Database session
            
        Returns:
            List of similar chunks with metadata and scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build the base query
            base_query = db.query(
                DocumentChunk.id,
                DocumentChunk.chunk_text,
                DocumentChunk.chunk_metadata,
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.embedding.cosine_distance(query_embedding).label('distance')
            ).filter(
                DocumentChunk.user_id == user_id,
                DocumentChunk.embedding.isnot(None)  # Only chunks with embeddings
            )
            
            # Apply document filter if provided
            if filter_document_ids:
                base_query = base_query.filter(
                    DocumentChunk.document_id.in_(filter_document_ids)
                )
            
            # Order by similarity and limit results
            results = base_query.order_by('distance').limit(k).all()
            
            # Format results
            formatted_results = []
            for result in results:
                # Convert distance to similarity score (1 - distance for cosine)
                similarity_score = 1 - result.distance
                
                if similarity_score >= score_threshold:
                    formatted_results.append({
                        "chunk_id": str(result.id),
                        "document_id": str(result.document_id),
                        "chunk_text": result.chunk_text,
                        "chunk_index": result.chunk_index,
                        "metadata": result.chunk_metadata,
                        "similarity_score": similarity_score
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            raise
    
    def get_retriever_for_user(
        self,
        user_id: UUID,
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get a LangChain retriever filtered for a specific user.
        
        Args:
            user_id: User ID to filter documents
            search_kwargs: Additional search parameters (k, score_threshold, etc.)
            
        Returns:
            LangChain retriever instance
        """
        vectorstore = self.initialize_vectorstore()
        
        # Default search parameters
        default_kwargs = {
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.25
        }
        
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        # Create retriever with user filter
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs=default_kwargs,
            filter={"user_id": str(user_id)}
        )
        
        return retriever
    
    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context for LLM prompt.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            
            context_part = f"[Source {i}: {filename}]\n{chunk['chunk_text']}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    @staticmethod
    def create_vector_index(db: Session):
        """Create vector similarity index for better performance."""
        try:
            # Create an index on the embedding column for faster similarity search
            create_index_sql = """
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            db.execute(text(create_index_sql))
            db.commit()
            logger.info("Vector index created successfully")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            db.rollback()


# Global instance
vector_store_service = VectorStoreService()