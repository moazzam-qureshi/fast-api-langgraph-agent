from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    thread_id: str
    message: str
    use_rag: Optional[bool] = Field(default=True, description="Whether to use RAG for this query")
    rag_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    rag_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")