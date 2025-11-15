"""Pydantic schemas for thread-related requests and responses."""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class ThreadCreate(BaseModel):
    """Schema for creating a thread."""
    title: Optional[str] = Field(default="New Chat", max_length=255)
    metadata: Optional[Dict[str, Any]] = None


class ThreadUpdate(BaseModel):
    """Schema for updating a thread."""
    title: Optional[str] = Field(default=None, max_length=255)
    metadata: Optional[Dict[str, Any]] = None


class ThreadResponse(BaseModel):
    """Schema for thread responses."""
    id: UUID
    user_id: str
    title: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)