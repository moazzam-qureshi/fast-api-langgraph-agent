"""Thread service for CRUD operations."""
from typing import Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import desc
import json

from models.threads import Thread
from schemas.threads import ThreadCreate, ThreadUpdate


class ThreadService:
    """Service class for thread CRUD operations."""
    
    @staticmethod
    def create_thread(db: Session, user_id: str, thread_data: ThreadCreate) -> Thread:
        """Create a new thread for a user."""
        db_thread = Thread(
            user_id=user_id,
            title=thread_data.title,
            thread_metadata=json.dumps(thread_data.metadata) if thread_data.metadata else None
        )
        
        db.add(db_thread)
        db.commit()
        db.refresh(db_thread)
        
        return db_thread
    
    @staticmethod
    def get_thread(db: Session, thread_id: UUID, user_id: Optional[str] = None) -> Optional[Thread]:
        """Retrieve a thread by ID."""
        query = db.query(Thread).filter(Thread.id == thread_id)
        
        if user_id:
            query = query.filter(Thread.user_id == user_id)
            
        return query.first()
    
    @staticmethod
    def get_user_threads(db: Session, user_id: str, skip: int = 0, limit: int = 20) -> List[Thread]:
        """Retrieve all threads for a specific user."""
        return db.query(Thread).filter(
            Thread.user_id == user_id
        ).order_by(
            desc(Thread.updated_at)
        ).offset(skip).limit(limit).all()
    
    @staticmethod
    def update_thread(db: Session, thread_id: UUID, user_id: str, thread_update: ThreadUpdate) -> Optional[Thread]:
        """Update a thread's information."""
        thread = db.query(Thread).filter(
            Thread.id == thread_id,
            Thread.user_id == user_id
        ).first()
        
        if not thread:
            return None
            
        if thread_update.title is not None:
            thread.title = thread_update.title
            
        if thread_update.metadata is not None:
            thread.thread_metadata = json.dumps(thread_update.metadata)
            
        db.commit()
        db.refresh(thread)
        
        return thread
    
    @staticmethod
    def delete_thread(db: Session, thread_id: UUID, user_id: str) -> bool:
        """Delete a thread."""
        thread = db.query(Thread).filter(
            Thread.id == thread_id,
            Thread.user_id == user_id
        ).first()
        
        if not thread:
            return False
            
        db.delete(thread)
        db.commit()
        
        return True