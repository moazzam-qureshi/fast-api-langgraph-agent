"""
Simple script to create the threads, users, and documents tables.
Run this once to set up the tables in your PostgreSQL database.

Usage: python create_threads_table.py
"""

from sqlalchemy import create_engine, text
from models import Base, Thread, User, Document  # Import models to register them
from database import DATABASE_URL

if __name__ == "__main__":
    print("Creating database tables...")
    engine = create_engine(DATABASE_URL)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Verify tables were created
    with engine.connect() as conn:
        # Check threads table
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'threads')"
        ))
        threads_exists = result.scalar()
        
        # Check users table
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users')"
        ))
        users_exists = result.scalar()
        
        if threads_exists:
            print("✓ Threads table created successfully!")
        else:
            print("✗ Failed to create threads table")
            
        if users_exists:
            print("✓ Users table created successfully!")
        else:
            print("✗ Failed to create users table")
            
        # Check documents table
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'documents')"
        ))
        documents_exists = result.scalar()
        
        if documents_exists:
            print("✓ Documents table created successfully!")
        else:
            print("✗ Failed to create documents table")
    
    engine.dispose()