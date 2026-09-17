from app.db import engine, Base
from sqlalchemy import text
import app.models  # registers all models

def main():
    print("Testing connection to NeonDB...")
    with engine.connect() as conn:
        res = conn.execute(text("SELECT version();")).fetchone()
        print("Connected to:", res[0])
        
        # Test enabling pgvector
        print("Enabling pgvector extension if not exists...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        
        ext = conn.execute(text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")).fetchone()
        print("pgvector extension installed:", ext)
        
    print("\nEnsuring all tables (Base.metadata.create_all) exist...")
    Base.metadata.create_all(bind=engine)
    print("Base.metadata.create_all completed!")
    
    with engine.connect() as conn:
        tables = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")).fetchall()
        print("Tables in public schema:", [t[0] for t in tables])

if __name__ == "__main__":
    main()
