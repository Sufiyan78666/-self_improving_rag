"""Debug script for pipeline storage issues."""
import asyncio, sys
sys.path.insert(0, '.')
from self_improving_rag.storage.db import get_connection

async def debug_db():
    try:
        conn = await get_connection()
        print("SESSIONS:")
        async with conn.execute("SELECT id, query, created_at FROM rag_sessions") as cursor:
            async for row in cursor:
                print(dict(row))
        
        print("\nRETRIEVALS:")
        async with conn.execute("SELECT id, session_id, chunk_id, rank_shown FROM rag_retrievals") as cursor:
            async for row in cursor:
                print(dict(row))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_db())
