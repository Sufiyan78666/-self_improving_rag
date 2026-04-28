"""Check latest sessions only."""
import asyncio, sys
sys.path.insert(0, '.')
from self_improving_rag.storage.db import get_connection

async def check():
    conn = await get_connection()
    async with conn.execute("SELECT id, query, created_at FROM rag_sessions ORDER BY created_at DESC LIMIT 5") as cursor:
        async for row in cursor:
            print(dict(row))

if __name__ == "__main__":
    asyncio.run(check())
