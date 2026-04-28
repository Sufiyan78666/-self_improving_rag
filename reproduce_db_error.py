import asyncio
import uuid
from self_improving_rag.storage.queries import insert_session
from self_improving_rag.storage.db import init_db

async def test():
    await init_db()
    sid = str(uuid.uuid4())
    print(f"Testing session ID: {sid}")
    
    print("First insertion...")
    await insert_session("test query", session_id=sid)
    
    print("Second insertion (should be ignored)...")
    try:
        await insert_session("test query", session_id=sid)
        print("Success: Second insertion was ignored as expected.")
    except Exception as e:
        print(f"Error: Second insertion failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
