import asyncio
import os
from dotenv import load_dotenv

# Mocking the provider for the test
os.environ["LLM_PROVIDER"] = "nvidia"

from self_improving_rag.generation.llm import generate_response

async def main():
    print("Testing NVIDIA integration...")
    try:
        from self_improving_rag.core.config import NVIDIA_API_KEY, LLM_MODEL
        print(f"Using model: {LLM_MODEL}")
        if not NVIDIA_API_KEY:
            print("Error: NVIDIA_API_KEY is not set in .env")
            return
            
        print("Sending test prompt to NVIDIA...")
        response = await generate_response("Hello, are you working via NVIDIA endpoints?")
        print(f"\nNVIDIA Response: {response}")
    except Exception as e:
        print(f"\nGeneration failed: {e}")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
