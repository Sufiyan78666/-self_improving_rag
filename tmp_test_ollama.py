import asyncio
import os
import requests
from dotenv import load_dotenv

# Mocking the provider for the test if not set
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["LLM_MODEL"] = "llama3" # Assuming llama3 is available

from self_improving_rag.generation.llm import generate_response

async def main():
    print("Testing Ollama integration...")
    try:
        # Check if Ollama is running first
        from self_improving_rag.core.config import OLLAMA_BASE_URL
        print(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")
        resp = requests.get(OLLAMA_BASE_URL)
        if resp.status_code == 200:
            print("Ollama is reachable!")
        else:
            print(f"Ollama returned status code {resp.status_code}")
    except Exception as e:
        print(f"Could not connect to Ollama: {e}")
        print("Please ensure Ollama is running (e.g., 'ollama serve') and 'llama3' is pulled.")
        return

    try:
        response = await generate_response("Hello, are you working?")
        print(f"\nOllama Response: {response}")
    except Exception as e:
        print(f"\nGeneration failed: {e}")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
