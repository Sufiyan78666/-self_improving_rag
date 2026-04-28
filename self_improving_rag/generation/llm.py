"""
LLM integration for the Self-Improving RAG system using the new Google GenAI SDK.

Handles authentication, client initialization, and generation logic.
"""

import logging
import os
import time
from typing import Optional

import requests
from google import genai
from google.genai import types
from self_improving_rag.core.config import LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL, NVIDIA_API_KEY
from self_improving_rag.core.exceptions import GenerationFailedError
from self_improving_rag.generation.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _create_gemini_client() -> genai.Client:
    """Create a new GenAI client instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY environment variable is not set.")
        return None

    try:
        # Avoid global state to prevent "Event loop is closed" errors in Streamlit
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise GenerationFailedError(f"Failed to initialize Gemini Client: {exc}") from exc


async def generate_response(prompt: str) -> str:
    """
    Send a prompt to the configured LLM provider and return the response text.
    """
    if LLM_PROVIDER == "ollama":
        return await _generate_ollama_response(prompt)
    elif LLM_PROVIDER == "nvidia":
        return await _generate_nvidia_response(prompt)
    else:
        return await _generate_gemini_response(prompt)


async def _generate_gemini_response(prompt: str) -> str:
    """
    Send a prompt to Gemini and return the generated response text.
    Uses a fresh client per call to ensure event loop compatibility.
    """
    client = _create_gemini_client()
    if client is None:
        raise GenerationFailedError(
            "Gemini library not initialized. Ensure GOOGLE_API_KEY is set in .env."
        )

    try:
        # The new SDK generate_content call
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
            )
        )

        if not response.text:
            raise GenerationFailedError("Gemini returned an empty response.")

        logger.info(f"Generated Gemini response of length {len(response.text)}")
        return response.text

    except Exception as exc:
        raise GenerationFailedError(f"Gemini generation failed: {exc}") from exc


async def _generate_ollama_response(prompt: str) -> str:
    """
    Send a prompt to a local Ollama instance and return the response text.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    # Constructing the full prompt with system instructions for Ollama
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "num_predict": 1024,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        text = data.get("response", "")
        if not text:
            raise GenerationFailedError("Ollama returned an empty response.")
            
        logger.info(f"Generated Ollama response of length {len(text)}")
        return text

    except requests.exceptions.RequestException as exc:
        raise GenerationFailedError(f"Ollama generation failed: {exc}") from exc
    except Exception as exc:
        raise GenerationFailedError(f"Unexpected error during Ollama generation: {exc}") from exc


async def _generate_nvidia_response(prompt: str) -> str:
    """
    Send a prompt to NVIDIA AI Foundation endpoints using LangChain's ChatNVIDIA.
    """
    start_time = time.time()
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        import asyncio
        from functools import partial
        
        if not NVIDIA_API_KEY:
            raise GenerationFailedError("NVIDIA_API_KEY not found in configuration.")

        # ChatNVIDIA initialization
        client = ChatNVIDIA(
            model=LLM_MODEL,
            api_key=NVIDIA_API_KEY,
            temperature=0.1,
            max_tokens=1024,
        )

        # We pass the system prompt as the first message and user prompt as the second
        messages = [
            ("system", SYSTEM_PROMPT),
            ("user", prompt),
        ]

        # Run synchronous invoke in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, partial(client.invoke, messages))
        
        if not response.content:
            raise GenerationFailedError("NVIDIA returned an empty response.")

        elapsed = time.time() - start_time
        logger.info(f"Generated NVIDIA response in {elapsed:.2f}s (length {len(response.content)})")
        return str(response.content)

    except ImportError:
        raise GenerationFailedError(
            "langchain-nvidia-ai-endpoints package not found. Run 'pip install langchain-nvidia-ai-endpoints'."
        )
    except Exception as exc:
        raise GenerationFailedError(f"NVIDIA generation failed: {exc}") from exc
