"""
llm.py - Groq LLM wrapper (fast LPU inference)
Uses the official `groq` Python SDK.
Supports streaming responses for real-time chat.
"""

import logging
from typing import Generator, List, Dict

from groq import Groq

from backend.config import config

logger = logging.getLogger(__name__)


class GroqLLM:
    """
    Wrapper for Groq API using the official groq SDK.
    Default model: llama-3.3-70b-versatile (excellent multilingual + math reasoning)
    """

    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL
        logger.info(f"✅ Groq LLM initialized | Model: {self.model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Send a chat request to Groq and yield streamed response tokens.

        Args:
            messages: List of {role, content} dicts
            stream: Whether to stream the response
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response

        Yields:
            Response text tokens as they arrive
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if stream:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
            else:
                yield response.choices[0].message.content

        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            yield f"\n\n[Error communicating with Groq API: {str(e)}]"

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming version — returns full response string."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# Singleton instance
_llm_instance = None


def get_llm() -> GroqLLM:
    """Returns a singleton GroqLLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GroqLLM()
    return _llm_instance
