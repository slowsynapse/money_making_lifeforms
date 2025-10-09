"""
Simple LLM factory for trading evolution system.

This provides a simpler interface than the main agent LLM system,
specifically for cell analysis and pattern discovery.
"""

import os
from typing import Optional, Dict, Any
from .local_llm import OllamaClient


def get_llm_client():
    """
    Get the appropriate LLM client based on USE_LOCAL_LLM flag.

    Returns:
        OllamaClient if USE_LOCAL_LLM=true, otherwise None (will use Anthropic via main system)
    """
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

    if use_local:
        print("🤖 Using local LLM (Ollama)")
        client = OllamaClient()

        # Check if available
        if not client.is_available():
            print("⚠️  Local LLM not available, falling back to Anthropic API")
            return None

        return client
    else:
        print("🤖 Using Anthropic API")
        return None


async def analyze_cell_with_llm(
    cell_context: str,
    system_prompt: str = "You are an expert trading strategy analyst.",
    use_json: bool = True,
) -> Dict[str, Any] | str:
    """
    Analyze a trading cell using either local or cloud LLM.

    Args:
        cell_context: The context about the cell to analyze
        system_prompt: System prompt for the LLM
        use_json: Whether to expect JSON response

    Returns:
        Dict if use_json=True, str otherwise
    """
    client = get_llm_client()

    if client is not None:
        # Use local Ollama
        if use_json:
            return await client.generate_with_json(
                prompt=cell_context,
                system=system_prompt,
                temperature=0.3,
            )
        else:
            return await client.generate(
                prompt=cell_context,
                system=system_prompt,
                temperature=0.7,
            )
    else:
        # Use Anthropic API (via main agent system)
        from anthropic import AsyncAnthropic

        anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        messages = [
            {"role": "user", "content": cell_context}
        ]

        if system_prompt:
            # Anthropic puts system separately
            response = await anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0.3 if use_json else 0.7,
                system=system_prompt,
                messages=messages,
            )
        else:
            response = await anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0.3 if use_json else 0.7,
                messages=messages,
            )

        text = response.content[0].text

        if use_json:
            import json
            # Try to extract JSON from response
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            return json.loads(text)
        else:
            return text
