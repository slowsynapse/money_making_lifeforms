"""
Local LLM client using Ollama for free, private LLM inference.
"""

import os
import requests
import json
from typing import Optional, Dict, Any


class OllamaClient:
    """
    Client for interacting with Ollama local LLM server.

    Ollama provides a compatible API for running models locally without
    external API costs or privacy concerns.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model name (default: from OLLAMA_MODEL env var)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma2:27b-instruct-q4_K_M")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate a completion from the local LLM.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for Ollama API

        Returns:
            Generated text
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Call Ollama API
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            timeout=300  # 5 minute timeout for large models
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["message"]["content"]

    async def generate_with_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the local LLM.

        Args:
            prompt: User prompt (should request JSON output)
            system: System prompt (optional)
            temperature: Lower temperature for more consistent JSON
            **kwargs: Additional parameters

        Returns:
            Parsed JSON dict
        """
        # Add JSON format instruction if not already present
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nRespond with valid JSON only."

        if system and "json" not in system.lower():
            system = f"{system}\n\nYou must respond with valid JSON only."

        # Generate response
        text = await self.generate(prompt, system=system, temperature=temperature, **kwargs)

        # Try to extract JSON from response
        # Sometimes LLMs wrap JSON in markdown code blocks
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response. "
                f"Error: {e}\n"
                f"Response: {text[:500]}..."
            )

    def is_available(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            True if server is reachable and model is available
        """
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if our model is available
            tags = response.json()
            models = [m["name"] for m in tags.get("models", [])]

            # Check exact match or base model name match
            # (Ollama sometimes uses full model names like "gemma2:27b-instruct-q4_K_M")
            model_base = self.model.split(":")[0] if ":" in self.model else self.model

            for available_model in models:
                available_base = available_model.split(":")[0] if ":" in available_model else available_model
                if self.model == available_model or model_base == available_base:
                    return True

            print(f"⚠️  Ollama server is running, but model '{self.model}' is not available.")
            print(f"Available models: {', '.join(models)}")
            print(f"Run: ollama pull {self.model}")
            return False

        except (requests.ConnectionError, requests.Timeout):
            print(f"⚠️  Cannot connect to Ollama at {self.base_url}")
            print(f"Is Ollama running? Start it with: ollama serve")
            return False

    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get usage information (local LLM usage is free).

        Returns:
            Dict with cost=$0.0 and local=True
        """
        return {
            "cost": 0.0,
            "tokens": 0,  # Ollama doesn't report token counts
            "local": True,
            "model": self.model,
        }
