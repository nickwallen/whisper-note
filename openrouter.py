import logging
import os
import requests
from typing import Optional
from lang_model import LangModel

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLangModel(LangModel):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV)
        self.model = model or os.environ.get(
            OPENROUTER_MODEL_ENV, "openai/gpt-3.5-turbo"
        )
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be set in the environment variable 'OPENROUTER_API_KEY' or passed to the constructor."
            )
        logging.getLogger(__name__).debug(
            f"Initialized OpenRouterLangModel with model: {self.model}"
        )

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=data, timeout=60
            )
            response.raise_for_status()
            result = response.json()
            # OpenRouter returns OpenAI-compatible format
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            return f"OpenRouter API responded with {response.status_code}: {response.text}: {e}"
        except Exception as e:
            return f"OpenRouter API error: {e}"
