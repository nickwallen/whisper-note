import requests
import os
import json
from typing import Optional
from lang_model import LangModel


OLLAMA_URL_ENV = "OLLAMA_URL"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"


class OllamaLangModel(LangModel):
    def __init__(self, url: Optional[str] = None, model: Optional[str] = None):
        self.url = url or os.environ.get(OLLAMA_URL_ENV, "http://localhost:11434")
        self.model = model or os.environ.get(OLLAMA_MODEL_ENV, "llama2")

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=60,
                stream=True,
            )
            response.raise_for_status()
            answer = ""
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        answer += data["message"]["content"]
                except json.JSONDecodeError:
                    continue  # skip malformed lines
            return answer.strip()
        except requests.exceptions.HTTPError as e:
            return f"Ollama API responded with {response.status_code}: {response.text}: {e}"
