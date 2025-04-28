import requests
from dataclasses import dataclass
from typing import List, Any, Optional
from embeddings import Embedder
from vector_store import VectorStore
import json
import logging


@dataclass
class ContextChunk:
    id: str
    text: Optional[str]
    metadata: Optional[Any]
    distance: Optional[float]


@dataclass
class QueryResult:
    answer: str
    context: List[ContextChunk]


class QueryEngine:
    def __init__(
        self,
        embedder=None,
        vector_store=None,
        ollama_url="http://localhost:11434",
        ollama_model="llama2",
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def query(self, query_string, n_results=5, prompt_template=None) -> QueryResult:
        """
        Retrieve top matching chunks and use Ollama to answer the query using those chunks as context.
        Returns a QueryResult with the answer and the context used.
        """
        context = self._find_similar_context(query_string, max_results=n_results)
        prompt = self._build_prompt(query_string, context, prompt_template)
        answer = self.query_lang_model(prompt)
        return QueryResult(answer=answer, context=context)

    def query_lang_model(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
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
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        answer += data["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                answer = "Ollama API endpoint not found. Is the Ollama server running and is the model pulled?"
            else:
                answer = f"Ollama API error: {e}"
        except Exception as e:
            answer = f"Error communicating with Ollama: {e}"
        return answer.strip()

    def _build_prompt(
        self,
        query_string: str,
        similar_context: List[ContextChunk],
        prompt_template: str = None,
    ) -> str:
        context_texts = [
            self.ensure_str(chunk.text) for chunk in similar_context if chunk.text
        ]
        context = "\n".join(context_texts)
        if not prompt_template:
            prompt_template = (
                "Answer the following question using only the provided context.\n"
                "If the answer cannot be found in the context, say you don't know.\n\n"
                "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
        prompt = prompt_template.format(context=context, query=query_string)

        logger = logging.getLogger(__name__)
        logger.debug(f"Built prompt: {prompt}")
        return prompt

    def _find_similar_context(
        self, query: str, max_results: int = 10
    ) -> List[ContextChunk]:
        query_embedding = self.embedder.embed([query])[0]
        results = self.vector_store.query(query_embedding, n_results=max_results)
        similar_context = []
        for i in range(len(results["ids"])):
            chunk = ContextChunk(
                id=results["ids"][i],
                text=results["documents"][i] if "documents" in results else None,
                metadata=results["metadatas"][i] if "metadatas" in results else None,
                distance=results["distances"][i] if "distances" in results else None,
            )
            similar_context.append(chunk)

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Found {len(similar_context)} similar context chunks for query: {query}"
        )
        return similar_context

    # Compose context string, ensuring all items are strings
    @staticmethod
    def ensure_str(x):
        if isinstance(x, list):
            return "\n".join(str(i) for i in x)
        return str(x)
