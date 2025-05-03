from dataclasses import dataclass
from typing import List, Any, Optional
from embeddings import Embedder
from vector_store import VectorStore

import logging
from lang_model import LangModel, OllamaLangModel
from datetime import datetime


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
    def __init__(self, embedder=None, vector_store=None, lang_model: LangModel = None):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.lang_model = lang_model or OllamaLangModel()

    def query(self, query_string, n_results=10, prompt_template=None) -> QueryResult:
        """
        Retrieve top matching chunks and use LangModel to answer the query using those chunks as context.
        Returns a QueryResult with the answer and the context used.
        """
        context = self._find_similar_context(query_string, max_results=n_results)
        prompt = self._build_prompt(query_string, context, prompt_template)
        answer = self.lang_model.generate(prompt)
        return QueryResult(answer=answer, context=context)

    def _build_prompt(
        self,
        query_string: str,
        similar_context: List[ContextChunk],
        prompt_template: str = None,
    ) -> str:
        context_texts = [
            self.ensure_str(chunk.text) for chunk in similar_context if chunk.text
        ]
        context_texts.append(self._current_date_context())
        context = "\n".join(context_texts)
        if not prompt_template:
            prompt_template = (
                "Answer the following question. The provided context includes the daily notes from the user.\n"
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
        logger = logging.getLogger(__name__)
        logger.debug(f"ChromaDB raw results: {results}")

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
            f"Found {len(similar_context)} similar context chunks: {similar_context}"
        )
        return similar_context

    @staticmethod
    def _current_date_context():
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today's date is {current_date}\n"

    # Compose context string, ensuring all items are strings
    @staticmethod
    def ensure_str(x):
        if isinstance(x, list):
            return "\n".join(str(i) for i in x)
        return str(x)
