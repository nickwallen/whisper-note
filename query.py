from dataclasses import dataclass
from typing import List, Any, Optional
from embeddings import Embedder
from time_range import TimeRangeExtractor
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
        self.time_range_extractor = TimeRangeExtractor(lang_model=self.lang_model)

    def query(self, query_string, max_results=10, prompt_template=None) -> QueryResult:
        """
        Retrieve top matching chunks and use LangModel to answer the query using those chunks as context.
        Returns a QueryResult with the answer and the context used.
        """
        time_range = self.time_range_extractor.extract(query_string)
        context = self._find_similar_context(
            query_string,
            max_results=max_results,
            start_time=time_range.start,
            end_time=time_range.end,
        )
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
        context = "\n\n".join(context_texts)
        if not prompt_template:
            prompt_template = (
                "Answer the following question using only the information in the "
                "context. Respond directly and do not reference the context by saying "
                "something like ' Based on the information provided in the context...'.\n"
                "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
        prompt = prompt_template.format(context=context, query=query_string)

        logging.getLogger(__name__).debug(f"Built prompt: {prompt}")
        return prompt

    def _find_similar_context(
        self,
        query: str,
        max_results: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ContextChunk]:
        """
        Retrieve top matching context chunks for a single query.
        NOTE: This implementation only supports single-query (one embedding at a time).
        If you want to support multi-query (batch queries), you must update this logic.
        """
        logging.getLogger(__name__).debug(
            f"Finding similar context for query: {query}, start_time: {start_time}, end_time: {end_time} max_results: {max_results}"
        )

        query_embedding = self.embedder.embed([query])[0]
        results = self.vector_store.query(
            query_embedding,
            max_results=max_results,
            start_time=start_time.timestamp() if start_time else None,
            end_time=end_time.timestamp() if end_time else None,
        )

        ids = self.get_first_list("ids", results)
        documents = self.get_first_list("documents", results)
        metadatas = self.get_first_list("metadatas", results)
        distances = self.get_first_list("distances", results)

        similar_context = []
        for i in range(len(ids)):
            chunk = ContextChunk(
                id=ids[i],
                text=documents[i] if i < len(documents) else None,
                metadata=metadatas[i] if i < len(metadatas) else None,
                distance=distances[i] if i < len(distances) else None,
            )
            similar_context.append(chunk)

        logging.getLogger(__name__).debug(
            f"Found {len(similar_context)} similar context chunks"
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

    @staticmethod
    def get_first_list(key, results):
        """
        Utility to extract the first list from ChromaDB results for a given key.
        Only supports single-query (not multi-query).
        """
        val = results.get(key, [])
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            return val[0]
        return val
