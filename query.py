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

    PROMPT_TEMPLATE = (
        "Given the following context and question, generate a helpful response to the user's question.\n"
        "\n"
        "Instructions:\n"
        "- Only include work that the user personally completed, led, or contributed to.\n"
        "- Ignore information about other people unless it directly relates to the user's own work or deliverables.\n"
        "- Ignore notes that are procedural, instructional, or reference material (e.g., how-to guides, meeting agendas, copied documentation, or class notes). Do not include these as completed tasks.\n"
        "- If unsure whether an item is a completed task, omit it.\n"
        "- Begin your answer with a single, concise sentence that states the time period the summary covers (e.g., 'On Friday, May 1st, you...').\n"
        "- Include all relevant information as a concise, high-level bulleted list. **Do not include any additional paragraphs, commentary, or summaries outside of the bulleted list and the opening sentence.**\n"
        "- Each bullet should represent a completed task or high-level accomplishment. If you attended a class or training, summarize it as a single bullet (e.g., 'Attended Workspaces 101 class').\n"
        "- Be brief and avoid unnecessary details or repetition.\n"
        "- Do not reference the context or say things like 'Based on the information provided...'.\n"
        "- Today's date is {today}.\n"
        "\n"
        "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    def __init__(self, embedder=None, vector_store=None, lang_model: LangModel = None):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.lang_model = lang_model or OllamaLangModel()
        self.time_range_extractor = TimeRangeExtractor(lang_model=self.lang_model)

    def query(
        self, query_string, max_results=10, prompt_template=PROMPT_TEMPLATE
    ) -> QueryResult:
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
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> str:
        from datetime import datetime

        context_texts = [
            self.ensure_str(chunk.text) for chunk in similar_context if chunk.text
        ]
        context = "\n\n".join(context_texts)
        today = datetime.now().strftime("%A, %B %d, %Y")
        prompt = prompt_template.format(
            context=context, query=query_string, today=today
        )

        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Built prompt: {self._format_log_message(prompt)}")
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

    def _format_log_message(self, msg: str, max_length: int = 2000):
        """
        Remove newlines and limit the length of the log message for readability.
        """
        single_line = msg.replace("\n", " ").replace("\r", " ")
        return single_line[:max_length] + (
            "..." if len(single_line) > max_length else ""
        )
