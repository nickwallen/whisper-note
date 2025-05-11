from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import json
import logging


@dataclass
class TimeRange:
    start: Optional[datetime]
    end: Optional[datetime]


class TimeRangeExtractor:
    """
    Uses an LLM to determine if a user query is time-sensitive and extracts start/end dates if applicable.
    """

    PROMPT_TEMPLATE = """
        Given the following user question, determine whether it has a time-sensitive component. If it does, return the start
        and end date that the question refers to in ISO 8601 format (YYYY-MM-DD). If the question is not time-sensitive,
        return null for both.

        IMPORTANT:
        - Output ONLY a single line of valid JSON.
        - Do NOT include any explanation, markdown, or extra text before or after the JSON.
        - Do NOT include phrases like 'Here's the output:' or code blocks.
        - Your response will be parsed by a computer. If you include anything other than the JSON object, it will cause an error.
        - Assume today's date is {today}.
        
        Output strictly in this JSON format:
        {{
            "start": "YYYY-MM-DD" or null,
            "end": "YYYY-MM-DD" or null
        }}

        For example, if the user's question is "What did I do yesterday?", return {{ "start": "{yesterday}", "end": "{yesterday}" }}.
        User question: "{query}"
        """

    def __init__(self, lang_model):
        self.lang_model = lang_model

    def extract(self, query: str) -> TimeRange:
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        prompt = self.PROMPT_TEMPLATE.format(
            today=today, yesterday=yesterday, query=query
        )
        response = self.lang_model.generate(prompt)
        try:
            data = json.loads(response)

            # Always use start of day for start date
            start = self._parse_date_str(data.get("start"))
            if start:
                start = start.replace(hour=0, minute=0, second=0, microsecond=0)

            # Always use end of day for end date
            end = self._parse_date_str(data.get("end"))
            if end:
                end = end.replace(hour=23, minute=59, second=59, microsecond=0)

            logging.getLogger(__name__).debug(
                f"Time range from '{query}' is start={start}, end={end}"
            )
            return TimeRange(
                start=start,
                end=end,
            )

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to extract time range from '{query}', error: {str(e)}, response: {response}"
            )
            return TimeRange(start=None, end=None)

    @staticmethod
    def _parse_date_str(date_str):
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Invalid date string: {date_str}, error: {str(e)}"
            )
        return None
