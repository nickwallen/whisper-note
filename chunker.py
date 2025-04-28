from typing import List, Optional
import re


class Chunker:
    """
    Creates chunks of text by splitting input text or files into smaller
    pieces based on the specified chunk size and overlap. First
    splits by the specified regex pattern (e.g., paragraphs) if provided.
    Then splits by the specified chunk size with optional overlap.
    """

    def __init__(
        self, chunk_size: int = 512, overlap: int = 0, split_on: Optional[str] = r"\n\n"
    ):
        """
        chunk_size: number of characters per chunk (default)
        overlap: number of characters to overlap between chunks
        split_on: optional regex pattern to split on (e.g., '\n\n' for paragraphs)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_on = split_on

    def chunk_file(self, file_path: str) -> List[str]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return self.chunk_text(text)
        except UnicodeDecodeError as e:
            raise Exception(f"UnicodeDecodeError in file {file_path}: {e}")

    def chunk_text(self, text: str) -> List[str]:
        if self.split_on:
            # Split on regex pattern (e.g., paragraphs)
            parts = re.split(self.split_on, text)
            chunks = []
            for part in parts:
                chunks.extend(self._chunk_by_size(part))
            return [c for c in chunks if c.strip()]
        else:
            return self._chunk_by_size(text)

    def _chunk_by_size(self, text: str) -> List[str]:
        # Sliding window chunking with optional overlap
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += (
                self.chunk_size - self.overlap
                if self.chunk_size > self.overlap
                else self.chunk_size
            )
        return chunks
