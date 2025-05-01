import os
from typing import List, Optional
import re
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileMetadata:
    file_name: str
    file_size: int
    modified_at: float
    created_at: float


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
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            f"Initialized Chunker with chunk_size={chunk_size}, overlap={overlap}, split_on={split_on}"
        )

    def chunk_file(self, file_path: str) -> List[str]:
        """
        Chunk a file into smaller pieces.
        Returns a list of strings.
        """
        try:
            metadata = self._extract_metadata(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return self._chunk_text(text, metadata)
        except UnicodeDecodeError as e:
            raise ValueError(f"Could not decode file {file_path}: {e}") from e

    def _chunk_text(
        self, text: str, metadata: Optional[FileMetadata] = None
    ) -> List[str]:
        """
        Chunk text into smaller pieces based on patterns in the text (e.g., paragraphs).
        Returns a list of strings.
        """
        if self.split_on:
            # Split on regex pattern (e.g., paragraphs)
            parts = re.split(self.split_on, text)
            chunks = []
            for part in parts:
                chunks.extend(self._chunk_by_size(part, metadata))
            return [c for c in chunks if c.strip()]

        return self._chunk_by_size(text, metadata)

    def _chunk_by_size(
        self, text: str, metadata: Optional[FileMetadata] = None
    ) -> List[str]:
        """
        Chunks text into smaller pieces based on the specified chunk size and overlap.
        Returns a list of strings.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            text_segment = text[start:end]
            if text_segment.strip():
                chunk = self._create_chunk(text_segment, metadata)
                chunks.append(chunk)
            start += (
                self.chunk_size - self.overlap
                if self.chunk_size > self.overlap
                else self.chunk_size
            )
        return chunks

    def _create_chunk(self, text_segment: str, metadata: Optional[FileMetadata]) -> str:
        """
        Format the chunk with metadata if provided, and log the chunk.
        """
        if metadata:
            created_at = format_date(metadata.created_at)
            modified_at = format_date(metadata.modified_at)
            chunk = f"User note: title '{metadata.file_name}', created at '{created_at}', last modified at '{modified_at}': {text_segment}"
        else:
            chunk = text_segment
        self.logger.debug(f"Created chunk: {chunk}")
        return chunk

    def _extract_metadata(self, file_path: str) -> FileMetadata:
        return FileMetadata(
            file_name=os.path.basename(file_path),
            file_size=os.path.getsize(file_path),
            modified_at=os.path.getmtime(file_path),
            created_at=os.path.getctime(file_path),
        )


def format_date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%A, %B %d, %Y")
