import os
import logging
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from embeddings import Embedder
from chunker import Chunker
from vector_store import Metadata, VectorStore
import hashlib


@dataclass
class IndexerMetrics:
    file_count: int
    chunk_count: int
    failed_files: List[dict] = field(default_factory=list)

    def merge(self, other: "IndexerMetrics") -> "IndexerMetrics":
        return IndexerMetrics(
            file_count=self.file_count + other.file_count,
            chunk_count=self.chunk_count + other.chunk_count,
            failed_files=self.failed_files + other.failed_files,
        )


class Indexer:
    """
    Responsible for indexing all files in a directory.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder = embedder or Embedder()
        self.chunker = chunker or Chunker(split_on=None)
        self.vector_store = vector_store or VectorStore()

    def index_dir(
        self, dir: str, file_exts: Optional[List[str]] = None
    ) -> IndexerMetrics:
        """
        Index all files in a directory (recursively).
        dir: Directory to index
        file_exts: Optional list of file extensions to include (e.g., [".txt", ".md"])
        Returns: IndexerMetrics dataclass with indexing metrics.
        """
        logging.getLogger(__name__).debug(
            f"Indexing directory: {dir}, extensions: {file_exts}"
        )

        metrics = IndexerMetrics(file_count=0, chunk_count=0, failed_files=[])
        files = self._find_files(dir, file_exts)
        for file_path in files:
            try:
                file_metrics = self.index_file(file_path)
                metrics = metrics.merge(file_metrics)
            except Exception as e:
                metrics.failed_files.append({"file": file_path, "error": str(e)})
                continue

        return metrics

    def index_file(self, file_path) -> IndexerMetrics:
        """
        Index a single file. Returns IndexerMetrics for this file.
        Skips indexing if the file hash is already present.
        """
        logging.getLogger(__name__).debug(f"Indexing file: {file_path}")

        try:
            # Check if file is already indexed
            file_hash = self._compute_file_hash(file_path)
            if self.vector_store.is_file_hash_indexed(file_path, file_hash):
                logging.getLogger(__name__).debug(f"File already indexed: {file_path}")
                return IndexerMetrics(file_count=0, chunk_count=0)

            # Index file
            self.vector_store.delete_by_file_path(file_path)
            chunks = self.chunker.chunk_file(file_path)
            if not chunks:
                return IndexerMetrics(file_count=1, chunk_count=0)

            # Create embeddings for each chunk
            embeddings = self.embedder.embed(chunks)
            ids, metadatas = [], []
            for i, chunk in enumerate(chunks):
                ids.append(f"{file_hash}::chunk{i}")
                metadatas.append(
                    Metadata(
                        file=file_path,
                        file_hash=file_hash,
                        chunk_index=i,
                        text=chunk,
                        modified_at=get_modified_at(file_path),
                        created_at=get_created_at(file_path),
                    )
                )
            self.vector_store.add(ids, embeddings, chunks, metadatas)
            return IndexerMetrics(file_count=1, chunk_count=len(ids))
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to index file: {file_path}, error: {str(e)}"
            )
            return IndexerMetrics(
                file_count=0,
                chunk_count=0,
                failed_files=[{"file": file_path, "error": str(e)}],
            )

    def _find_files(
        self, directory: str, file_extensions: Optional[List[str]] = None
    ) -> List[str]:
        """
        Find all files in a directory with optional file extensions filter.
        Returns: List of file paths.
        """
        result = []
        for root, _, files in os.walk(directory):
            for f in files:
                if not file_extensions or any(
                    f.lower().endswith(ext) for ext in file_extensions
                ):
                    result.append(os.path.join(root, f))
        return result

    def _compute_file_hash(self, path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


def get_modified_at(file_path: str) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(file_path))


def get_created_at(file_path: str) -> datetime:
    stat = os.stat(file_path)
    if hasattr(stat, "st_birthtime"):
        created = stat.st_birthtime
    else:
        created = stat.st_ctime
    return datetime.fromtimestamp(created)
