import os
from typing import List, Optional
from dataclasses import dataclass, field
from embeddings import Embedder
from chunker import Chunker
from vector_store import VectorStore
import hashlib


@dataclass
class IndexerMetrics:
    file_count: int
    chunk_count: int
    failed_files: List[dict] = field(
        default_factory=list
    )  # List of {"file": ..., "error": ...}


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
        self.chunker = chunker or Chunker()
        self.vector_store = vector_store or VectorStore()

    def index_directory(
        self, directory: str, file_extensions: Optional[List[str]] = None
    ) -> IndexerMetrics:
        """
        Index all files in a directory (recursively).
        file_extensions: Optional list of file extensions to include (e.g., [".txt", ".md"])
        Returns: IndexerMetrics dataclass instance with various metrics.
        """
        chunk_count = 0
        failed_files = []

        files = self._find_files(directory, file_extensions)
        for file_path in files:
            rel_path = os.path.relpath(file_path, directory)
            try:
                file_hash = self._compute_file_hash(file_path)
                if self.vector_store.is_file_hash_indexed(rel_path, file_hash):
                    continue # No need to re-index the same file contents
                self.vector_store.delete_by_file_path(rel_path)
                chunks = self.chunker.chunk_file(file_path)
                embeddings = self.embedder.embed(chunks) if chunks else []
                ids = []
                metadatas = []
                mod_time_iso = self._get_mod_time(file_path)
                for i, chunk in enumerate(chunks):
                    ids.append(f"{file_hash}::chunk{i}")
                    metadatas.append(
                        self._create_metadata(
                            rel_path, file_hash, i, chunk, mod_time_iso
                        )
                    )
                if ids:
                    self.vector_store.add(ids, embeddings, metadatas)
                    chunk_count += len(ids)
            except Exception as e:
                failed_files.append({"file": rel_path, "error": str(e)})
                continue

        return IndexerMetrics(
            file_count=len(files),
            chunk_count=chunk_count,
            failed_files=failed_files,
        )

    def _create_metadata(self, rel_path, file_hash, chunk_index, chunk, mod_time_iso):
        return {
            "file": rel_path,
            "file_hash": file_hash,
            "chunk_index": chunk_index,
            "text": chunk,
            "modification_time": mod_time_iso,
        }

    def _get_mod_time(self, file_path):
        mod_time = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mod_time).isoformat()

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
