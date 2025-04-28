import os
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from embeddings import Embedder
from chunker import Chunker
from vector_store import VectorStore
import hashlib

@dataclass
class IndexerMetrics:
    file_count: int
    chunk_count: int
    extensions_indexed: Set[str] = field(default_factory=set)
    failed_files: List[dict] = field(default_factory=list)  # List of {"file": ..., "error": ...}

class Indexer:
    def __init__(self, 
                 embedder: Optional[Embedder] = None,
                 chunker: Optional[Chunker] = None,
                 vector_store: Optional[VectorStore] = None):
        self.embedder = embedder or Embedder()
        self.chunker = chunker or Chunker()
        self.vector_store = vector_store or VectorStore()

    def index_directory(self, directory: str, file_extensions: Optional[List[str]] = None) -> 'IndexerMetrics':
        """
        Index all files in a directory (recursively).
        file_extensions: Optional list of file extensions to include (e.g., [".txt", ".md"])
        Returns: IndexerMetrics dataclass instance with various metrics.
        """
        files = self._find_files(directory, file_extensions)
        chunk_count = 0
        extensions_indexed = set()
        failed_files = []
        for file_path in files:
            rel_path = os.path.relpath(file_path, directory)
            ext = os.path.splitext(file_path)[1].lower()
            if ext:
                extensions_indexed.add(ext)
            # Compute file hash (SHA256)
            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
            except Exception as e:
                failed_files.append({"file": rel_path, "error": f"Hash error: {str(e)}"})
                continue
            # Skip if this file version is already indexed
            if self.vector_store.is_file_hash_indexed(rel_path, file_hash):
                continue
            # Clean up old vectors for this file path
            self.vector_store.delete_by_file_path(rel_path)
            try:
                chunks = self.chunker.chunk_file(file_path)
            except Exception as e:
                failed_files.append({"file": rel_path, "error": str(e)})
                continue
            embeddings = self.embedder.embed(chunks) if chunks else []
            ids = []
            metadatas = []
            from datetime import datetime
            mod_time = os.path.getmtime(file_path)
            mod_time_iso = datetime.fromtimestamp(mod_time).isoformat()
            for i, chunk in enumerate(chunks):
                ids.append(f"{file_hash}::chunk{i}")
                metadatas.append({
                    "file": rel_path,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "text": chunk,
                    "modification_time": mod_time_iso,
                })
            if ids:
                self.vector_store.add(ids, embeddings, metadatas)
                chunk_count += len(ids)
        return IndexerMetrics(
            file_count=len(files),
            chunk_count=chunk_count,
            extensions_indexed=extensions_indexed,
            failed_files=failed_files,
        )

    def _find_files(self, directory: str, file_extensions: Optional[List[str]] = None) -> List[str]:
        result = []
        for root, _, files in os.walk(directory):
            for f in files:
                if not file_extensions or any(f.lower().endswith(ext) for ext in file_extensions):
                    result.append(os.path.join(root, f))
        return result
