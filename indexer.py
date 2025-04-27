import os
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from embeddings import Embedder
from chunker import Chunker
from vector_store import VectorStore

@dataclass
class IndexerMetrics:
    file_count: int
    chunk_count: int
    files_indexed: List[str]
    chunks_per_file: Dict[str, int]
    empty_files: List[str] = field(default_factory=list)
    extensions_indexed: Set[str] = field(default_factory=set)

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
        files_indexed = []
        chunks_per_file = {}
        empty_files = []
        extensions_indexed = set()
        for file_path in files:
            rel_path = os.path.relpath(file_path, directory)
            ext = os.path.splitext(file_path)[1].lower()
            if ext:
                extensions_indexed.add(ext)
            chunks = self.chunker.chunk_file(file_path)
            embeddings = self.embedder.embed(chunks) if chunks else []
            ids = []
            metadatas = []
            for i, chunk in enumerate(chunks):
                ids.append(f"{rel_path}::chunk{i}")
                metadatas.append({"file": rel_path, "chunk_index": i, "text": chunk})
            if ids:
                self.vector_store.add(ids, embeddings, metadatas)
                files_indexed.append(rel_path)
                chunks_per_file[rel_path] = len(ids)
                chunk_count += len(ids)
            else:
                empty_files.append(rel_path)
        return IndexerMetrics(
            file_count=len(files),
            chunk_count=chunk_count,
            files_indexed=files_indexed,
            chunks_per_file=chunks_per_file,
            empty_files=empty_files,
            extensions_indexed=extensions_indexed,
        )

    def _find_files(self, directory: str, file_extensions: Optional[List[str]] = None) -> List[str]:
        result = []
        for root, _, files in os.walk(directory):
            for f in files:
                if not file_extensions or any(f.lower().endswith(ext) for ext in file_extensions):
                    result.append(os.path.join(root, f))
        return result
