import os
from typing import List, Dict, Optional
from embeddings import Embedder
from chunker import Chunker
from vector_store import VectorStore

class Indexer:
    def __init__(self, 
                 embedder: Optional[Embedder] = None,
                 chunker: Optional[Chunker] = None,
                 vector_store: Optional[VectorStore] = None):
        self.embedder = embedder or Embedder()
        self.chunker = chunker or Chunker()
        self.vector_store = vector_store or VectorStore()

    def index_directory(self, directory: str, file_extensions: Optional[List[str]] = None):
        """
        Index all files in a directory (recursively).
        file_extensions: Optional list of file extensions to include (e.g., [".txt", ".md"])
        Returns: number of files indexed, number of chunks added
        """
        files = self._find_files(directory, file_extensions)
        chunk_id = 0
        for file_path in files:
            chunks = self.chunker.chunk_file(file_path)
            embeddings = self.embedder.embed(chunks) if chunks else []
            ids = []
            metadatas = []
            for i, chunk in enumerate(chunks):
                ids.append(f"{os.path.relpath(file_path, directory)}::chunk{i}")
                metadatas.append({"file": os.path.relpath(file_path, directory), "chunk_index": i, "text": chunk})
                chunk_id += 1
            if ids:
                self.vector_store.add(ids, embeddings, metadatas)
        return len(files), chunk_id

    def _find_files(self, directory: str, file_extensions: Optional[List[str]] = None) -> List[str]:
        result = []
        for root, _, files in os.walk(directory):
            for f in files:
                if not file_extensions or any(f.lower().endswith(ext) for ext in file_extensions):
                    result.append(os.path.join(root, f))
        return result
