import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, collection_name: str = "notes", persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(collection_name)

    def delete_by_file_path(self, rel_path: str):
        """
        Delete all vectors whose metadata['file'] matches rel_path.
        """
        self.collection.delete(where={"file": rel_path})

    def is_file_hash_indexed(self, rel_path: str, file_hash: str) -> bool:
        """
        Return True if any vector with metadata['file'] == rel_path and metadata['file_hash'] == file_hash exists.
        """
        results = self.collection.get(where={"$and": [{"file": rel_path}, {"file_hash": file_hash}]})
        return len(results["ids"]) > 0

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        """
        Add embeddings to the vector store.
        ids: List of unique string IDs
        embeddings: List of embedding vectors (same length as ids)
        metadatas: List of metadata dicts (same length as ids, optional)
        """
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding: List[float], n_results: int = 3):
        """
        Query the vector store for the most similar embeddings.
        embedding: The query embedding vector
        n_results: Number of results to return
        """
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)

# Example usage:
# store = VectorStore()
# store.add(ids=[...], embeddings=[...], metadatas=[...])
# results = store.query(embedding=[...], n_results=3)
