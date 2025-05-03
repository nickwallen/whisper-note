import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Metadata:
    file: str
    file_hash: str = ""
    # Add other fields as needed


class VectorStore:
    def __init__(
        self,
        collection_name: str = "notes",
        chroma_client: Optional[chromadb.ClientAPI] = None,
    ):
        self.client = chroma_client or chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(collection_name)

    def get_all_metadata(self) -> List[Metadata]:
        """
        Return all metadata objects for the collection as a list of Metadata instances.
        Filters out any keys not present in the Metadata dataclass.
        """
        from dataclasses import fields

        results = self.collection.get()
        metadatas = results.get("metadatas", [])
        metadata_fields = {f.name for f in fields(Metadata)}
        filtered = []
        for md in metadatas:
            if md:
                filtered.append(
                    Metadata(**{k: v for k, v in md.items() if k in metadata_fields})
                )
        return filtered

    def delete_by_file_path(self, rel_path: str):
        """
        Delete all vectors whose metadata['file'] matches rel_path.
        """
        self.collection.delete(where={"file": rel_path})

    def is_file_hash_indexed(self, rel_path: str, file_hash: str) -> bool:
        """
        Return True if any vector with metadata['file'] == rel_path and metadata['file_hash'] == file_hash exists.
        """
        results = self.collection.get(
            where={"$and": [{"file": rel_path}, {"file_hash": file_hash}]}
        )
        return len(results["ids"]) > 0

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Metadata]] = None,
    ):
        """
        Add embeddings to the vector store.
        ids: List of unique string IDs
        embeddings: List of embedding vectors (same length as ids)
        documents: List of chunk texts (same length as ids)
        metadatas: List of metadata dicts (same length as ids, optional)
        """
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(
        self, embedding: List[float], n_results: int = 10
    ) -> chromadb.QueryResult:
        """
        Query the vector store for the most similar embeddings.
        embedding: The query embedding vector
        n_results: Number of results to return
        """
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
