from embeddings import Embedder
from vector_store import VectorStore

class QueryEngine:
    def __init__(self, embedder=None, vector_store=None):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def query(self, query_string, n_results=5):
        query_embedding = self.embedder.embed([query_string])[0]
        results = self.vector_store.query(query_embedding, n_results=n_results)
        formatted = []
        for i in range(len(results["ids"])):
            item = {
                "id": results["ids"][i],
                "text": results["documents"][i] if "documents" in results else None,
                "metadata": results["metadatas"][i] if "metadatas" in results else None,
                "distance": results["distances"][i] if "distances" in results else None,
            }
            formatted.append(item)
        return formatted
