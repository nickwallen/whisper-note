import pytest
from query_engine import QueryEngine

class DummyEmbedder:
    def embed(self, texts):
        # Return a vector of [len(text)] for each chunk (deterministic, fast)
        return [[float(len(t))] for t in texts]

class DummyVectorStore:
    def __init__(self):
        self.queries = []
    def query(self, embedding, n_results=5):
        # Return dummy results based on embedding length
        # For testing, just echo the embedding and n_results
        return {
            "ids": [f"id_{i}" for i in range(n_results)],
            "documents": [f"doc_{i}" for i in range(n_results)],
            "metadatas": [{"meta": i} for i in range(n_results)],
            "distances": [float(i) for i in range(n_results)],
        }

def test_query_engine_basic():
    engine = QueryEngine(embedder=DummyEmbedder(), vector_store=DummyVectorStore())
    results = engine.query("test", n_results=3)
    assert isinstance(results, dict)
    assert "context" in results
    assert isinstance(results["context"], list)
    assert len(results["context"]) == 3
    for i, item in enumerate(results["context"]):
        assert item["id"] == f"id_{i}"
        assert item["text"] == f"doc_{i}"
        assert item["metadata"] == {"meta": i}
        assert item["distance"] == float(i)
    assert "answer" in results

def test_query_engine_empty():
    class EmptyVectorStore(DummyVectorStore):
        def query(self, embedding, n_results=5):
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}
    engine = QueryEngine(embedder=DummyEmbedder(), vector_store=EmptyVectorStore())
    results = engine.query("", n_results=5)
    assert isinstance(results, dict)
    assert "context" in results
    assert results["context"] == []
    assert "answer" in results

def test_query_engine_handles_missing_fields():
    class PartialVectorStore(DummyVectorStore):
        def query(self, embedding, n_results=2):
            return {"ids": ["a", "b"]}  # missing other fields
    engine = QueryEngine(embedder=DummyEmbedder(), vector_store=PartialVectorStore())
    results = engine.query("foo", n_results=2)
    assert isinstance(results, dict)
    assert "context" in results
    context = results["context"]
    assert all(item["text"] is None for item in context)
    assert "answer" in results
    assert all(item["metadata"] is None for item in context)
    assert all(item["distance"] is None for item in context)
