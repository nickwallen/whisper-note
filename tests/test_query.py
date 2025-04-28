import pytest
from query import QueryEngine, QueryResult, ContextChunk

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

def test_query_basic():
    engine = QueryEngine(embedder=DummyEmbedder(), vector_store=DummyVectorStore())
    actual = engine.query("test", n_results=3)
    expected = QueryResult(
        answer=actual.answer,  # Accept whatever answer the engine generates
        context=[
            ContextChunk(id="id_0", text="doc_0", metadata={"meta": 0}, distance=0.0),
            ContextChunk(id="id_1", text="doc_1", metadata={"meta": 1}, distance=1.0),
            ContextChunk(id="id_2", text="doc_2", metadata={"meta": 2}, distance=2.0),
        ]
    )
    assert actual == expected
    assert actual.answer

def test_query_empty():
    class EmptyVectorStore(DummyVectorStore):
        def query(self, embedding, n_results=5):
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}
    engine = QueryEngine(embedder=DummyEmbedder(), vector_store=EmptyVectorStore())
    actual = engine.query("", n_results=5)
    expected = QueryResult(
        answer=actual.answer,  # Accept whatever answer the engine generates
        context=[]
    )
    assert actual == expected
    assert actual.answer

def test_query_handles_missing_fields():
    class PartialVectorStore(DummyVectorStore):
        def query(self, embedding, n_results=2):
            return {"ids": ["a", "b"]}  # missing other fields
    engine = QueryEngine(embedder=DummyEmbedder(), vector_store=PartialVectorStore())
    actual = engine.query("foo", n_results=2)
    expected = QueryResult(
        answer=actual.answer,  # Accept whatever answer the engine generates
        context=[
            ContextChunk(id="a", text=None, metadata=None, distance=None),
            ContextChunk(id="b", text=None, metadata=None, distance=None),
        ]
    )
    assert actual == expected
    assert actual.answer
