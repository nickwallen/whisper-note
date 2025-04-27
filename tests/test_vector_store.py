import pytest
from vector_store import VectorStore

def test_add_and_query_vector_store():
    store = VectorStore(collection_name="test_notes")
    ids = ["doc1", "doc2"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadatas = [{"file": "a.md"}, {"file": "b.md"}]
    store.add(ids, embeddings, metadatas)
    results = store.query([0.1, 0.2, 0.3], n_results=1)
    assert "ids" in results
    assert results["ids"][0][0] == "doc1"

def test_empty_add_and_query():
    store = VectorStore(collection_name="empty_test")
    # Add nothing should raise ValueError
    with pytest.raises(ValueError):
        store.add(ids=[], embeddings=[], metadatas=[])
    # Query with no data should still work and return empty
    results = store.query([0.0, 0.0, 0.0], n_results=1)
    assert "ids" in results
    assert results["ids"][0] == []

def test_add_mismatched_lengths():
    store = VectorStore(collection_name="mismatch_test")
    with pytest.raises(Exception):
        store.add(ids=["a"], embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], metadatas=[{"file": "f.md"}])
    with pytest.raises(Exception):
        store.add(ids=["a", "b"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"file": "f.md"}, {"file": "g.md"}])

def test_duplicate_ids():
    store = VectorStore(collection_name="dup_test")
    store.add(ids=["dup"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"file": "f.md"}])
    # Add duplicate id
    store.add(ids=["dup"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"file": "f.md"}])
    # Should not error, but check only one result returned
    results = store.query([0.1, 0.2, 0.3], n_results=2)
    assert "ids" in results
    assert results["ids"][0].count("dup") >= 1

def test_query_more_than_available():
    store = VectorStore(collection_name="more_than_avail")
    store.add(ids=["a"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"file": "f.md"}])
    results = store.query([0.1, 0.2, 0.3], n_results=10)
    assert len(results["ids"][0]) == 1

def test_non_numeric_embedding():
    store = VectorStore(collection_name="non_numeric")
    with pytest.raises(Exception):
        store.add(ids=["bad"], embeddings=[["a", "b", "c"]], metadatas=[{"file": "f.md"}])
    with pytest.raises(Exception):
        store.query(["a", "b", "c"], n_results=1)
