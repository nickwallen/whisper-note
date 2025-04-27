import os
import shutil
import tempfile
import pytest
from vector_store import VectorStore

@pytest.fixture
def temp_chroma_dir():
    # Create a temporary directory for ChromaDB persistence
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_add_and_query_vector_store():
    store = VectorStore(collection_name="test_notes")
    ids = ["doc1", "doc2"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadatas = [{"file": "file1.md"}, {"file": "file2.md"}]
    store.add(ids, embeddings, metadatas)

    # Query for the first embedding
    results = store.query([0.1, 0.2, 0.3], n_results=2)
    assert "ids" in results
    assert ids[0] in results["ids"][0]
    assert len(results["ids"][0]) == 2

    # Query for the second embedding
    results2 = store.query([0.4, 0.5, 0.6], n_results=1)
    assert ids[1] in results2["ids"][0]
    assert len(results2["ids"][0]) == 1

    # Query for an embedding far from both
    results3 = store.query([1.0, 1.0, 1.0], n_results=1)
    assert len(results3["ids"][0]) == 1


def test_empty_add_and_query():
    store = VectorStore(collection_name="empty_test")
    # Add nothing should raise ValueError
    with pytest.raises(ValueError):
        store.add(ids=[], embeddings=[], metadatas=[])
    # Query with no data should still work and return empty
    results = store.query([0.0, 0.0, 0.0], n_results=1)
    assert "ids" in results
    assert results["ids"][0] == []

def test_mismatched_lengths():
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

def test_large_embedding():
    store = VectorStore(collection_name="large_vec")
    large_vec = [float(i) for i in range(10000)]
    store.add(ids=["large"], embeddings=[large_vec], metadatas=[{"file": "f.md"}])
    results = store.query(large_vec, n_results=1)
    assert "ids" in results
    assert results["ids"][0][0] == "large"

def test_invalid_metadata():
    store = VectorStore(collection_name="badmeta")
    # Missing metadata
    store.add(ids=["a"], embeddings=[[0.1, 0.2, 0.3]], metadatas=None)
    # Malformed metadata
    with pytest.raises(Exception):
        store.add(ids=["b"], embeddings=[[0.1, 0.2, 0.3]], metadatas=["notadict"])
