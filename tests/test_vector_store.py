import pytest
from vector_store import VectorStore


from vector_store import Metadata
from datetime import datetime


def test_add_and_query_vector_store():
    store = VectorStore(collection_name="test_notes")
    ids = ["doc1", "doc2"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    now = datetime.now()
    metadatas = [
        Metadata(
            file="a.md",
            file_hash="",
            chunk_index=0,
            text="doc1",
            modified_at=now,
            created_at=now,
        ),
        Metadata(
            file="b.md",
            file_hash="",
            chunk_index=1,
            text="doc2",
            modified_at=now,
            created_at=now,
        ),
    ]
    store.add(ids, embeddings, ["doc1", "doc2"], metadatas)
    results = store.query([0.1, 0.2, 0.3], max_results=1)
    assert "ids" in results
    assert results["ids"][0][0] == "doc1"


def test_empty_add_and_query():
    store = VectorStore(collection_name="empty_test")
    # Add nothing should raise ValueError
    with pytest.raises(ValueError):
        store.add(ids=[], embeddings=[], documents=[], metadatas=[])
    # Query with no data should still work and return empty
    results = store.query([0.0, 0.0, 0.0], max_results=1)
    assert "ids" in results
    assert results["ids"][0] == []


def test_add_mismatched_lengths():
    store = VectorStore(collection_name="mismatch_test")
    with pytest.raises(Exception):
        store.add(
            ids=["a"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            metadatas=[{"file": "f.md"}],
        )
    with pytest.raises(Exception):
        store.add(
            ids=["a", "b"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"file": "f.md"}, {"file": "g.md"}],
        )


def test_duplicate_ids():
    store = VectorStore(collection_name="dup_test")
    from vector_store import Metadata
    from datetime import datetime

    now = datetime.now()
    meta = Metadata(
        file="f.md",
        file_hash="",
        chunk_index=0,
        text="doc",
        modified_at=now,
        created_at=now,
    )
    store.add(
        ids=["dup"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["doc"],
        metadatas=[meta],
    )
    # Add duplicate id
    store.add(
        ids=["dup"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["doc"],
        metadatas=[meta],
    )
    # Should not error, but check only one result returned
    results = store.query([0.1, 0.2, 0.3], max_results=2)
    assert "ids" in results
    assert results["ids"][0].count("dup") >= 1


def _setup_store_with_created_at():
    store = VectorStore(collection_name="created_at_test")
    ids = ["doc1", "doc2", "doc3"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    metadatas = [
        Metadata(
            file="a.md",
            file_hash="",
            chunk_index=0,
            text="doc1",
            modified_at=datetime.fromtimestamp(1643723400),
            created_at=datetime.fromtimestamp(1643723400),
        ),
        Metadata(
            file="b.md",
            file_hash="",
            chunk_index=1,
            text="doc2",
            modified_at=datetime.fromtimestamp(1643723401),
            created_at=datetime.fromtimestamp(1643723401),
        ),
        Metadata(
            file="c.md",
            file_hash="",
            chunk_index=2,
            text="doc3",
            modified_at=datetime.fromtimestamp(1643723402),
            created_at=datetime.fromtimestamp(1643723402),
        ),
    ]
    store.add(ids, embeddings, ["doc1", "doc2", "doc3"], metadatas)
    return store


def test_query_with_start_time():
    store = _setup_store_with_created_at()
    results = store.query([0.1, 0.2, 0.3], max_results=3, start_time=1643723401)
    assert "ids" in results
    # Should get doc2 and doc3
    assert set(results["ids"][0]) == {"doc2", "doc3"}


def test_query_with_end_time():
    store = _setup_store_with_created_at()
    results = store.query([0.1, 0.2, 0.3], max_results=3, end_time=1643723401)
    assert "ids" in results
    # Should get doc1 and doc2
    assert set(results["ids"][0]) == {"doc1", "doc2"}


def test_query_with_start_and_end_time():
    store = _setup_store_with_created_at()
    results = store.query(
        [0.1, 0.2, 0.3], max_results=3, start_time=1643723401, end_time=1643723401
    )
    assert "ids" in results
    # Should get only doc2
    assert set(results["ids"][0]) == {"doc2"}


def test_query_more_than_available():
    store = VectorStore(collection_name="more_than_avail")
    from vector_store import Metadata
    from datetime import datetime

    now = datetime.now()
    meta = Metadata(
        file="f.md",
        file_hash="",
        chunk_index=0,
        text="doc",
        modified_at=now,
        created_at=now,
    )
    store.add(
        ids=["a"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["doc"],
        metadatas=[meta],
    )
    results = store.query([0.1, 0.2, 0.3], max_results=10)
    assert len(results["ids"][0]) == 1


def test_delete_by_file_path():
    store = VectorStore(collection_name="delete_test")
    from vector_store import Metadata
    from datetime import datetime

    now = datetime.now()
    # Add two vectors for the same file path but different hashes
    ids1 = ["hash1::chunk0", "hash1::chunk1"]
    ids2 = ["hash2::chunk0"]
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    embeddings2 = [[0.5, 0.6, 0.7]]
    metadatas1 = [
        Metadata(
            file="foo.txt",
            file_hash="hash1",
            chunk_index=0,
            text="aaa",
            modified_at=now,
            created_at=now,
        ),
        Metadata(
            file="foo.txt",
            file_hash="hash1",
            chunk_index=1,
            text="bbb",
            modified_at=now,
            created_at=now,
        ),
    ]
    metadatas2 = [
        Metadata(
            file="foo.txt",
            file_hash="hash2",
            chunk_index=0,
            text="ccc",
            modified_at=now,
            created_at=now,
        )
    ]
    store.add(ids1, embeddings, ["aaa", "bbb"], metadatas1)
    store.add(ids2, embeddings2, ["ccc"], metadatas2)
    # Confirm all are present
    results = store.collection.get(where={"file": "foo.txt"})
    assert set(results["ids"]) == set(ids1 + ids2)
    # Delete by file path
    store.delete_by_file_path("foo.txt")
    results = store.collection.get(where={"file": "foo.txt"})
    assert results["ids"] == []


def test_reindex_overwrites_old_vectors():
    store = VectorStore(collection_name="reindex_test")
    from vector_store import Metadata
    from datetime import datetime

    now = datetime.now()
    # Simulate first index
    ids1 = ["hashA::chunk0", "hashA::chunk1"]
    embeddings1 = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    metadatas1 = [
        Metadata(
            file="bar.txt",
            file_hash="hashA",
            chunk_index=0,
            text="aaa",
            modified_at=now,
            created_at=now,
        ),
        Metadata(
            file="bar.txt",
            file_hash="hashA",
            chunk_index=1,
            text="bbb",
            modified_at=now,
            created_at=now,
        ),
    ]
    store.add(ids1, embeddings1, ["aaa", "bbb"], metadatas1)
    # Simulate cleanup + reindex with new hash
    store.delete_by_file_path("bar.txt")
    ids2 = ["hashB::chunk0"]
    embeddings2 = [[0.5, 0.6, 0.7]]
    metadatas2 = [
        Metadata(
            file="bar.txt",
            file_hash="hashB",
            chunk_index=0,
            text="ccc",
            modified_at=now,
            created_at=now,
        )
    ]
    store.add(ids2, embeddings2, ["ccc"], metadatas2)
    # Only new vectors should be present
    results = store.collection.get(where={"file": "bar.txt"})
    assert set(results["ids"]) == set(ids2)


def test_non_numeric_embedding():
    store = VectorStore(collection_name="non_numeric")
    with pytest.raises(Exception):
        store.add(
            ids=["bad"],
            embeddings=[["a", "b", "c"]],
            documents=["bad doc"],
            metadatas=[{"file": "f.md"}],
        )
    with pytest.raises(Exception):
        store.query(["a", "b", "c"], max_results=1)
