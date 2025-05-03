import os
import tempfile
import shutil
import pytest
import re
from indexer import Indexer
from vector_store import VectorStore
import chromadb

from fastapi.testclient import TestClient
from api import app


class DummyEmbedder:
    def embed(self, texts):
        # Return a vector of [len(text)] for each chunk (deterministic, fast)
        return [[float(len(t))] for t in texts]


class DummyChunker:
    def chunk_file(self, file_path):
        # Each line is a chunk
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]


@pytest.fixture
def temp_dir_with_files():
    temp_dir = tempfile.mkdtemp()
    files = {
        "a.txt": "hello\nworld\n",
        "b.txt": "foo\nbar\nbaz",
        "subdir/c.txt": "subfile1\nsubfile2",
    }
    for rel_path, content in files.items():
        abs_path = os.path.join(temp_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(content)
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_indexer_basic(temp_dir_with_files):
    # Use dummy embedder/chunker for speed and determinism
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(collection_name="test_indexer"),
    )
    metrics = indexer.index_dir(temp_dir_with_files, file_exts=[".txt"])
    assert metrics.file_count == 3
    assert metrics.chunk_count == 7  # 2 in a.txt, 3 in b.txt, 2 in c.txt
    # Check that all chunks have modification_time metadata
    for file in ["a.txt", "b.txt", "subdir/c.txt"]:
        abs_file = os.path.join(temp_dir_with_files, file)
        results = indexer.vector_store.collection.get(where={"file": abs_file})
        for md in results["metadatas"]:
            assert "modification_time" in md
            assert re.match(
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", md["modification_time"]
            )


def test_get_index_metrics_endpoint(temp_dir_with_files):
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(collection_name="test_indexer_metrics"),
    )
    metrics = indexer.index_dir(temp_dir_with_files, file_exts=[".txt"])
    assert metrics.file_count == 3
    assert metrics.chunk_count == 7
    assert isinstance(metrics.failed_files, list)


def test_indexer_skips_unchanged_file(tmp_path):
    file_path = tmp_path / "foo.txt"
    file_path.write_text("alpha\nbeta\n")
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(
            collection_name="skip_test", chroma_client=chromadb.Client()
        ),
    )
    # First index
    metrics1 = indexer.index_dir(str(tmp_path), file_exts=[".txt"])
    # Get vector count after first index
    abs_file = str(file_path)
    count1 = len(indexer.vector_store.collection.get(where={"file": abs_file})["ids"])
    # Second index (should skip, as file unchanged)
    metrics2 = indexer.index_dir(str(tmp_path), file_exts=[".txt"])
    count2 = len(indexer.vector_store.collection.get(where={"file": abs_file})["ids"])
    assert count1 == 2
    assert count2 == 2  # Should not increase
    assert metrics2.file_count == 0  # Should be zero, as nothing was indexed
    # Now change the file
    file_path.write_text("gamma\ndelta\n")
    metrics3 = indexer.index_dir(str(tmp_path), file_exts=[".txt"])
    count3 = len(indexer.vector_store.collection.get(where={"file": abs_file})["ids"])
    assert count3 == 2  # Still two, but new content


def test_indexer_empty_dir(tmp_path):
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(
            collection_name="test_indexer_empty", chroma_client=chromadb.Client()
        ),
    )
    metrics = indexer.index_dir(str(tmp_path), file_exts=[".txt"])
    assert metrics.file_count == 0
    assert metrics.chunk_count == 0


def test_indexer_non_matching_extensions(temp_dir_with_files):
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(
            collection_name="test_indexer_nomatch", chroma_client=chromadb.Client()
        ),
    )
    metrics = indexer.index_dir(temp_dir_with_files, file_exts=[".md"])  # No .md files
    assert metrics.file_count == 0
    assert metrics.chunk_count == 0


def test_indexer_file_no_extension(tmp_path):
    file_path = tmp_path / "filewithoutext"
    file_path.write_text("line1\nline2\n")
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(
            collection_name="test_indexer_noext", chroma_client=chromadb.Client()
        ),
    )
    metrics = indexer.index_dir(str(tmp_path))
    assert metrics.file_count == 1
    assert metrics.chunk_count == 2
    # Check vector store with absolute path
    results = indexer.vector_store.collection.get(where={"file": str(file_path)})
    assert len(results["ids"]) == 2
    for md in results["metadatas"]:
        assert md["file"] == str(file_path)


def test_indexer_file_whitespace_only(tmp_path):
    file_path = tmp_path / "whitespace.txt"
    file_path.write_text("   \n   \n")
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(
            collection_name="test_indexer_ws", chroma_client=chromadb.Client()
        ),
    )
    metrics = indexer.index_dir(str(tmp_path), file_exts=[".txt"])
    assert metrics.file_count == 1
    assert metrics.chunk_count == 0


def test_indexer_file_empty():

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "empty.txt")
    with open(file_path, "w") as f:
        pass  # Write nothing
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(collection_name="test_indexer_emptyfile"),
    )
    metrics = indexer.index_dir(temp_dir, file_exts=[".txt"])
    assert metrics.file_count == 1
    assert metrics.chunk_count == 0


def test_indexer_file_non_txt_extension():
    import tempfile

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "note.md")
    with open(file_path, "w") as f:
        f.write("alpha\nbeta\n")
    indexer = Indexer(
        embedder=DummyEmbedder(),
        chunker=DummyChunker(),
        vector_store=VectorStore(collection_name="test_indexer_md"),
    )
    metrics = indexer.index_dir(temp_dir, file_exts=[".md"])
    assert metrics.file_count == 1
    assert metrics.chunk_count == 2
