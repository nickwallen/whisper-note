import os
import tempfile
import shutil
import pytest
from indexer import Indexer
from vector_store import VectorStore

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
        "subdir/c.txt": "subfile1\nsubfile2"
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
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer"))
    metrics = indexer.index_directory(temp_dir_with_files, file_extensions=[".txt"])
    assert metrics.file_count == 3
    assert metrics.chunk_count == 7  # 2 in a.txt, 3 in b.txt, 2 in c.txt
    assert metrics.extensions_indexed == {".txt"}

def test_indexer_skips_unchanged_file():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "foo.txt")
    with open(file_path, "w") as f:
        f.write("alpha\nbeta\n")
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="skip_test"))
    # First index
    metrics1 = indexer.index_directory(temp_dir, file_extensions=[".txt"])
    # Get vector count after first index
    count1 = len(indexer.vector_store.collection.get(where={"file": "foo.txt"})["ids"])
    # Second index (should skip, as file unchanged)
    metrics2 = indexer.index_directory(temp_dir, file_extensions=[".txt"])
    count2 = len(indexer.vector_store.collection.get(where={"file": "foo.txt"})["ids"])
    assert count1 == 2
    assert count2 == 2  # Should not increase
    # Now change the file
    with open(file_path, "w") as f:
        f.write("gamma\ndelta\n")
    metrics3 = indexer.index_directory(temp_dir, file_extensions=[".txt"])
    count3 = len(indexer.vector_store.collection.get(where={"file": "foo.txt"})["ids"])
    assert count3 == 2  # Still two, but new content
    shutil.rmtree(temp_dir)

def test_indexer_empty_dir():
    temp_dir = tempfile.mkdtemp()
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_empty"))
    metrics = indexer.index_directory(temp_dir, file_extensions=[".txt"])
    assert metrics.file_count == 0
    assert metrics.chunk_count == 0
    assert metrics.extensions_indexed == set()
    shutil.rmtree(temp_dir)

def test_indexer_non_matching_extensions(temp_dir_with_files):
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_nomatch"))
    metrics = indexer.index_directory(temp_dir_with_files, file_extensions=[".md"])  # No .md files
    assert metrics.file_count == 0
    assert metrics.chunk_count == 0
    assert metrics.extensions_indexed == set()

def test_indexer_file_no_extension():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "filewithoutext")
    with open(file_path, "w") as f:
        f.write("line1\nline2\n")
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_noext"))
    metrics = indexer.index_directory(temp_dir)
    assert metrics.file_count == 1
    assert metrics.chunk_count == 2
    assert metrics.extensions_indexed == set()
    shutil.rmtree(temp_dir)

def test_indexer_file_whitespace_only():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "whitespace.txt")
    with open(file_path, "w") as f:
        f.write("   \n   \n")
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_ws"))
    metrics = indexer.index_directory(temp_dir, file_extensions=[".txt"])
    assert metrics.file_count == 1
    assert metrics.chunk_count == 0
    assert metrics.extensions_indexed == {".txt"}
    shutil.rmtree(temp_dir)

def test_indexer_file_empty():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "empty.txt")
    with open(file_path, "w") as f:
        pass  # Write nothing
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_emptyfile"))
    metrics = indexer.index_directory(temp_dir, file_extensions=[".txt"])
    assert metrics.file_count == 1
    assert metrics.chunk_count == 0
    assert metrics.extensions_indexed == {".txt"}
    shutil.rmtree(temp_dir)

def test_indexer_file_non_txt_extension():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "note.md")
    with open(file_path, "w") as f:
        f.write("alpha\nbeta\n")
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_md"))
    metrics = indexer.index_directory(temp_dir, file_extensions=[".md"])
    assert metrics.file_count == 1
    assert metrics.chunk_count == 2
    assert metrics.extensions_indexed == {".md"}
    shutil.rmtree(temp_dir)
