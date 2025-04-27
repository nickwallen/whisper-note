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
    num_files, num_chunks = indexer.index_directory(temp_dir_with_files, file_extensions=[".txt"])
    assert num_files == 3
    assert num_chunks == 7  # 2 in a.txt, 3 in b.txt, 2 in c.txt

def test_indexer_empty_dir():
    temp_dir = tempfile.mkdtemp()
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_empty"))
    num_files, num_chunks = indexer.index_directory(temp_dir, file_extensions=[".txt"])
    assert num_files == 0
    assert num_chunks == 0
    shutil.rmtree(temp_dir)

def test_indexer_non_matching_extensions(temp_dir_with_files):
    indexer = Indexer(embedder=DummyEmbedder(), chunker=DummyChunker(), vector_store=VectorStore(collection_name="test_indexer_nomatch"))
    num_files, num_chunks = indexer.index_directory(temp_dir_with_files, file_extensions=[".md"])  # No .md files
    assert num_files == 0
    assert num_chunks == 0
