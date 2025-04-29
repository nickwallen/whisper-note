import pytest
from embeddings import Embedder


@pytest.fixture(scope="module")
def embedder():
    # Use module scope to avoid repeated model loading
    return Embedder()


def test_embed_one_returns_vector(embedder):
    text = "Hello world!"
    vec = embedder.embed_one(text)
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    assert len(vec) > 0


def test_embed_returns_vectors(embedder):
    texts = ["First text", "Second text"]
    vecs = embedder.embed(texts)
    assert isinstance(vecs, list)
    assert len(vecs) == 2
    assert all(isinstance(v, list) for v in vecs)
    assert all(isinstance(x, float) for v in vecs for x in v)
    # All vectors should have the same length
    lengths = set(len(v) for v in vecs)
    assert len(lengths) == 1


def test_embed_empty_list(embedder):
    vecs = embedder.embed([])
    assert isinstance(vecs, list)
    assert vecs == []


def test_embed_one_and_embed_match(embedder):
    text = "Test string"
    vec1 = embedder.embed_one(text)
    vec2 = embedder.embed([text])[0]
    assert vec1 == vec2
