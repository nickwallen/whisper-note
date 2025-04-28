import pytest
from chunker import Chunker

# Basic chunking by size
def test_chunk_by_size_basic():
    chunker = Chunker(chunk_size=5)
    text = "abcdefghij"  # length 10
    chunks = chunker.chunk_text(text)
    assert chunks == ["abcde", "fghij"]

# Chunking with overlap
def test_chunk_by_size_overlap():
    chunker = Chunker(chunk_size=5, overlap=2)
    text = "abcdefghij"  # length 10
    chunks = chunker.chunk_text(text)
    assert chunks == ["abcde", "defgh", "ghij", "j"]

# Chunking with split_on (paragraph)
def test_chunk_by_paragraph():
    chunker = Chunker(chunk_size=100, split_on=r"\n\n")
    text = "Para1.\n\nPara2 is a bit longer.\n\nPara3."
    chunks = chunker.chunk_text(text)
    assert chunks == ["Para1.", "Para2 is a bit longer.", "Para3."]

# Chunking with split_on (paragraph) by default
def test_chunk_by_paragraph():
    chunker = Chunker(chunk_size=100)
    text = "Para1.\n\nPara2 is a bit longer.\n\nPara3."
    chunks = chunker.chunk_text(text)
    assert chunks == ["Para1.", "Para2 is a bit longer.", "Para3."]

# Chunking with split_on and long paragraph
def test_chunk_long_paragraph():
    chunker = Chunker(chunk_size=5, split_on=r"\n\n")
    text = "Short.\n\nThisIsALongPara."
    chunks = chunker.chunk_text(text)
    # 'Short.' = 6 chars, 'ThisIsALongPara.' = 16 chars
    assert chunks == ["Short", ".", "ThisI", "sALon", "gPara", "."]

# Empty string
def test_chunk_empty():
    chunker = Chunker(chunk_size=5)
    assert chunker.chunk_text("") == []

# All whitespace string
def test_chunk_all_whitespace():
    chunker = Chunker(chunk_size=5)
    assert chunker.chunk_text("     ") == []

# Overlap greater than chunk size
def test_overlap_greater_than_chunk_size():
    chunker = Chunker(chunk_size=5, overlap=10)
    text = "abcdefghij"
    chunks = chunker.chunk_text(text)
    assert chunks == ["abcde", "fghij"]
