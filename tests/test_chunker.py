import pytest
import os
import tempfile
from chunker import Chunker, FileMetadata
import re


def test_chunk_by_size_basic(tmp_path):
    chunker = Chunker(chunk_size=5)
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("abcdefghij", encoding="utf-8")
    chunks = chunker.chunk_file(str(file_path))
    # Should include both chunks and metadata
    assert any("abcde" in chunk for chunk in chunks)
    assert any("fghij" in chunk for chunk in chunks)
    assert any("User note: title" in chunk for chunk in chunks)
    # Check for correctly formatted date (e.g., Tuesday, April 29, 2025)
    date_pattern = re.compile(r"[A-Za-z]+, [A-Za-z]+ \d{2}, \d{4}")
    assert any(date_pattern.search(chunk) for chunk in chunks)


def test_chunk_by_size_overlap(tmp_path):
    chunker = Chunker(chunk_size=5, overlap=2)
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("abcdefghij", encoding="utf-8")
    chunks = chunker.chunk_file(str(file_path))
    assert any("abcde" in chunk for chunk in chunks)
    assert any("defgh" in chunk for chunk in chunks)
    assert any("ghij" in chunk for chunk in chunks)
    assert any("j" in chunk for chunk in chunks)


def test_chunk_by_paragraph(tmp_path):
    chunker = Chunker(chunk_size=100, split_on=r"\n\n")
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("Para1.\n\nPara2 is a bit longer.\n\nPara3.", encoding="utf-8")
    chunks = chunker.chunk_file(str(file_path))
    assert any("Para1." in chunk for chunk in chunks)
    assert any("Para2 is a bit longer." in chunk for chunk in chunks)
    assert any("Para3." in chunk for chunk in chunks)


def test_chunk_long_paragraph(tmp_path):
    chunker = Chunker(chunk_size=5, split_on=r"\n\n")
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("Short.\n\nThisIsALongPara.", encoding="utf-8")
    chunks = chunker.chunk_file(str(file_path))
    # 'Short.' = 6 chars, 'ThisIsALongPara.' = 16 chars
    assert any("Short" in chunk for chunk in chunks)
    assert any("." in chunk for chunk in chunks)
    assert any("ThisI" in chunk for chunk in chunks)
    assert any("sALon" in chunk for chunk in chunks)
    assert any("gPara" in chunk for chunk in chunks)


def test_chunk_empty(tmp_path):
    chunker = Chunker(chunk_size=5)
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("", encoding="utf-8")
    chunks = chunker.chunk_file(str(file_path))
    assert chunks == []


def test_chunk_all_whitespace(tmp_path):
    chunker = Chunker(chunk_size=5)
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("     ", encoding="utf-8")
    chunks = chunker.chunk_file(str(file_path))
    assert chunks == []


def test_overlap_greater_than_chunk_size(tmp_path):
    chunker = Chunker(chunk_size=5, overlap=10)
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("abcdefghij", encoding="utf-8")
    chunks = chunker.chunk_file(str(file_path))
    assert any("abcde" in chunk for chunk in chunks)
    assert any("fghij" in chunk for chunk in chunks)
