import os
import tempfile
import uuid
from fastapi.testclient import TestClient
from api import app, get_collection_name
import pytest


@pytest.fixture
def temp_dir_with_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        monday = os.path.join(tmpdir, "Daily, Monday.txt")
        with open(monday, "w") as f:
            f.write("Fixed the login bug.")
        tuesday = os.path.join(tmpdir, "Daily, Tuesday.txt")
        with open(tuesday, "w") as f:
            f.write("Added the new shiny button feature.")
        wednesday = os.path.join(tmpdir, "Daily, Wednesday.txt")
        with open(wednesday, "w") as f:
            f.write("Was on-call and responded to multiple pages.")
        yield tmpdir


def unique_collection():
    return f"test_collection_{uuid.uuid4()}"


def override_collection(collection_name):
    app.dependency_overrides[get_collection_name] = lambda: collection_name


def clear_override():
    app.dependency_overrides = {}


@pytest.mark.integration
def test_index_directory(temp_dir_with_files):
    collection = unique_collection()
    override_collection(collection)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/index",
        json={"directory": temp_dir_with_files, "file_extensions": [".txt"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_count"] == 3
    assert data["chunk_count"] == 3  # One chunk per file
    assert isinstance(data["failed_files"], list)
    clear_override()


@pytest.mark.integration
def test_get_index(temp_dir_with_files):
    collection = unique_collection()
    override_collection(collection)
    client = TestClient(app)
    client.post(
        "/api/v1/index",
        json={"directory": temp_dir_with_files, "file_extensions": [".txt"]},
    )
    # Now test GET
    resp = client.get("/api/v1/index")
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_count"] == 3
    assert data["chunk_count"] == 3
    assert isinstance(data["failed_files"], list)
    clear_override()


@pytest.mark.integration
def test_query(temp_dir_with_files):
    collection = unique_collection()
    override_collection(collection)
    client = TestClient(app)
    # Index first
    client.post(
        "/api/v1/index",
        json={"directory": temp_dir_with_files, "file_extensions": [".txt"]},
    )
    # Query
    resp = client.post("/api/v1/query", json={"query": "What did Nick fix on Monday?"})
    assert resp.status_code == 200
    data = resp.json()
    answer = data["results"]["answer"]
    assert "login bug" in answer.lower() or "fixed" in answer.lower()
    clear_override()
