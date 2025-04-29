import os
import tempfile
import shutil
from fastapi.testclient import TestClient
from main import app
import pytest
from query import OLLAMA_URL_ENV
import logging

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.integration
def test_index_and_query(ollama_container):
    # Point the app to the test Ollama container
    os.environ[OLLAMA_URL_ENV] = ollama_container
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:

        # Create a set of notes to index
        monday = os.path.join(tmpdir, "Daily, Monday.txt")
        with open(monday, "w") as f:
            f.write("Fixed the login bug.")
        tuesday = os.path.join(tmpdir, "Daily, Tuesday.txt")
        with open(tuesday, "w") as f:
            f.write("Added the new shiny button feature.")
        wednesday = os.path.join(tmpdir, "Daily, Wednesday.txt")
        with open(wednesday, "w") as f:
            f.write("Was on-call and responded to multiple pages.")

        # Index the directory
        resp = client.post(
            "/api/v1/index", json={"directory": tmpdir, "file_extensions": [".txt"]}
        )
        assert resp.status_code == 200

        # Query for something relevant
        resp = client.post(
            "/api/v1/query", json={"query": "What did Nick fix on Monday?"}
        )
        assert resp.status_code == 200
        data = resp.json()
        answer = data["results"]["answer"] if "results" in data else data["answer"]

        # Check that the answer is reasonable
        assert "login bug" in answer.lower() or "fixed" in answer.lower()

        # Query for something else
        resp = client.post("/api/v1/query", json={"query": "What happened on Tuesday?"})
        assert resp.status_code == 200
        answer = resp.json()["results"]["answer"]
        assert "documentation" in answer.lower() or "tuesday" in answer.lower()
