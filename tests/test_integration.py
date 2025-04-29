import os
import tempfile
from fastapi.testclient import TestClient
from api import app
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.integration
def test_index_and_query(ollama_container):
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
        answer = data["results"]["answer"]
        assert "login bug" in answer.lower() or "fixed" in answer.lower()
