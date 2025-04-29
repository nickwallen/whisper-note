import pytest
from testcontainers.ollama import OllamaContainer
from query import OLLAMA_URL_ENV, OLLAMA_MODEL_ENV
import os


@pytest.fixture(scope="session")
def ollama_container(model = "llama2:7b"):
    with OllamaContainer() as ollama:
        ollama.pull_model(model)
        endpoint = ollama.get_endpoint()
        os.environ[OLLAMA_URL_ENV] = endpoint
        os.environ[OLLAMA_MODEL_ENV] = model
        yield endpoint
