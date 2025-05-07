import pytest
from testcontainers.ollama import OllamaContainer
from ollama import OLLAMA_URL_ENV, OLLAMA_MODEL_ENV
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def ollama_container(model="llama2:7b"):
    # Uses a persistent temp directory for caching models
    model_cache = tempfile.gettempdir() + "/ollama_model_cache"
    os.makedirs(model_cache, exist_ok=True)

    ollama = OllamaContainer()
    ollama.with_volume_mapping(model_cache, "/root/.ollama/models", mode="rw")
    with ollama:
        ollama.pull_model(model)
        endpoint = ollama.get_endpoint()
        os.environ[OLLAMA_URL_ENV] = endpoint
        os.environ[OLLAMA_MODEL_ENV] = model
        yield endpoint
