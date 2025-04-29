import pytest
import time
import requests
from testcontainers.core.container import DockerContainer
from query import OLLAMA_URL_ENV
import logging
import os


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def ollama_container(image="ollama/ollama:latest", model_name="llama2"):
    with DockerContainer(image).with_exposed_ports(11434) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(11434)
        url = f"http://{host}:{port}"

        # Step 1: Wait for Ollama to be up (just /api/tags)
        wait_for_startup(url, image)
        # Step 2: Pull the model
        pull_model(url, model_name)
        # Step 3: Wait for /api/chat to be ready with the model
        wait_for_chat_ready(url, model_name)

        os.environ[OLLAMA_URL_ENV] = url
        yield url


def pull_model(url, model_name):
    pull_resp = requests.post(f"{url}/api/pull", json={"name": model_name})
    pull_resp.raise_for_status()
    logger.info(f"Pulled model {model_name} for Ollama test container.")


def wait_for_startup(url, image):
    for _ in range(30):
        try:
            r = requests.get(f"{url}/api/tags")
            if r.ok:
                logger.info(f"/api/tags available at: {url}")
                return
        except Exception as e:
            logger.info(f"Waiting for Ollama container to start at: {url} ({e})")
        time.sleep(1)
    raise RuntimeError(f"{image} did not start in time!")


def wait_for_chat_ready(url, model_name):
    for _ in range(30):
        try:
            chat_resp = requests.post(
                f"{url}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                timeout=5,
            )
            if chat_resp.status_code == 200:
                logger.info(
                    f"/api/chat endpoint READY for model '{model_name}' at: {url}"
                )
                return
            else:
                logger.info(
                    f"/api/chat for model '{model_name}' returned status {chat_resp.status_code}: {chat_resp.text}"
                )
        except Exception as e:
            logger.info(
                f"/api/chat not ready yet for model '{model_name}' at: {url} ({e})"
            )
        time.sleep(1)
    raise RuntimeError(
        f"/api/chat for model '{model_name}' did not become ready in time!"
    )
