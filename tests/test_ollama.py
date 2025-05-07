import pytest
from ollama import OllamaLangModel


@pytest.mark.integration
def test_ollama_lang_model_hello(ollama_container):
    lang_model = OllamaLangModel()
    response = lang_model.generate("say hello")
    assert "hello" in response.lower()
