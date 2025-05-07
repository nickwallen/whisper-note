from abc import ABC, abstractmethod


class LangModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        pass
