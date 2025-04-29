from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Returns a list of vectors (one per text).
        Ensures output is always a list of lists of floats.
        """
        result = self.model.encode(texts, convert_to_numpy=False)
        # If result is a tensor, convert to list
        if hasattr(result, "tolist"):
            result = result.tolist()
        # If result is a list of tensors, convert each
        if len(result) > 0 and hasattr(result[0], "tolist"):
            result = [v.tolist() for v in result]
        return result

    def embed_one(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        Ensures output is always a list of floats.
        """
        result = self.embed([text])[0]
        if hasattr(result, "tolist"):
            result = result.tolist()
        return result
