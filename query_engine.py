import requests
from embeddings import Embedder
from vector_store import VectorStore

class QueryEngine:
    def __init__(self, embedder=None, vector_store=None, ollama_url="http://localhost:11434", ollama_model="llama2"):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def query(self, query_string, n_results=5, prompt_template=None):
        """
        Retrieve top matching chunks and use Ollama to answer the query using those chunks as context.
        Returns a dict with the answer and the context used.
        """
        # Retrieve context chunks
        query_embedding = self.embedder.embed([query_string])[0]
        results = self.vector_store.query(query_embedding, n_results=n_results)
        context_chunks = []
        for i in range(len(results["ids"])):
            item = {
                "id": results["ids"][i],
                "text": results["documents"][i] if "documents" in results else None,
                "metadata": results["metadatas"][i] if "metadatas" in results else None,
                "distance": results["distances"][i] if "distances" in results else None,
            }
            context_chunks.append(item)
        # Compose context string, ensuring all items are strings
        def ensure_str(x):
            if isinstance(x, list):
                return "\n".join(str(i) for i in x)
            return str(x)
        context_texts = [ensure_str(chunk["text"]) for chunk in context_chunks if chunk["text"]]
        context = "\n".join(context_texts)
        if not prompt_template:
            prompt_template = (
                "Answer the following question using only the provided context.\n"
                "If the answer cannot be found in the context, say you don't know.\n\n"
                "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
        prompt = prompt_template.format(context=context, query=query_string)
        import json
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=60,
                stream=True
            )
            response.raise_for_status()
            answer = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        answer += data["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                answer = "Ollama API endpoint not found. Is the Ollama server running and is the model pulled?"
            else:
                answer = f"Ollama API error: {e}"
        except Exception as e:
            answer = f"Error communicating with Ollama: {e}"
        return {"answer": answer.strip(), "context": context_chunks}
