## ✨ Never Dread Status Updates Again

If you’re a developer, chances are you spend too much time writing repetitive status updates—daily stand-ups, weekly team summaries, biweekly group reports. While these updates are essential for coordination and transparency, the process of preparing them is tedious, time-consuming, and often interrupts your flow.

**This app is built to eliminate that friction.**

By connecting directly to your daily notes, it lets you instantly generate context-aware, relevant updates—on demand. Instead of wracking your brain to recall what you did, just ask:

- “Summarize the work I completed this week.”
- “What updates should I provide for stand-up?”
- “What were my biggest wins and blockers over the last two weeks?”

No more manual sifting, no more copy-pasting. You get accurate, tailored updates in seconds, so you can focus on your real work—not on reporting it.

**Note:** That's the vision, but we're far from that today. Consider the description above aspirational for now.

## 📚 What is whisper-note?

whisper-note is a personal, privacy-focused application that uses a Retrieval-Augmented Generation (RAG) approach to generate clear summaries of your work activity from natural language queries. It combines semantic search, time-aware filtering, and creative prompting to provide accurate and relevant responses based on your own notes.


## 🛠️ Setup

```
make install
```

### Ollama Installation

Ollama is required for running local language models (LLMs) privately on your machine. You can install it using:

```sh
brew install ollama
ollama pull llama2
```

## 🚀 Usage

Start the app.
```
make run
```

Activate the Python virtual environment.
```
source venv/bin/activate
```

Index your notes. For example your Obsidian notes stored at `~/Documents/Obsidian/`.
```sh
python cli.py index ~/Documents/Obsidian/ --file-extensions .txt .md
```

Generate that update for stand-up.
```
$ python cli.py chat                                                      
Type your question and press Enter. Type 'q' or 'Ctrl+C' to end the session.

> What did I work on yesterday?

╭─ AI ───────────────────────────────────────────────────────────────────────────────────╮
│ * Split out frontend endpoints for improved clarity and maintainability.               │
│ * Fixed authN/authZ issues across all routes.                                          │
│ * Resolved devflow and integration branch merge issue.                                 │
│ * Completed MTTR initial framing document and reviewed with stakeholders.              │
╰────────────────────────────────────────────────────────────────────────────────────────╯

```

## 🛠️ Technical Architecture

whisper-note uses a Retrieval-Augmented Generation (RAG) approach to generate clear summaries of your work activity from natural language queries. It combines semantic search, time-aware filtering, and creative prompting to provide accurate and relevant responses based on your own notes.

### Workflows

#### 1. Indexing

Your notes are preprocessed and stored in a vector database for fast semantic search.
- **Chunking**: Notes are split into smaller passages for more precise retrieval.
- **Enrichment**: Metadata is added to each chunk (e.g., timestamps, source file).
- **Embedding**: Each chunk is converted into a vector using an embedding model, capturing its semantic meaning.
- **Storage**: Chunks, embeddings, and metadata are saved to ChromaDB for later retrieval.

#### 2. Query Serving

When you ask a question like “What did I complete last week?”, the system executes:
- **Temporal Analysis**: An LLM parses your query to extract a structured time range to ensure only contextually relevant notes are retrieved.
- **Embedding & Retrieval**: The query is converted into an embedding vector. ChromaDB is queried for semantically similar context within the time window.
- **Prompt Augmentation**: A carefully designed prompt is constructed that includes the prompt, relevant context, and instructions to guide the LLM.
- **LLM Generation**: The prompt is sent to a local (via Ollama) or remote LLM. The model generates a concise summary aligned with the original intent and style (e.g., standup-style bullets).

### ✅ Benefits

* Grounded Summarization: Answers are always rooted in your real notes — not hallucinated.
* High Relevance: Combines vector similarity with time filtering for precision.
* Speed & Convenience: Generate work summaries in seconds.
* Local-First & Private: Run fully offline if desired — your notes never leave your machine.
* Developer Friendly: Modular and extensible for custom prompts, embedding models, and LLMs.

