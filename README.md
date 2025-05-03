

## Goal

An app that can access your daily notes allowing you to ask questions about them.

Use Cases:
* Summarize the work that I completed this week?
* What updates should I provide for stand-up?

## Usage

### Start

```
make ollama
```
```
make run
```

### CLI

#### Example Commands

```sh
# Index all .txt and .md files in a directory
python cli.py index /path/to/notes --file-extensions .txt .md

# Query your indexed notes
python cli.py query "What did I work on last week?"

# Show current index status (files, chunks, failed files)
python cli.py status

# Start an interactive chat session
python cli.py chat
```

### Daemon

#### Index

```
curl -s -X POST "http://localhost:8000/api/v1/index" \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/Users/nick.allen/Dropbox/Documents/Obsidian Vaults",
    "file_extensions": [".txt", ".md"]
  }' | jq
```

### Query

```
curl -s -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did I get done last week?"
  }' | jq
```

## Plan

### Index

1. Read from a directory of markdown or text
2. Chunk the file content
3. Add additional metadata like create/modified time and file name
4. Create embeddings for each chunk
5. Store embeddings in a simple local vector store (like ChromaDB)

### Query

1. Create embeddings for the query
2. Find the most similar chunks
3. Send the most similar chunks to a language model along with the query
4. Get the response from the language model


## Ollama Installation

Ollama is required for running local language models (LLMs) privately on your machine. You can install it using one of the following methods:

### Homebrew
```sh
brew install ollama
```

After installation, you can start the Ollama service using the Makefile:
```sh
make ollama
```

This will run `ollama serve` in the foreground. You can then query local LLMs from the API.

If you see a message about the model not being pulled, run:

```sh
ollama pull llama2
```