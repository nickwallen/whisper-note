## Goal: Never Dread Status Updates Again

If you’re a developer, chances are you spend too much time writing repetitive status updates—daily stand-ups, weekly team summaries, biweekly group reports. While these updates are essential for coordination and transparency, the process of preparing them is tedious, time-consuming, and often interrupts your flow.

**This app is built to eliminate that friction.**

By connecting directly to your daily notes, it lets you instantly generate context-aware, relevant updates—on demand. Instead of wracking your brain to recall what you did, just ask:

- “Summarize the work I completed this week.”
- “What updates should I provide for stand-up?”
- “What were my biggest wins and blockers over the last two weeks?”

No more manual sifting, no more copy-pasting. You get accurate, tailored updates in seconds, so you can focus on your real work—not on reporting it.

**Note:**

That's the vision, but we're far from that today. Consider the description above aspirational for now.


## Usage

### Setup

```
make install
```

### Start

```
make ollama
```
```
make run
```

### Usage


Index your notes.
```sh
python cli.py index /path/to/notes --file-extensions .txt .md
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
│ * Completed initial framing document and reviewed with stakeholders (MTTR Initiative). │
╰────────────────────────────────────────────────────────────────────────────────────────╯

```

## Ollama Installation

Ollama is required for running local language models (LLMs) privately on your machine. You can install it using:

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