import typer
from typing import List, Optional
from pathlib import Path
import sys
import requests
import json

WHISPER_NOTE_DAEMON_URL = (
    "http://localhost:8000"  # Change this if your server runs elsewhere
)
TIMEOUT = 60  # seconds

app = typer.Typer(help="Whisper Note: Index and query your files with AI.")


@app.command()
def index(
    directory: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, help="Directory to index."
    ),
    file_extensions: List[str] = typer.Option([".txt", ".md"], help="File extensions to include."),
):
    """Index a directory with specified file extensions."""
    try:
        resp = requests.post(
            f"{WHISPER_NOTE_DAEMON_URL}/api/v1/index",
            json={
                "directory": str(directory),
                "file_extensions": file_extensions,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        typer.echo(resp.text)
    except Exception as e:
        typer.secho(f"Indexing failed: {e}", fg=typer.colors.RED)


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question to ask the AI."),
    scope: Optional[Path] = typer.Option(
        None, help="Restrict search to a file or directory."
    ),
):
    """Ask a question about your indexed files."""
    try:
        payload = {"query": question}
        if scope:
            payload["scope"] = str(scope)
        resp = requests.post(
            f"{WHISPER_NOTE_DAEMON_URL}/api/v1/query", json=payload, timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("results", {}).get("answer")
        if answer:
            typer.echo(answer)
        else:
            typer.secho("No answer found in response.", fg=typer.colors.YELLOW)
    except Exception as e:
        typer.secho(f"Query failed: {e}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
