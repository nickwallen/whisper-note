from rich.table import Table
from rich.console import Console
import typer
from typing import List
from pathlib import Path
import requests
from rich.panel import Panel
from rich.markup import escape
from api import QueryResponse, ContextChunk, IndexMetricsResponse

WHISPER_NOTE_DAEMON_URL = (
    "http://localhost:8000"  # Change this if your server runs elsewhere
)
TIMEOUT = 60  # seconds

app = typer.Typer(help="Whisper Note: Index and query your files with AI.")


@app.command(
    name="index",
    help="Index a directory of files.",
    short_help="Index a directory.",
    rich_help_panel="Commands",
)
def index(
    directory: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, help="Directory to index."
    ),
    file_extensions: List[str] = typer.Option(
        [".txt", ".md"], help="File extensions to include."
    ),
):
    console = Console()
    try:
        metrics = submit_post_index(str(directory), file_extensions)
        view = show_index_metrics(metrics)
        console.print(view)
    except Exception as e:
        console.print(f"[red]Indexing failed: {e}[/red]")


@app.command(
    name="status",
    help="Show current status of the index.",
    short_help="Show index status.",
    rich_help_panel="Commands",
)
def status():
    console = Console()
    try:
        metrics = submit_get_index()
        view = show_index_metrics(metrics)
        console.print(view)
    except Exception as e:
        console.print(f"[red]Failed to retrieve index status: {e}[/red]")


@app.command(
    name="query",
    help="Ask the AI a question.",
    short_help="Ask the AI a question.",
    rich_help_panel="Commands",
)
def query(
    question: str = typer.Argument(..., help="Your question to ask the AI."),
    debug: bool = typer.Option(
        False, "--debug", help="Show context along with the answer."
    ),
):
    console = Console()
    try:
        payload = {"query": question}
        resp = submit_post_query(payload)
        if not resp.answer:
            console.print("[yellow]No answer found in response.[/yellow]")
            return
        view = show_answer(resp.answer)
        console.print(view)
        if debug and resp.context:
            for panel in show_context(resp.context):
                console.print(panel)
    except Exception as e:
        console.print(f"[red]Query failed: {e}[/red]")


@app.command(
    name="chat",
    help="Ask questions in an interactive session.",
    short_help="Ask questions in an interactive session.",
    rich_help_panel="Commands",
)
def chat(
    debug: bool = typer.Option(
        False, "--debug", help="Show context chunks used for the answer."
    )
):
    console = Console()
    console.print(
        "Type your question and press Enter. Type 'q' or 'Ctrl+C' to end the session.\n"
    )
    while True:
        try:
            try:
                question = console.input("[bold green]> [/bold green]")
            except EOFError:
                console.print("\nExiting chat.")
                return
            if question.strip().lower() in {"q", "quit"}:
                console.print("Exiting chat.")
                break
            if not question.strip():
                continue  # Ignore empty or whitespace-only input

            # Submit query
            with console.status("Thinking...", spinner="dots"):
                resp = submit_post_query({"query": question, "debug": debug})

            # Show answer
            console.print()
            if resp.answer:
                console.print(show_answer(resp.answer))
            else:
                console.print(no_answer_found())

            # Show context if debug is enabled
            if debug and resp.context:
                for panel in show_context(resp.context):
                    console.print(panel)
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting chat.")
            return
        except Exception as e:
            console.print(f"[red]Query failed: {e}[/red]")


def show_index_metrics(metrics: IndexMetricsResponse) -> Table:
    """Return a Table displaying index metrics."""
    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value", style="bold")
    metrics_table.add_row("Indexed files", str(metrics.file_count))
    metrics_table.add_row("Indexed chunks", str(metrics.chunk_count))
    metrics_table.add_row("Failed files", str(len(metrics.failed_files)))
    return metrics_table


def show_context(context: List[ContextChunk]) -> List[Panel]:
    """Return a list of Panels for the relevant context provided to the lang model."""
    panels = []
    for idx, chunk in enumerate(context, start=1):
        meta_text = chunk.text if chunk.text is not None else "[empty]"
        panel = Panel(
            escape(str(meta_text)),
            title=f"Context {idx}",
            title_align="left",
            border_style="bright_magenta",
        )
        panels.append(panel)
    return panels


def no_answer_found() -> Panel:
    """Return a panel indicating that no answer was found."""
    return Panel(
        "No answer found in response.",
        border_style="yellow",
        title="AI",
        title_align="left",
    )


def show_answer(answer: str) -> Panel:
    """Return the Panel for the answer provided by the lang model."""
    return Panel(
        escape(answer),
        title="AI",
        title_align="left",
        border_style="bright_cyan",
    )


def submit_post_query(payload: dict) -> QueryResponse:
    resp = requests.post(
        f"{WHISPER_NOTE_DAEMON_URL}/api/v1/query",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return QueryResponse(**resp.json())


def submit_get_index() -> IndexMetricsResponse:
    resp = requests.get(f"{WHISPER_NOTE_DAEMON_URL}/api/v1/index", timeout=TIMEOUT)
    resp.raise_for_status()
    return IndexMetricsResponse(**resp.json())


def submit_post_index(
    directory: str, file_extensions: list[str]
) -> IndexMetricsResponse:
    resp = requests.post(
        f"{WHISPER_NOTE_DAEMON_URL}/api/v1/index",
        json={
            "directory": directory,
            "file_extensions": file_extensions,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return IndexMetricsResponse(**resp.json())


if __name__ == "__main__":
    app()
