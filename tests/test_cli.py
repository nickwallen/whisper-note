from typer.testing import CliRunner
from cli import app

runner = CliRunner()


def test_cli_query_shows_answer_and_context(monkeypatch):
    class MockResp:
        answer = "42"
        context = [
            type("Chunk", (), {"text": "Context chunk 1"})(),
            type("Chunk", (), {"text": "Context chunk 2"})(),
        ]

    monkeypatch.setattr("cli.submit_post_query", lambda payload: MockResp())
    result = runner.invoke(app, ["query", "What is the answer?", "--debug"])
    assert result.exit_code == 0
    assert "42" in result.output
    assert "Context chunk 1" in result.output
    assert "Context chunk 2" in result.output


def test_cli_chat_handles_exit(monkeypatch):
    monkeypatch.setattr("cli.Console.input", lambda self, prompt: "q")
    monkeypatch.setattr(
        "cli.submit_post_query",
        lambda payload: type("MockResp", (), {"answer": "", "context": []})(),
    )
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 0
    assert "Exiting chat." in result.output


def test_cli_status_shows_index_metrics(monkeypatch):
    class MockMetrics:
        file_count = 5
        chunk_count = 10
        failed_files = [{"file": "foo.txt", "error": "fail"}]

    monkeypatch.setattr("cli.submit_get_index", lambda: MockMetrics())
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Indexed files" in result.output
    assert "5" in result.output
    assert "Indexed chunks" in result.output
    assert "10" in result.output
    assert "Failed files" in result.output


def test_cli_index_shows_index_metrics(monkeypatch, tmp_path):
    class MockMetrics:
        file_count = 2
        chunk_count = 4
        failed_files = []

    monkeypatch.setattr(
        "cli.submit_post_index", lambda directory, file_extensions: MockMetrics()
    )
    result = runner.invoke(app, ["index", str(tmp_path)])
    assert result.exit_code == 0
    assert "Indexed files" in result.output
    assert "2" in result.output
    assert "Indexed chunks" in result.output
    assert "4" in result.output
