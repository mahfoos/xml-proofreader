"""Tests for the command line, with a fake model in place of Claude.

Each test runs in an empty temporary folder, so the project's own .env file,
which may hold a real key, is never read.
"""

import json
import logging
import shutil
from pathlib import Path

import anthropic
import httpx2
import pytest

from xml_proofreader import cli
from xml_proofreader.cli import EXIT_FAILED, EXIT_OK, EXIT_PARTIAL, main, output_path
from xml_proofreader.findings import Finding
from xml_proofreader.model_client import ProofreadError, TokenUsage

FIXTURES = Path(__file__).parent / "fixtures"
STYLE_GUIDE = str(FIXTURES / "style_guide.docx")


def install_fake_model(monkeypatch, answer) -> None:
    """Replace ClaudeProofreader with a fake whose proofread() calls answer(text)."""

    class FakeProofreader:
        def __init__(self, client, style_guide, language, model, effort) -> None:
            self.usage = TokenUsage()

        async def proofread(self, text: str) -> list[Finding]:
            self.usage.calls += 1
            return answer(text)

    monkeypatch.setattr(cli, "ClaudeProofreader", FakeProofreader)


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    """An empty working folder with a copy of the example input and a dummy API key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    shutil.copy(FIXTURES / "example_input.xml", tmp_path / "input.xml")
    return tmp_path


def test_the_output_goes_next_to_the_input_with_a_corrected_suffix():
    assert output_path(Path("docs/input.xml")) == Path("docs/input.corrected.xml")


def test_a_full_run_with_a_fake_model_reproduces_the_example_output(workspace, monkeypatch, example_answers):
    install_fake_model(monkeypatch, lambda text: example_answers[text])
    code = main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE, "--metrics-json", "m.json"])
    assert code == EXIT_OK
    assert (workspace / "input.corrected.xml").read_bytes() == (FIXTURES / "example_output.xml").read_bytes()
    metrics = json.loads((workspace / "m.json").read_text())
    assert (metrics["paragraphs"], metrics["errors_marked"], metrics["api_calls"]) == (4, 10, 4)
    assert not (workspace / "input.corrected.xml.tmp").exists()


def test_a_failed_paragraph_still_writes_the_output_but_exits_with_3(workspace, monkeypatch):
    def answer(text: str) -> list[Finding]:
        raise ProofreadError("The model declined this paragraph.")

    install_fake_model(monkeypatch, answer)
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_PARTIAL
    assert (workspace / "input.corrected.xml").read_bytes() == (workspace / "input.xml").read_bytes()


def test_a_missing_api_key_stops_the_run_before_anything_is_written(workspace, monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    install_fake_model(monkeypatch, lambda text: [])
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_FAILED
    assert "No API key found" in caplog.text
    assert not (workspace / "input.corrected.xml").exists()


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda folder: (folder / "input.xml").write_bytes(b"<d><p>unclosed</d>"), "not well-formed"),
        (lambda folder: (folder / "input.xml").unlink(), "No such file"),
    ],
    ids=["malformed xml", "missing input"],
)
def test_bad_input_is_reported_and_nothing_is_written(workspace, monkeypatch, caplog, setup, message):
    setup(workspace)
    install_fake_model(monkeypatch, lambda text: [])
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_FAILED
    assert message in caplog.text
    assert not (workspace / "input.corrected.xml").exists()


def test_a_rejected_api_key_is_reported_without_a_traceback(workspace, monkeypatch, caplog):
    def answer(text: str) -> list[Finding]:
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
        )

    install_fake_model(monkeypatch, answer)
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_FAILED
    assert "The API rejected the key" in caplog.text


@pytest.mark.parametrize(
    "bad_arguments",
    [["--lang", "english!"], ["--lang", "en", "--concurrency", "0"], ["--lang", "en", "--effort", "huge"]],
    ids=["language tag", "concurrency", "effort"],
)
def test_a_bad_command_line_exits_with_code_2(bad_arguments):
    with pytest.raises(SystemExit) as raised:
        main(["input.xml", "--style-guide", STYLE_GUIDE, *bad_arguments])
    assert raised.value.code == 2


def test_rejected_findings_are_logged_with_their_line(workspace, monkeypatch, caplog):
    missing = Finding(original="zzz", correction="yyy", type="spelling", reason="")
    install_fake_model(monkeypatch, lambda text: [missing])
    caplog.set_level(logging.WARNING)
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_OK
    assert "Line 7: skipped 'zzz'" in caplog.text
