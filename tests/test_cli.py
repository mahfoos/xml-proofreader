"""Tests for the command line, with a fake model in place of Claude.

Each test runs in an empty temporary folder, so the project's own .env file,
which may hold a real key, is never read.
"""

import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import pytest

from xml_proofreader import cli
from xml_proofreader.annotator import Annotation, Placement, Rejection
from xml_proofreader.cli import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_PARTIAL,
    collect_metrics,
    main,
    output_path,
)
from xml_proofreader.findings import Finding
from xml_proofreader.model_client import ProofreadError, TokenUsage
from xml_proofreader.pipeline import Correction, Failure

FIXTURES = Path(__file__).parent / "fixtures"
STYLE_GUIDE = str(FIXTURES / "style_guide.docx")


def install_fake_model(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[str], list[Finding]]
) -> None:
    """Replace ClaudeProofreader with a fake whose proofread() calls answer(text)."""

    class FakeProofreader:
        def __init__(
            self,
            client: anthropic.AsyncAnthropic,
            style_guide: str,
            language: str,
            model: str,
            effort: str,
        ) -> None:
            self.usage = TokenUsage()

        async def proofread(self, text: str) -> list[Finding]:
            self.usage.calls += 1
            return answer(text)

    monkeypatch.setattr(cli, "ClaudeProofreader", FakeProofreader)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working folder with a copy of the example input and a dummy API key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    shutil.copy(FIXTURES / "example_input.xml", tmp_path / "input.xml")
    return tmp_path


def test_the_output_goes_next_to_the_input_with_a_corrected_suffix() -> None:
    assert output_path(Path("docs/input.xml")) == Path("docs/input.corrected.xml")


def test_a_full_run_with_a_fake_model_reproduces_the_example_output(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, example_answers: dict[str, list[Finding]]
) -> None:
    install_fake_model(monkeypatch, lambda text: example_answers[text])
    code = main(
        ["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE, "--metrics-json", "m.json"]
    )
    assert code == EXIT_OK
    assert (workspace / "input.corrected.xml").read_bytes() == (
        FIXTURES / "example_output.xml"
    ).read_bytes()
    metrics = json.loads((workspace / "m.json").read_text())
    assert (metrics["paragraphs"], metrics["errors_marked"], metrics["api_calls"]) == (4, 10, 4)
    assert not (workspace / "input.corrected.xml.tmp").exists()


def test_one_failed_paragraph_still_writes_the_output_but_exits_with_3(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    example_answers: dict[str, list[Finding]],
) -> None:
    def answer(text: str) -> list[Finding]:
        if text.startswith("My name is john"):
            raise ProofreadError("The model declined this paragraph.")
        return example_answers[text]

    install_fake_model(monkeypatch, answer)
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_PARTIAL
    output = (workspace / "input.corrected.xml").read_bytes()
    assert b"My name is john and I work at the new york office" in output
    assert b'correction="committee"' in output
    assert "Line 9: paragraph left unchanged" in caplog.text


def test_when_no_paragraph_can_be_proofread_nothing_is_written(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def answer(text: str) -> list[Finding]:
        raise ProofreadError("Could not reach the API: Connection error.")

    install_fake_model(monkeypatch, answer)
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_FAILED
    assert "No paragraph could be proofread" in caplog.text
    assert not (workspace / "input.corrected.xml").exists()


def test_an_unwritable_folder_is_reported_before_any_api_call(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = []
    install_fake_model(monkeypatch, lambda text: calls.append(text) or [])
    locked = workspace / "locked"
    locked.mkdir()
    shutil.copy(workspace / "input.xml", locked / "input.xml")
    locked.chmod(0o500)
    try:
        code = main([str(locked / "input.xml"), "--lang", "en", "--style-guide", STYLE_GUIDE])
    finally:
        locked.chmod(0o700)
    assert code == EXIT_FAILED
    assert "Cannot write to the folder" in caplog.text
    assert calls == []


def test_an_api_error_is_reported_with_the_apis_own_message(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def answer(text: str) -> list[Finding]:
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        body = {"type": "error", "error": {"type": "not_found_error", "message": "model: claude-x"}}
        raise anthropic.NotFoundError(
            "Error code: 404 - raw details",
            response=httpx2.Response(404, request=request),
            body=body,
        )

    install_fake_model(monkeypatch, answer)
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_FAILED
    assert "The API refused the request (404): model: claude-x" in caplog.text
    assert "raw details" not in caplog.text


def test_the_client_gives_up_on_a_connection_after_ten_seconds(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, example_answers: dict[str, list[Finding]]
) -> None:
    install_fake_model(monkeypatch, lambda text: example_answers[text])
    created: list[tuple[anthropic.Timeout, int]] = []
    real_client = anthropic.AsyncAnthropic

    def recording_client(timeout: anthropic.Timeout, max_retries: int) -> anthropic.AsyncAnthropic:
        created.append((timeout, max_retries))
        return real_client(timeout=timeout, max_retries=max_retries)

    monkeypatch.setattr(cli.anthropic, "AsyncAnthropic", recording_client)
    main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE])
    timeout, max_retries = created[0]
    assert (timeout.connect, timeout.read) == (10.0, 120.0)
    assert max_retries == 4


@pytest.mark.parametrize(
    ("level", "library_level"), [("INFO", logging.WARNING), ("DEBUG", logging.DEBUG)]
)
def test_library_messages_appear_only_at_debug_level(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_answers: dict[str, list[Finding]],
    level: str,
    library_level: int,
) -> None:
    install_fake_model(monkeypatch, lambda text: example_answers[text])
    main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE, "--log-level", level])
    assert logging.getLogger("anthropic").level == library_level
    assert logging.getLogger("httpx2").level == library_level


def test_a_missing_api_key_stops_the_run_before_anything_is_written(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    install_fake_model(monkeypatch, lambda text: [])
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_FAILED
    assert "No API key found" in caplog.text
    assert not (workspace / "input.corrected.xml").exists()


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (
            lambda folder: (folder / "input.xml").write_bytes(b"<d><p>unclosed</d>"),
            "not well-formed",
        ),
        (lambda folder: (folder / "input.xml").unlink(), "No such file"),
    ],
    ids=["malformed xml", "missing input"],
)
def test_bad_input_is_reported_and_nothing_is_written(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    setup: Callable[[Path], object],
    message: str,
) -> None:
    setup(workspace)
    install_fake_model(monkeypatch, lambda text: [])
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_FAILED
    assert message in caplog.text
    assert not (workspace / "input.corrected.xml").exists()


def test_a_rejected_api_key_is_reported_without_a_traceback(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
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
    [
        ["--lang", "english!"],
        ["--lang", "en", "--concurrency", "0"],
        ["--lang", "en", "--effort", "huge"],
    ],
    ids=["language tag", "concurrency", "effort"],
)
def test_a_bad_command_line_exits_with_code_2(bad_arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["input.xml", "--style-guide", STYLE_GUIDE, *bad_arguments])
    assert raised.value.code == 2


def test_rejected_findings_are_logged_with_their_line(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    missing = Finding(original="zzz", correction="yyy", type="spelling", reason="")
    install_fake_model(monkeypatch, lambda text: [missing])
    caplog.set_level(logging.WARNING)
    assert main(["input.xml", "--lang", "en", "--style-guide", STYLE_GUIDE]) == EXIT_OK
    assert "Line 7: skipped 'zzz'" in caplog.text


def test_metrics_are_collected_from_the_run_and_the_token_usage() -> None:
    finding = Finding(original="teh", correction="the", type="spelling", reason="")
    annotation = Annotation(
        output=b"",
        placements=[Placement(0, 3, finding), Placement(5, 8, finding)],
        rejections=[Rejection(7, finding, "it overlaps an earlier finding")],
    )
    correction = Correction(
        output=b"",
        annotation=annotation,
        failures=[Failure(line=9, problem="declined")],
        paragraph_count=4,
        texts_proofread=3,
        texts_failed=1,
    )
    usage = TokenUsage(
        calls=3, input_tokens=10, output_tokens=20, cache_read_tokens=30, cache_write_tokens=40
    )
    metrics = collect_metrics(Path("input.xml"), time.perf_counter() - 2.0, correction, usage)
    assert metrics.file == "input.xml"
    assert metrics.seconds >= 2.0
    assert metrics.peak_memory_mb is not None
    assert (metrics.paragraphs, metrics.texts_proofread) == (4, 3)
    assert (metrics.errors_marked, metrics.findings_rejected, metrics.paragraphs_failed) == (
        2,
        1,
        1,
    )
    assert (metrics.api_calls, metrics.input_tokens, metrics.output_tokens) == (3, 10, 20)
    assert (metrics.cache_read_tokens, metrics.cache_write_tokens) == (30, 40)
