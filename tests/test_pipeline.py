"""Tests for correcting a whole document, with a fake model in place of Claude."""

import asyncio
import logging
from pathlib import Path

import pytest

from xml_proofreader import pipeline
from xml_proofreader.annotator import Annotation
from xml_proofreader.findings import Finding
from xml_proofreader.locator import Paragraph
from xml_proofreader.model_client import ProofreadError
from xml_proofreader.pipeline import Correction, Failure, Proofread, correct_document
from xml_proofreader.verifier import VerificationError

FIXTURES = Path(__file__).parent / "fixtures"
TEH = Finding(original="teh", correction="the", type="spelling", reason="")


def correct(raw: bytes, proofread: Proofread, concurrency: int = 8) -> Correction:
    """Run the pipeline to completion, as the command line does."""
    return asyncio.run(correct_document(raw, proofread, concurrency))


def test_the_example_document_is_corrected_end_to_end(
    example_answers: dict[str, list[Finding]],
) -> None:
    async def proofread(text: str) -> list[Finding]:
        return example_answers[text]

    result = correct((FIXTURES / "example_input.xml").read_bytes(), proofread)
    assert result.output == (FIXTURES / "example_output.xml").read_bytes()
    assert result.failures == []
    assert (result.paragraph_count, result.texts_proofread) == (4, 4)


def test_identical_paragraphs_are_proofread_once() -> None:
    calls = []

    async def proofread(text: str) -> list[Finding]:
        calls.append(text)
        return [TEH]

    result = correct(b"<d><p>teh cat</p><p>teh cat</p></d>", proofread)
    assert calls == ["teh cat"]
    assert result.output.count(b"<error ") == 2


def test_blank_paragraphs_are_not_sent_to_the_model() -> None:
    calls = []

    async def proofread(text: str) -> list[Finding]:
        calls.append(text)
        return []

    correct(b"<d><p>  </p><p/><p>ok</p></d>", proofread)
    assert calls == ["ok"]


def test_a_paragraph_that_fails_is_left_unchanged_and_reported() -> None:
    async def proofread(text: str) -> list[Finding]:
        if "two" in text:
            raise ProofreadError("The model declined this paragraph.")
        return [TEH]

    result = correct(b"<d>\n<p>teh one</p>\n<p>teh two</p>\n</d>", proofread)
    assert result.failures == [Failure(line=3, problem="The model declined this paragraph.")]
    assert (result.texts_proofread, result.texts_failed) == (2, 1)
    assert result.output == (
        b'<d>\n<p><error type="spelling" correction="the">teh</error> one</p>\n<p>teh two</p>\n</d>'
    )


def test_any_other_error_stops_the_run_as_itself_not_as_a_group() -> None:
    async def proofread(text: str) -> list[Finding]:
        if text == "b":
            raise RuntimeError("bad key")
        return []

    with pytest.raises(RuntimeError, match="bad key"):
        correct(b"<d><p>a</p><p>b</p><p>c</p></d>", proofread)


def test_the_first_paragraph_finishes_before_the_rest_start() -> None:
    events = []

    async def proofread(text: str) -> list[Finding]:
        events.append(("start", text))
        await asyncio.sleep(0.01)
        events.append(("end", text))
        return []

    correct(b"<d><p>a</p><p>b</p><p>c</p></d>", proofread)
    assert events[:2] == [("start", "a"), ("end", "a")]


def test_no_more_than_the_concurrency_limit_run_at_once() -> None:
    in_flight = 0
    most_at_once = 0

    async def proofread(text: str) -> list[Finding]:
        nonlocal in_flight, most_at_once
        in_flight += 1
        most_at_once = max(most_at_once, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return []

    raw = b"<d>" + b"".join(b"<p>%d</p>" % number for number in range(10)) + b"</d>"
    correct(raw, proofread, concurrency=3)
    assert most_at_once == 3


def test_a_document_that_fails_verification_is_never_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_annotate = pipeline.annotate

    def damaging_annotate(
        raw: bytes, paragraphs: list[Paragraph], findings: list[list[Finding]]
    ) -> Annotation:
        result = real_annotate(raw, paragraphs, findings)
        return Annotation(
            output=result.output.replace(b"teh", b"the"),
            placements=result.placements,
            rejections=result.rejections,
        )

    monkeypatch.setattr(pipeline, "annotate", damaging_annotate)

    async def proofread(text: str) -> list[Finding]:
        return [TEH]

    with pytest.raises(VerificationError):
        correct(b"<d><p>teh cat</p></d>", proofread)


def test_each_paragraph_is_logged_with_its_line_and_time_at_debug_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def proofread(text: str) -> list[Finding]:
        return [TEH] if "teh" in text else []

    caplog.set_level(logging.DEBUG, logger="xml_proofreader")
    correct(b"<d>\n<p>teh cat</p>\n<p>fine</p>\n</d>", proofread)
    assert "Line 2: 1 findings in" in caplog.text
    assert "Line 3: 0 findings in" in caplog.text


def test_a_concurrency_below_one_is_refused_instead_of_hanging() -> None:
    async def proofread(text: str) -> list[Finding]:
        return []

    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        correct(b"<d><p>a</p></d>", proofread, concurrency=0)
