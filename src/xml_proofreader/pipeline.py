"""Correct one document from start to finish: find, proofread, annotate, verify.

The first paragraph is proofread on its own, so that its request writes the
system prompt into the API's cache. The others are then sent concurrently, up
to a limit, and all of them read the prompt from that cache. Paragraphs with
the same text are proofread once, and blank paragraphs are not sent at all.

A paragraph that cannot be proofread is left unchanged and reported, while the
rest of the document is still corrected. Any other error stops the run.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .annotator import Annotation, annotate
from .findings import Finding
from .locator import find_paragraphs
from .model_client import ProofreadError
from .verifier import verify

log = logging.getLogger(__name__)

Proofread = Callable[[str], Awaitable[list[Finding]]]
Answer = list[Finding] | ProofreadError


@dataclass(frozen=True)
class Failure:
    """A paragraph that could not be proofread, so it was left unchanged.

    Attributes:
        line: Line number of the paragraph's <p> start tag.
        problem: Why it could not be proofread.
    """

    line: int
    problem: str


@dataclass(frozen=True)
class Correction:
    """Everything one run produced.

    Attributes:
        output: The verified, annotated document.
        annotation: The findings applied and the findings refused.
        failures: The paragraphs left unchanged because they failed.
        paragraph_count: How many <p> elements the document has.
        texts_proofread: How many distinct, non-blank texts were sent to the model.
        texts_failed: How many of those texts could not be proofread.
    """

    output: bytes
    annotation: Annotation
    failures: list[Failure]
    paragraph_count: int
    texts_proofread: int
    texts_failed: int


async def correct_document(raw: bytes, proofread: Proofread, concurrency: int = 8) -> Correction:
    """Proofread every paragraph of a document and annotate its errors.

    Args:
        raw: The original document bytes.
        proofread: The function that proofreads one paragraph's text.
        concurrency: The most paragraphs proofread at the same time.

    Returns:
        The verified, annotated document and an account of the run.

    Raises:
        ValueError: If concurrency is less than 1, which would otherwise wait
            forever.
        XmlInputError: If the document is not well-formed UTF-8 XML.
        VerificationError: If the annotated document fails its checks.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    paragraphs = find_paragraphs(raw)
    first_lines: dict[str, int] = {}
    for paragraph in paragraphs:
        if paragraph.text.strip():
            first_lines.setdefault(paragraph.text, paragraph.line)
    answers = await _proofread_all(first_lines, proofread, concurrency)

    findings: list[list[Finding]] = []
    failures: list[Failure] = []
    for paragraph in paragraphs:
        answer = answers.get(paragraph.text, [])
        if isinstance(answer, ProofreadError):
            failures.append(Failure(line=paragraph.line, problem=str(answer)))
            answer = []
        findings.append(answer)

    annotation = annotate(raw, paragraphs, findings)
    verify(raw, annotation.output, expected_tags=len(annotation.placements))
    return Correction(
        output=annotation.output,
        annotation=annotation,
        failures=failures,
        paragraph_count=len(paragraphs),
        texts_proofread=len(first_lines),
        texts_failed=sum(isinstance(answer, ProofreadError) for answer in answers.values()),
    )


async def _proofread_all(
    first_lines: dict[str, int], proofread: Proofread, concurrency: int
) -> dict[str, Answer]:
    """Proofread the first text on its own, then the rest concurrently.

    Args:
        first_lines: Each distinct text, in document order, mapped to the
            line of its first paragraph, so the debug log can say where
            each text is.
        proofread: The function that proofreads one paragraph's text.
        concurrency: The most texts proofread at the same time.

    Returns:
        Each text mapped to its findings, or to the ProofreadError that
        stopped it.

    Raises:
        Exception: Any error from proofread other than a ProofreadError,
            such as a rejected API key. It is taken out of the task
            group's ExceptionGroup, so callers see the original error.
    """
    limit = asyncio.Semaphore(concurrency)

    async def proofread_one(text: str) -> Answer:
        """Proofread one text, returning a ProofreadError instead of raising it.

        Args:
            text: The paragraph text to proofread.

        Returns:
            The findings, or the ProofreadError, so that one failed
            paragraph cannot cancel the others.
        """
        async with limit:
            started = time.perf_counter()
            try:
                findings = await proofread(text)
            except ProofreadError as error:
                return error
            seconds = time.perf_counter() - started
            log.debug("Line %d: %d findings in %.1f s", first_lines[text], len(findings), seconds)
            return findings

    texts = list(first_lines)
    if not texts:
        return {}
    answers = {texts[0]: await proofread_one(texts[0])}
    try:
        async with asyncio.TaskGroup() as group:
            tasks = {text: group.create_task(proofread_one(text)) for text in texts[1:]}
    except ExceptionGroup as errors:
        raise errors.exceptions[0] from None
    answers.update({text: task.result() for text, task in tasks.items()})
    return answers
