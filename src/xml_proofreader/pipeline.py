"""Correct one document from start to finish: find, proofread, annotate, verify.

The first paragraph is proofread on its own, so that its request writes the
system prompt into the API's cache. The others are then sent concurrently, up
to a limit, and all of them read the prompt from that cache. Paragraphs with
the same text are proofread once, and blank paragraphs are not sent at all.

A paragraph that cannot be proofread is left unchanged and reported, while the
rest of the document is still corrected. Any other error stops the run.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .annotator import Annotation, annotate
from .findings import Finding
from .locator import find_paragraphs
from .model_client import ProofreadError
from .verifier import verify

Proofread = Callable[[str], Awaitable[list[Finding]]]
Answer = list[Finding] | ProofreadError


@dataclass(frozen=True)
class Failure:
    """A paragraph that could not be proofread, so it was left unchanged."""

    line: int
    problem: str


@dataclass(frozen=True)
class Correction:
    """Everything one run produced."""

    output: bytes
    annotation: Annotation
    failures: list[Failure]
    paragraph_count: int
    texts_proofread: int


async def correct_document(raw: bytes, proofread: Proofread, concurrency: int = 8) -> Correction:
    """Proofread every paragraph of raw and return the verified, annotated document."""
    paragraphs = find_paragraphs(raw)
    texts = list(dict.fromkeys(p.text for p in paragraphs if p.text.strip()))
    answers = await _proofread_all(texts, proofread, concurrency)

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
        texts_proofread=len(texts),
    )


async def _proofread_all(texts: list[str], proofread: Proofread, concurrency: int) -> dict[str, Answer]:
    """Proofread the first text on its own, then the rest concurrently."""
    limit = asyncio.Semaphore(concurrency)

    async def proofread_one(text: str) -> Answer:
        async with limit:
            try:
                return await proofread(text)
            except ProofreadError as error:
                return error

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
