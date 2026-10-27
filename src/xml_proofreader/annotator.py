"""Turn findings into <error> tags placed around the original bytes.

The model reports each error as text, not as a position. This module finds
that text in the paragraph, rejects any finding that cannot be applied safely,
and builds the output from slices of the input. The bytes between the tags are
the input's own bytes, so every paragraph's text stays exactly as it was.
"""

from dataclasses import dataclass
from xml.sax.saxutils import escape

from .findings import Finding
from .locator import Paragraph

EMPTY_ORIGINAL = "the original text is empty"
NO_CHANGE = "the correction is identical to the original"
MISSING_REASON = "a styleguide finding needs a reason"
NOT_FOUND = "the original text is not in the paragraph"
OVERLAPS = "it overlaps an earlier finding"
UNSAFE_SPAN = "it would cross markup, sit inside CDATA or split an entity"

_ATTRIBUTE_ESCAPES = {'"': "&quot;", "\n": "&#10;", "\r": "&#13;", "\t": "&#9;"}


@dataclass(frozen=True)
class Placement:
    """A finding located in the file. Its tags go around raw[start:end]."""

    start: int
    end: int
    finding: Finding


@dataclass(frozen=True)
class Rejection:
    """A finding that was not applied, with its paragraph's line and the reason."""

    line: int
    finding: Finding
    problem: str


@dataclass(frozen=True)
class Annotation:
    """The annotated document, the findings applied, and the findings rejected."""

    output: bytes
    placements: list[Placement]
    rejections: list[Rejection]


def annotate(raw: bytes, paragraphs: list[Paragraph], findings: list[list[Finding]]) -> Annotation:
    """Apply every safe finding to the document. findings[i] belongs to paragraphs[i]."""
    placements: list[Placement] = []
    rejections: list[Rejection] = []
    for paragraph, paragraph_findings in zip(paragraphs, findings, strict=True):
        placed, rejected = place_findings(paragraph, paragraph_findings)
        placements.extend(placed)
        rejections.extend(rejected)
    output = insert_error_tags(raw, placements)
    return Annotation(output=output, placements=placements, rejections=rejections)


def place_findings(
    paragraph: Paragraph, findings: list[Finding]
) -> tuple[list[Placement], list[Rejection]]:
    """Work out where each finding's tags go, and reject the ones that are unsafe.

    The model lists findings in reading order, so each search starts where the
    previous accepted finding ended. That is how a repeated word is matched to
    the right occurrence. If the text is not found after that point, a second
    search from the start catches a finding that was listed out of order.
    """
    text = paragraph.text
    placements: list[Placement] = []
    rejections: list[Rejection] = []
    taken: list[tuple[int, int]] = []
    search_from = 0

    for finding in findings:
        problem = _content_problem(finding)
        if problem is not None:
            rejections.append(Rejection(paragraph.line, finding, problem))
            continue

        char_start = text.find(finding.original, search_from)
        if char_start == -1:
            char_start = text.find(finding.original)
        if char_start == -1:
            rejections.append(Rejection(paragraph.line, finding, NOT_FOUND))
            continue

        char_end = char_start + len(finding.original)
        if any(char_start < end and start < char_end for start, end in taken):
            rejections.append(Rejection(paragraph.line, finding, OVERLAPS))
            continue

        span = paragraph.byte_span(char_start, char_end)
        if span is None:
            rejections.append(Rejection(paragraph.line, finding, UNSAFE_SPAN))
            continue

        placements.append(Placement(start=span[0], end=span[1], finding=finding))
        taken.append((char_start, char_end))
        search_from = char_end

    return placements, rejections


def insert_error_tags(raw: bytes, placements: list[Placement]) -> bytes:
    """Return a copy of raw with an <error> element around every placement.

    The copy is assembled from slices of raw, so no input byte is altered and
    no offset shifts while the copy is being built.
    """
    pieces: list[bytes] = []
    copied_up_to = 0
    for placement in sorted(placements, key=lambda placement: placement.start):
        if placement.start < copied_up_to:
            raise ValueError("Placements overlap, so their error tags would cross.")
        pieces.append(raw[copied_up_to : placement.start])
        pieces.append(_opening_tag(placement.finding))
        pieces.append(raw[placement.start : placement.end])
        pieces.append(b"</error>")
        copied_up_to = placement.end
    pieces.append(raw[copied_up_to:])
    return b"".join(pieces)


def _content_problem(finding: Finding) -> str | None:
    """Say why a finding is unusable on its own terms, or return None if it is fine."""
    if finding.original == "":
        return EMPTY_ORIGINAL
    if finding.correction == finding.original:
        return NO_CHANGE
    if finding.type == "styleguide" and finding.reason.strip() == "":
        return MISSING_REASON
    return None


def _opening_tag(finding: Finding) -> bytes:
    """Build <error type="..." correction="..." reason="...">, leaving out an empty reason."""
    attributes = {"type": finding.type, "correction": finding.correction}
    if finding.reason.strip():
        attributes["reason"] = finding.reason
    rendered = "".join(
        f' {name}="{escape(value, _ATTRIBUTE_ESCAPES)}"' for name, value in attributes.items()
    )
    return f"<error{rendered}>".encode("utf-8")
