"""Turn findings into <error> tags placed around the original bytes.

The model reports each error as text, not as a position. This module finds
that text in the paragraph, rejects any finding that cannot be applied safely,
and builds the output from slices of the input. The bytes between the tags are
the input's own bytes, so every paragraph's text stays exactly as it was.
"""

import re
from dataclasses import dataclass
from xml.sax.saxutils import escape

from .findings import Finding
from .locator import Paragraph

EMPTY_ORIGINAL = "the original text is empty"
NO_CHANGE = "the correction is identical to the original"
MISSING_REASON = "a styleguide finding needs a reason"
BAD_CHARACTER = "the correction or reason has a character that XML does not allow"
NOT_FOUND = "the original text is not in the paragraph"
OVERLAPS = "it overlaps an earlier finding"
UNSAFE_SPAN = "it would cross markup, sit inside CDATA or split an entity"

_ATTRIBUTE_ESCAPES = {'"': "&quot;", "\n": "&#10;", "\r": "&#13;", "\t": "&#9;"}
# XML 1.0 allows tab, line feed, carriage return and every character from the
# space upward, except the surrogate range and the two characters U+FFFE and U+FFFF.
_NOT_ALLOWED_IN_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


@dataclass(frozen=True)
class Placement:
    """A finding that has been located in the file.

    Attributes:
        start: Byte offset where the opening tag goes.
        end: Byte offset where the closing tag goes.
        finding: The finding whose tags go around raw[start:end].
    """

    start: int
    end: int
    finding: Finding


@dataclass(frozen=True)
class Rejection:
    """A finding that was not applied.

    Attributes:
        line: Line number of the paragraph the finding belongs to.
        finding: The finding that was refused.
        problem: Why it was refused, as one of this module's message constants.
    """

    line: int
    finding: Finding
    problem: str


@dataclass(frozen=True)
class Annotation:
    """The annotated document, with the findings applied and those refused.

    Attributes:
        output: The document with every accepted finding wrapped in tags.
        placements: The findings that were applied.
        rejections: The findings that were refused, with reasons.
    """

    output: bytes
    placements: list[Placement]
    rejections: list[Rejection]


def annotate(raw: bytes, paragraphs: list[Paragraph], findings: list[list[Finding]]) -> Annotation:
    """Apply every safe finding to the document.

    Args:
        raw: The original document bytes.
        paragraphs: The document's paragraphs, from find_paragraphs.
        findings: One list of findings per paragraph, in the same order, so
            findings[i] belongs to paragraphs[i]. A blank paragraph gets an
            empty list.

    Returns:
        The annotated document, with what was applied and what was refused.

    Raises:
        ValueError: If paragraphs and findings have different lengths.
    """
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

    The model lists findings in reading order, so a match after the previous
    accepted finding is preferred. That is how a repeated word is matched to
    the right occurrence. _locate describes every rule for choosing a match.

    Args:
        paragraph: The paragraph the findings belong to.
        findings: The model's findings for that paragraph, in reading order.

    Returns:
        The findings that can be applied, as placements, and the findings
        that cannot, as rejections with reasons.
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

        char_start = _locate(text, finding.original, search_from)
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
    """Wrap an <error> element around every placement.

    The copy is assembled from slices of raw, so no input byte is altered and
    no offset shifts while the copy is being built.

    Args:
        raw: The original document bytes.
        placements: Where each pair of tags goes, in any order.

    Returns:
        A new copy of the document with the tags inserted.

    Raises:
        ValueError: If two placements overlap, because their tags would cross
            and break the XML.
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
    """Check a finding on its own terms, before searching the text for it.

    Args:
        finding: The finding to check.

    Returns:
        The reason the finding is unusable, or None if it looks fine.
    """
    if finding.original == "":
        return EMPTY_ORIGINAL
    if finding.correction == finding.original:
        return NO_CHANGE
    if finding.type == "styleguide" and finding.reason.strip() == "":
        return MISSING_REASON
    if _NOT_ALLOWED_IN_XML.search(finding.correction + finding.reason):
        return BAD_CHARACTER
    return None


def _locate(text: str, original: str, search_from: int) -> int:
    """Choose where a finding's original text sits in the paragraph.

    Matches that begin and end at the edges of words come first, so "its" is
    not found inside "Credits". A match inside a longer word is used only when
    there is no other, which keeps languages written without spaces working.
    Within each group, the first match after search_from wins, because the
    model lists findings in reading order. Failing that, the first match
    anywhere wins, which catches a finding listed out of order.

    Args:
        text: The paragraph's text.
        original: The finding's original text. Never empty.
        search_from: Where the previous accepted finding ended.

    Returns:
        The index where original starts in text, or -1 if it is not there.
    """
    matches = [index for index in range(len(text)) if text.startswith(original, index)]
    whole_word_matches = [
        index for index in matches if not _cuts_a_word(text, index, index + len(original))
    ]
    for candidates in (whole_word_matches, matches):
        later = [index for index in candidates if index >= search_from]
        if later:
            return later[0]
        if candidates:
            return candidates[0]
    return -1


def _cuts_a_word(text: str, start: int, end: int) -> bool:
    """Tell whether a span of text begins or ends in the middle of a word.

    Args:
        text: The paragraph's text.
        start: Index of the span's first character.
        end: Index just past the span's last character.

    Returns:
        True if a letter or digit at either edge of the span has another
        letter or digit right beside it, outside the span, as "its" does
        inside "Credits".
    """
    cuts_at_start = start > 0 and text[start - 1].isalnum() and text[start].isalnum()
    cuts_at_end = end < len(text) and text[end - 1].isalnum() and text[end].isalnum()
    return cuts_at_start or cuts_at_end


def _opening_tag(finding: Finding) -> bytes:
    """Build the opening <error> tag for a finding.

    Attributes are written in the order type, correction, reason, as in the
    brief's example output, and an empty reason is left out.

    Args:
        finding: The finding to describe.

    Returns:
        The tag as UTF-8 bytes, with every attribute value escaped.
    """
    attributes = {"type": finding.type, "correction": finding.correction}
    if finding.reason.strip():
        attributes["reason"] = finding.reason
    rendered = "".join(
        f' {name}="{escape(value, _ATTRIBUTE_ESCAPES)}"' for name, value in attributes.items()
    )
    return f"<error{rendered}>".encode()
