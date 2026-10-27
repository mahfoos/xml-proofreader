"""Tests for turning findings into <error> tags."""

from pathlib import Path
from xml.parsers import expat

import pytest
from pydantic import ValidationError

from xml_proofreader.annotator import (
    EMPTY_ORIGINAL,
    MISSING_REASON,
    NO_CHANGE,
    NOT_FOUND,
    OVERLAPS,
    UNSAFE_SPAN,
    Annotation,
    Placement,
    annotate,
    insert_error_tags,
)
from xml_proofreader.findings import Finding
from xml_proofreader.locator import find_paragraphs
from xml_proofreader.verifier import verify

FIXTURES = Path(__file__).parent / "fixtures"


def finding(original: str, correction: str, error_type: str = "spelling", reason: str = "") -> Finding:
    """Build a Finding with short positional arguments, to keep the tests readable."""
    return Finding(original=original, correction=correction, type=error_type, reason=reason)


def annotate_one_paragraph(raw: bytes, findings: list[Finding]) -> Annotation:
    """Annotate a document that contains exactly one paragraph."""
    return annotate(raw, find_paragraphs(raw), [findings])


EXAMPLE_FINDINGS = [
    [
        finding("committe", "committee"),
        finding("untill", "until"),
        finding("wednesday", "Wednesday", "capitalization"),
        finding("availble", "available"),
    ],
    [
        finding("john", "John", "capitalization"),
        finding("new york", "New York", "capitalization"),
    ],
    [
        finding("$2,000,000", "$2 million", "styleguide", "Spell out million, billion, etc."),
        finding("January, 2024", "January 2024", "styleguide", "No comma between month and year"),
        finding("it's", "its", "punctuation", "Possessive its has no apostrophe"),
        finding("publicly-owned", "publicly owned", "styleguide", "Do not hyphenate ly adverbs"),
    ],
    [],
]


# --- Whole documents


def test_the_example_output_is_reproduced_byte_for_byte():
    raw = (FIXTURES / "example_input.xml").read_bytes()
    result = annotate(raw, find_paragraphs(raw), EXAMPLE_FINDINGS)
    assert result.rejections == []
    assert result.output == (FIXTURES / "example_output.xml").read_bytes()


def test_wrapping_every_word_of_the_sample_keeps_the_document_intact():
    raw = (FIXTURES / "sample_input.xml").read_bytes()
    paragraphs = find_paragraphs(raw)
    findings = [[finding(word, word + "x") for word in p.text.split()] for p in paragraphs]
    result = annotate(raw, paragraphs, findings)
    assert result.rejections == []
    assert len(result.placements) == sum(len(words) for words in findings)
    verify(raw, result.output, expected_tags=len(result.placements))


# --- Finding the right text


def test_a_repeated_word_is_matched_to_successive_occurrences():
    raw = b"<d><p>teh cat and teh dog</p></d>"
    result = annotate_one_paragraph(raw, [finding("teh", "the"), finding("teh", "the")])
    assert result.output == (
        b'<d><p><error type="spelling" correction="the">teh</error> cat and '
        b'<error type="spelling" correction="the">teh</error> dog</p></d>'
    )


def test_findings_listed_out_of_order_are_still_placed():
    raw = b"<d><p>The committe met untill noon.</p></d>"
    result = annotate_one_paragraph(raw, [finding("untill", "until"), finding("committe", "committee")])
    assert result.rejections == []
    assert [raw[p.start : p.end] for p in result.placements] == [b"untill", b"committe"]


def test_text_written_with_an_entity_is_wrapped_as_its_raw_bytes():
    raw = b"<d><p>It&apos;s a test.</p></d>"
    result = annotate_one_paragraph(raw, [finding("It's", "It is", "clarity", "Avoid contractions")])
    assert result.output == (
        b'<d><p><error type="clarity" correction="It is" reason="Avoid contractions">'
        b"It&apos;s</error> a test.</p></d>"
    )


# --- Writing the attributes


def test_attribute_values_survive_a_round_trip_through_an_xml_parser():
    raw = b"<d><p>Use x here.</p></d>"
    tricky = 'say "a < b" & mean a > b\nnext line'
    result = annotate_one_paragraph(raw, [finding("x", tricky, "clarity", tricky)])

    seen: dict[str, str] = {}

    def remember_error_attributes(name: str, attributes: dict[str, str]) -> None:
        if name == "error":
            seen.update(attributes)

    parser = expat.ParserCreate()
    parser.StartElementHandler = remember_error_attributes
    parser.Parse(result.output, True)
    assert seen["correction"] == tricky
    assert seen["reason"] == tricky
    verify(raw, result.output, expected_tags=1)


# --- Findings that must not be applied


@pytest.mark.parametrize(
    ("bad_finding", "problem"),
    [
        (finding("", "x"), EMPTY_ORIGINAL),
        (finding("met", "met"), NO_CHANGE),
        (finding("met", "meet", "styleguide", ""), MISSING_REASON),
        (finding("absent", "present"), NOT_FOUND),
        (finding("The committe", "The committee"), UNSAFE_SPAN),
    ],
)
def test_an_unusable_finding_is_rejected_and_the_document_is_unchanged(bad_finding, problem):
    raw = b"<d><p>The <b>committe</b> met.</p></d>"
    result = annotate_one_paragraph(raw, [bad_finding])
    assert [r.problem for r in result.rejections] == [problem]
    assert result.output == raw


def test_a_finding_that_overlaps_an_earlier_one_is_rejected():
    raw = b"<d><p>The committe met.</p></d>"
    first = finding("committe", "committee")
    second = finding("committe met", "committee met")
    result = annotate_one_paragraph(raw, [first, second])
    assert [p.finding for p in result.placements] == [first]
    assert [r.problem for r in result.rejections] == [OVERLAPS]


def test_overlapping_placements_are_refused_rather_than_crossed():
    one = finding("ab", "x")
    with pytest.raises(ValueError, match="overlap"):
        insert_error_tags(b"abc", [Placement(0, 2, one), Placement(1, 3, one)])


def test_each_paragraph_needs_its_own_list_of_findings():
    raw = b"<d><p>one</p><p>two</p></d>"
    with pytest.raises(ValueError):
        annotate(raw, find_paragraphs(raw), [[]])


def test_an_unknown_error_type_is_refused_when_the_finding_is_built():
    with pytest.raises(ValidationError):
        Finding(original="x", correction="y", type="typo", reason="")
