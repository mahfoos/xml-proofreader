"""Tests for the checks that run before a corrected document is written."""

from pathlib import Path

import pytest

from xml_proofreader.verifier import VerificationError, verify

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE_INPUT = (FIXTURES / "example_input.xml").read_bytes()
EXAMPLE_OUTPUT = (FIXTURES / "example_output.xml").read_bytes()


def test_the_example_output_from_the_brief_passes() -> None:
    verify(EXAMPLE_INPUT, EXAMPLE_OUTPUT, expected_tags=10)


def test_a_wrong_tag_count_fails() -> None:
    with pytest.raises(VerificationError, match="Expected 9"):
        verify(EXAMPLE_INPUT, EXAMPLE_OUTPUT, expected_tags=9)


def test_changed_text_inside_a_tag_fails() -> None:
    damaged = EXAMPLE_OUTPUT.replace(b">committe<", b">committee<")
    with pytest.raises(VerificationError, match="Paragraph text changed"):
        verify(EXAMPLE_INPUT, damaged, expected_tags=10)


def test_a_change_outside_the_paragraphs_fails() -> None:
    damaged = EXAMPLE_OUTPUT.replace(b'id="sec-1"', b'id="sec-2"')
    with pytest.raises(VerificationError, match="outside the error tags"):
        verify(EXAMPLE_INPUT, damaged, expected_tags=10)


def test_a_tag_inside_an_entity_fails_even_though_stripping_tags_would_hide_it() -> None:
    original = b"<d><p>AT&amp;T</p></d>"
    damaged = b'<d><p>AT&am<error type="spelling" correction="x">p;</error>T</p></d>'
    with pytest.raises(VerificationError, match="does not parse"):
        verify(original, damaged, expected_tags=1)


def test_an_input_that_already_has_error_tags_is_checked_correctly() -> None:
    with_one_more = EXAMPLE_OUTPUT.replace(
        b"The quarterly report",
        b'The <error type="spelling" correction="x">quarterly</error> report',
    )
    verify(EXAMPLE_OUTPUT, with_one_more, expected_tags=1)
