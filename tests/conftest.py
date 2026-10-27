"""Shared test data.

pytest loads this file automatically, and any test that names one of these
fixtures as a parameter receives its value.
"""

from pathlib import Path

import pytest

from xml_proofreader.findings import ErrorType, Finding
from xml_proofreader.locator import find_paragraphs

FIXTURES = Path(__file__).parent / "fixtures"


def _finding(original: str, correction: str, error_type: ErrorType, reason: str = "") -> Finding:
    return Finding(original=original, correction=correction, type=error_type, reason=reason)


EXAMPLE_FINDINGS = [
    [
        _finding("committe", "committee", "spelling"),
        _finding("untill", "until", "spelling"),
        _finding("wednesday", "Wednesday", "capitalization"),
        _finding("availble", "available", "spelling"),
    ],
    [
        _finding("john", "John", "capitalization"),
        _finding("new york", "New York", "capitalization"),
    ],
    [
        _finding("$2,000,000", "$2 million", "styleguide", "Spell out million, billion, etc."),
        _finding("January, 2024", "January 2024", "styleguide", "No comma between month and year"),
        _finding("it's", "its", "punctuation", "Possessive its has no apostrophe"),
        _finding("publicly-owned", "publicly owned", "styleguide", "Do not hyphenate ly adverbs"),
    ],
    [],
]


@pytest.fixture
def example_findings() -> list[list[Finding]]:
    """The errors in example_input.xml, one list per paragraph, as example_output.xml marks them."""
    return EXAMPLE_FINDINGS


@pytest.fixture
def example_answers() -> dict[str, list[Finding]]:
    """The same findings, keyed by the text of the paragraph they belong to."""
    raw = (FIXTURES / "example_input.xml").read_bytes()
    texts = [paragraph.text for paragraph in find_paragraphs(raw)]
    return dict(zip(texts, EXAMPLE_FINDINGS, strict=True))
