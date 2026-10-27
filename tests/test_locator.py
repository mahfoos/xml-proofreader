"""Tests for the paragraph locator.

Most tests build a tiny XML document, locate its paragraphs, and check two
things: the text the model would see, and the exact raw bytes that a pair of
<error> tags would wrap around a phrase.
"""

from pathlib import Path

import pytest

from xml_proofreader.locator import Paragraph, XmlInputError, find_paragraphs

FIXTURES = Path(__file__).parent / "fixtures"
XHTML_DOCTYPE = (
    b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
    b'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
)


def wrapped_bytes(raw: bytes, paragraph: Paragraph, phrase: str) -> bytes | None:
    """The raw bytes that tags around `phrase` would enclose, or None if not allowed."""
    start = paragraph.text.index(phrase)
    span = paragraph.byte_span(start, start + len(phrase))
    if span is None:
        return None
    return raw[span[0] : span[1]]


def only_paragraph(raw: bytes) -> Paragraph:
    """Locate the paragraphs of a document that should contain exactly one."""
    paragraphs = find_paragraphs(raw)
    assert len(paragraphs) == 1
    return paragraphs[0]


# --- The assignment's own files


def test_example_input_has_four_paragraphs_on_the_expected_lines() -> None:
    paragraphs = find_paragraphs((FIXTURES / "example_input.xml").read_bytes())
    assert [p.line for p in paragraphs] == [7, 9, 11, 13]
    assert paragraphs[1].text == (
        "My name is john and I work at the new york office, "
        "where I have been managing client relations for over five years."
    )


def test_sample_input_has_fourteen_paragraphs_and_skips_other_elements() -> None:
    paragraphs = find_paragraphs((FIXTURES / "sample_input.xml").read_bytes())
    assert len(paragraphs) == 14
    all_text = " ".join(p.text for p in paragraphs)
    assert "Legal Technology and Compliance Report" not in all_text
    assert "Guidelines For Compliance Teams" not in all_text


def test_every_word_of_the_sample_maps_back_to_its_own_bytes() -> None:
    raw = (FIXTURES / "sample_input.xml").read_bytes()
    for paragraph in find_paragraphs(raw):
        for word in paragraph.text.split():
            assert wrapped_bytes(raw, paragraph, word) == word.encode("utf-8")


# --- Mapping text back to bytes


def test_a_plain_word_maps_to_its_exact_bytes() -> None:
    raw = b"<d><p>The committe met.</p></d>"
    paragraph = only_paragraph(raw)
    assert paragraph.text == "The committe met."
    assert wrapped_bytes(raw, paragraph, "committe") == b"committe"


def test_offsets_count_bytes_not_characters() -> None:
    raw = "<d><p>Café menu</p></d>".encode()
    paragraph = only_paragraph(raw)
    assert wrapped_bytes(raw, paragraph, "menu") == b"menu"


def test_entities_are_decoded_and_wrapped_whole() -> None:
    raw = b"<d><p>It&apos;s AT&amp;T caf&#233;</p></d>"
    paragraph = only_paragraph(raw)
    assert paragraph.text == "It's AT&T café"
    assert wrapped_bytes(raw, paragraph, "It's") == b"It&apos;s"
    assert wrapped_bytes(raw, paragraph, "AT&T") == b"AT&amp;T"
    assert wrapped_bytes(raw, paragraph, "café") == b"caf&#233;"


def test_a_span_cannot_split_an_entity_that_stands_for_several_characters() -> None:
    raw = b'<!DOCTYPE d [<!ENTITY co "Acme Corp">]><d><p>Ask &co; today</p></d>'
    paragraph = only_paragraph(raw)
    assert paragraph.text == "Ask Acme Corp today"
    assert wrapped_bytes(raw, paragraph, "Acme") is None
    assert wrapped_bytes(raw, paragraph, "Acme Corp") == b"&co;"


def test_a_windows_line_break_is_one_character_but_two_bytes() -> None:
    raw = b"<d><p>one\r\ntwo</p></d>"
    paragraph = only_paragraph(raw)
    assert paragraph.text == "one\ntwo"
    assert wrapped_bytes(raw, paragraph, "one\ntwo") == b"one\r\ntwo"


def test_empty_or_out_of_range_spans_are_rejected() -> None:
    paragraph = only_paragraph(b"<d><p>short</p></d>")
    assert paragraph.byte_span(2, 2) is None
    assert paragraph.byte_span(0, 99) is None


# --- Where an <error> tag may and may not go


def test_text_in_child_elements_is_included_but_a_span_cannot_cross_a_tag() -> None:
    raw = b"<d><p>The <b>committe</b> met</p></d>"
    paragraph = only_paragraph(raw)
    assert paragraph.text == "The committe met"
    assert wrapped_bytes(raw, paragraph, "committe") == b"committe"
    assert wrapped_bytes(raw, paragraph, "The committe") is None


def test_cdata_text_is_read_but_never_wrapped() -> None:
    raw = b"<d><p>See <![CDATA[a < b]]> here</p></d>"
    paragraph = only_paragraph(raw)
    assert paragraph.text == "See a < b here"
    assert wrapped_bytes(raw, paragraph, "a < b") is None
    assert wrapped_bytes(raw, paragraph, "here") == b"here"


def test_a_span_cannot_cross_a_comment() -> None:
    raw = b"<d><p>before<!-- note -->after</p></d>"
    paragraph = only_paragraph(raw)
    assert paragraph.text == "beforeafter"
    assert wrapped_bytes(raw, paragraph, "before") == b"before"
    assert wrapped_bytes(raw, paragraph, "beforeafter") is None


# --- Which elements count as paragraphs


def test_paragraphs_are_found_at_any_depth_and_with_a_namespace_prefix() -> None:
    raw = b'<d xmlns:h="urn:x"><s><div><p>one</p></div></s><h:p>two</h:p></d>'
    assert [p.text for p in find_paragraphs(raw)] == ["one", "two"]


def test_a_nested_paragraph_is_part_of_the_outer_one() -> None:
    raw = b"<d><p>outer <p>inner</p> tail</p></d>"
    assert [p.text for p in find_paragraphs(raw)] == ["outer inner tail"]


def test_empty_paragraphs_are_returned_with_their_text() -> None:
    raw = b"<d><p/><p></p><p>  </p></d>"
    assert [p.text for p in find_paragraphs(raw)] == ["", "", "  "]


def test_markup_inside_comments_and_attribute_values_is_ignored() -> None:
    raw = b'<d note="a>b"><!-- <p>fake</p> --><p>real</p></d>'
    assert [p.text for p in find_paragraphs(raw)] == ["real"]


# --- Bad input


def test_malformed_xml_raises_a_clear_error() -> None:
    with pytest.raises(XmlInputError, match="not well-formed"):
        find_paragraphs(b"<d><p>unclosed</d>")


def test_encodings_other_than_utf8_are_rejected() -> None:
    with pytest.raises(XmlInputError, match="ISO-8859-1"):
        find_paragraphs(b'<?xml version="1.0" encoding="ISO-8859-1"?><d><p>x</p></d>')
    with pytest.raises(XmlInputError, match="UTF-16"):
        find_paragraphs("<d><p>x</p></d>".encode("utf-16"))


def test_an_entity_defined_outside_the_file_is_refused_inside_a_paragraph() -> None:
    raw = XHTML_DOCTYPE + b"<html><body><p>Don&rsquo;t worry.</p></body></html>"
    with pytest.raises(XmlInputError, match="rsquo"):
        find_paragraphs(raw)


def test_an_entity_defined_outside_the_file_is_harmless_outside_paragraphs() -> None:
    raw = XHTML_DOCTYPE + b"<html><body><h1>Caf&eacute;</h1><p>Fine.</p></body></html>"
    assert [p.text for p in find_paragraphs(raw)] == ["Fine."]
