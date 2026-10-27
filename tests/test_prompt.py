"""Tests for the prompt sent to the model."""

from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from xml_proofreader.annotator import annotate
from xml_proofreader.locator import find_paragraphs
from xml_proofreader.prompt import EXAMPLES, Example, build_system_prompt, build_user_message
from xml_proofreader.style_guide import load_style_guide

FIXTURES = Path(__file__).parent / "fixtures"
GUIDE = load_style_guide(FIXTURES / "style_guide.docx")


def test_the_whole_style_guide_is_in_the_system_prompt() -> None:
    prompt = build_system_prompt(GUIDE, "en-US")
    assert f"<style_guide>\n{GUIDE}\n</style_guide>" in prompt


def test_the_system_prompt_is_identical_every_time_so_it_can_be_cached() -> None:
    assert build_system_prompt(GUIDE, "en-GB") == build_system_prompt(GUIDE, "en-GB")


@pytest.mark.parametrize(
    ("tag", "shown_as"),
    [("en", '"en-US"'), ("EN", '"en-US"'), ("en-GB", '"en-GB"'), ("fr-CA", '"fr-CA"')],
)
def test_the_language_tag_is_passed_on_and_bare_en_means_american(tag: str, shown_as: str) -> None:
    assert f"language tag is {shown_as}" in build_system_prompt(GUIDE, tag)


def test_the_paragraph_is_wrapped_with_nothing_added_or_trimmed() -> None:
    assert build_user_message(" a b ") == "<paragraph> a b </paragraph>"


@pytest.mark.parametrize("example", EXAMPLES, ids=["with errors", "quotation"])
def test_every_worked_example_can_be_applied_by_the_annotator(example: Example) -> None:
    raw = f"<d><p>{escape(example.paragraph)}</p></d>".encode()
    result = annotate(raw, find_paragraphs(raw), [example.findings])
    assert result.rejections == []
    assert len(result.placements) == len(example.findings)


def test_the_examples_do_not_reuse_errors_from_the_sample_files() -> None:
    sample_text = " ".join(
        paragraph.text
        for name in ["example_input.xml", "sample_input.xml"]
        for paragraph in find_paragraphs((FIXTURES / name).read_bytes())
    )
    for example in EXAMPLES:
        for finding in example.findings:
            assert finding.original not in sample_text
