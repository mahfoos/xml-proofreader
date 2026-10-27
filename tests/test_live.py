"""One real call to the Claude API, skipped unless you ask for it.

It costs a few cents, so it only runs when XML_PROOFREADER_LIVE=1 is set as
well as ANTHROPIC_API_KEY. Run it with -s to see the model's answer:

    XML_PROOFREADER_LIVE=1 pytest tests/test_live.py -s
"""

import asyncio
import os
from pathlib import Path

import anthropic
import pytest

from xml_proofreader.findings import ParagraphFindings
from xml_proofreader.model_client import ClaudeProofreader, TokenUsage
from xml_proofreader.style_guide import load_style_guide

FIXTURES = Path(__file__).parent / "fixtures"
PARAGRAPHS = [
    "The committe has decided to postpone the meeting untill next wednesday, "
    "when all members will be availble to attend.",
    "The quarterly report was submitted on time and all metrics exceeded expectations.",
]


@pytest.mark.skipif(
    os.environ.get("XML_PROOFREADER_LIVE") != "1",
    reason="costs money: set XML_PROOFREADER_LIVE=1 and ANTHROPIC_API_KEY to run",
)
def test_a_real_call_finds_an_obvious_misspelling_and_reuses_the_cached_prompt() -> None:
    guide = load_style_guide(FIXTURES / "style_guide.docx")

    async def run() -> tuple[list[ParagraphFindings], TokenUsage]:
        async with anthropic.AsyncAnthropic() as client:
            proofreader = ClaudeProofreader(client, guide, "en")
            answers = []
            for paragraph in PARAGRAPHS:
                answers.append(ParagraphFindings(findings=await proofreader.proofread(paragraph)))
            return answers, proofreader.usage

    answers, usage = asyncio.run(run())
    for paragraph, answer in zip(PARAGRAPHS, answers, strict=True):
        print(f"\nPARAGRAPH: {paragraph}\nANSWER: {answer.model_dump_json(indent=2)}")
    print(f"\nUSAGE: {usage}")

    first = answers[0].findings
    assert any(f.original == "committe" and f.correction == "committee" for f in first)
    assert usage.cache_read_tokens > 0
