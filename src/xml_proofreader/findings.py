"""The shape of one proofreading finding.

The model client fills these in from the model's answer, and the annotator
turns them into <error> tags. Pydantic checks every finding as it is built,
so an unknown error type is refused before it can reach the document.
"""

from typing import Literal

from pydantic import BaseModel, Field

ErrorType = Literal[
    "grammar",
    "spelling",
    "punctuation",
    "capitalization",
    "clarity",
    "styleguide",
]


class Finding(BaseModel):
    """One error found in a paragraph."""

    original: str = Field(
        description="The incorrect text, copied exactly from the paragraph, no longer than needed."
    )
    correction: str = Field(description="The text that should replace the original.")
    type: ErrorType = Field(description="The category of the error.")
    reason: str = Field(
        description="A short explanation. For styleguide errors, the style guide rule broken."
    )


class ParagraphFindings(BaseModel):
    """Every error found in one paragraph."""

    findings: list[Finding] = Field(
        description="The errors in reading order. An empty list if the paragraph has none."
    )
