"""Independent checks on the corrected document, run before it is written.

The annotator is designed to keep every input byte, but the tool's promise
should not rest on that design alone. These checks look only at the input and
the output, and refuse to let a damaged document through.
"""

import re

from .locator import XmlInputError, find_paragraphs

_ERROR_TAG = re.compile(rb"<error\b[^>]*>|</error>")
_OPENING_ERROR_TAG = re.compile(rb"<error\b[^>]*>")


class VerificationError(Exception):
    """The corrected document failed a check, so it must not be written."""


def verify(original: bytes, corrected: bytes, expected_tags: int) -> None:
    """Raise VerificationError unless corrected is original plus expected_tags error elements.

    Error tags are removed from both documents before they are compared, so an
    input that already contains error tags is still checked correctly.
    """
    try:
        corrected_texts = [p.text for p in find_paragraphs(corrected)]
    except XmlInputError as error:
        raise VerificationError(f"Corrected document does not parse: {error}") from error

    original_texts = [p.text for p in find_paragraphs(original)]
    if corrected_texts != original_texts:
        raise VerificationError("Paragraph text changed between input and output.")

    if _ERROR_TAG.sub(b"", corrected) != _ERROR_TAG.sub(b"", original):
        raise VerificationError("Output differs from the input outside the error tags.")

    added = len(_OPENING_ERROR_TAG.findall(corrected)) - len(_OPENING_ERROR_TAG.findall(original))
    if added != expected_tags:
        raise VerificationError(f"Expected {expected_tags} new error tags, found {added}.")
