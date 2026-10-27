"""Guards the sample files that came with the assignment.

Later tests compare output files byte for byte against these fixtures, so a
fixture that an editor has silently re-saved would make those tests fail for
the wrong reason. These checks catch that first.
"""

import re
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

FIXTURE_NAMES = [
    "example_input.xml",
    "example_output.xml",
    "sample_input.xml",
    "style_guide.docx",
]


def test_all_fixtures_are_present() -> None:
    for name in FIXTURE_NAMES:
        assert (FIXTURES / name).is_file(), f"missing fixture: {name}"


def test_example_output_is_example_input_plus_error_tags() -> None:
    original = (FIXTURES / "example_input.xml").read_bytes()
    annotated = (FIXTURES / "example_output.xml").read_bytes()

    # Delete every opening <error ...> tag and every closing </error> tag.
    stripped = re.sub(rb"<error\b[^>]*>|</error>", b"", annotated)

    assert stripped == original
