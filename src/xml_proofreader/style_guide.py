"""Read the style guide .docx into plain text that the model can follow.

Word stores a document's structure in paragraph styles, so the loader keeps
that structure visible: heading styles become Markdown headings, list styles
become bullets, fully bold paragraphs become bold labels, and each table row
becomes one line with its cells separated by " | ". Everything else becomes a
plain line of text. Empty paragraphs are dropped.
"""

import zipfile
from pathlib import Path

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph


class StyleGuideError(Exception):
    """The style guide could not be read, or it contains no text."""


def load_style_guide(path: Path) -> str:
    """Read a style guide .docx into text, one line per paragraph or table row.

    Args:
        path: The style guide file.

    Returns:
        The style guide as text, with headings, bullets, bold labels and
        table rows marked as the module docstring describes.

    Raises:
        StyleGuideError: If the file is missing, is not a Word document, or
            contains no text.
    """
    try:
        document = docx.Document(str(path))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, OSError, ValueError) as error:
        raise StyleGuideError(f"Cannot read the style guide {path}: {error}") from error

    lines: list[str] = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            lines.extend(_table_lines(block))
        else:
            line = _paragraph_line(block)
            if line:
                lines.append(line)

    if not lines:
        raise StyleGuideError(f"The style guide {path} contains no text.")
    return "\n".join(lines).strip()


def _paragraph_line(paragraph: Paragraph) -> str:
    """Render one Word paragraph as a line of text.

    Args:
        paragraph: The Word paragraph.

    Returns:
        The paragraph's text, marked as a heading, bullet or bold label when
        its style says so, or "" if the paragraph is blank.
    """
    text = paragraph.text.strip()
    if not text:
        return ""
    style = _style_name(paragraph)
    level = _heading_level(style)
    if level is not None:
        return "\n" + "#" * (level + 1) + " " + text
    if "list" in style.lower():
        return "- " + text
    if _is_bold(paragraph):
        return "**" + text + "**"
    return text


def _style_name(paragraph: Paragraph) -> str:
    """Get a paragraph's style name safely.

    Args:
        paragraph: The Word paragraph.

    Returns:
        The style's name, or "" when the paragraph has no style or the style
        has no name.
    """
    if paragraph.style is None or paragraph.style.name is None:
        return ""
    return paragraph.style.name


def _heading_level(style: str) -> int | None:
    """Work out whether a style is a title or heading, and at what level.

    Built-in styles are named "Title" and "Heading 1", "Heading 2" and so on.
    Custom styles such as "PLI_Level 1 Head" follow the same idea, so any
    style whose name contains "head" counts, and its first digit is its level.

    Args:
        style: The style's name.

    Returns:
        0 for a title, the heading's level for a heading, or None for any
        other style.
    """
    name = style.lower()
    if name == "title":
        return 0
    if "head" not in name:
        return None
    digits = [character for character in name if character.isdigit()]
    return int(digits[0]) if digits else 1


def _is_bold(paragraph: Paragraph) -> bool:
    """Check whether a paragraph is entirely bold, as in a hand-made heading.

    Args:
        paragraph: The Word paragraph.

    Returns:
        True when every run with visible text is bold.
    """
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(run.bold for run in runs)


def _table_lines(table: Table) -> list[str]:
    """Render a Word table as lines of text.

    Args:
        table: The Word table.

    Returns:
        One line per non-empty row, with its cell texts joined by " | ".
    """
    lines = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return lines
