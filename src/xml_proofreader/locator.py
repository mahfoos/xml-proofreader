"""Find the text of every <p> element in an XML file, with exact byte offsets.

The output file must be byte-for-byte identical to the input apart from the
<error> tags we add, so this module never rebuilds the document from a parsed
tree. It uses expat, the streaming XML parser in Python's standard library.
Expat reports the byte offset where every tag and every piece of text begins,
and later steps use those offsets to insert tags into the original bytes.

Expat delivers text in chunks, splitting at every entity reference and line
break: "It&apos;s" arrives as "It", "'" and "s". Each chunk remembers where its
raw bytes sit in the file, which is how a span of text is mapped back to bytes.
"""

import codecs
from dataclasses import dataclass, field
from xml.parsers import expat


class XmlInputError(Exception):
    """The input is not well-formed XML, or is not encoded as UTF-8."""


@dataclass
class Chunk:
    """One piece of paragraph text, exactly as expat reported it."""

    text: str
    start: int
    run: int
    in_cdata: bool
    end: int = 0
    plain: bool = False


@dataclass
class Paragraph:
    """One <p> element and its text, including text inside child elements."""

    line: int
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The paragraph's full text: what the model proofreads."""
        return "".join(chunk.text for chunk in self.chunks)

    def byte_span(self, char_start: int, char_end: int) -> tuple[int, int] | None:
        """Map the text slice [char_start:char_end] to (start, end) byte offsets.

        Returns None when tags cannot go around that slice without changing the
        document: the slice is empty or out of range, crosses a tag, comment or
        CDATA marker, lies inside CDATA, or starts or ends in the middle of an
        entity that stands for several characters.
        """
        if not 0 <= char_start < char_end <= len(self.text):
            return None
        first, first_offset = self._chunk_at(char_start, is_end=False)
        last, last_offset = self._chunk_at(char_end, is_end=True)
        if first.run != last.run or first.in_cdata:
            return None
        start = _byte_offset(first, first_offset)
        end = _byte_offset(last, last_offset)
        if start is None or end is None:
            return None
        return start, end

    def _chunk_at(self, index: int, is_end: bool) -> tuple[Chunk, int]:
        """Return the chunk holding text position `index`, and the offset inside it.

        A position exactly between two chunks goes to the later chunk when it
        starts a span and to the earlier chunk when it ends one, so a span never
        swallows a tag that sits between them.
        """
        position = 0
        for chunk in self.chunks:
            chunk_end = position + len(chunk.text)
            if index < chunk_end or (is_end and index == chunk_end):
                return chunk, index - position
            position = chunk_end
        raise IndexError(index)


def find_paragraphs(raw: bytes) -> list[Paragraph]:
    """Return every <p> element in document order.

    Paragraphs are found at any depth and with any namespace prefix. A <p>
    inside another <p> counts as part of the outer one, so no text is
    proofread twice. Empty paragraphs are included, and callers skip them.
    """
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        raise XmlInputError("Input is UTF-16; only UTF-8 files are supported.")
    collector = _Collector(raw)
    paragraphs = collector.parse()
    encoding = collector.declared_encoding
    if encoding is not None and encoding.lower() != "utf-8":
        raise XmlInputError(
            f"Input declares encoding {encoding!r}; only UTF-8 files are supported."
        )
    return paragraphs


class _Collector:
    """Receives expat's events one by one and builds the list of paragraphs."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.paragraphs: list[Paragraph] = []
        self.declared_encoding: str | None = None
        self.depth = 0
        self.paragraph_depth: int | None = None
        self.current_run = 0
        self.in_cdata = False
        self.pending: Chunk | None = None

        self.parser = expat.ParserCreate()
        self.parser.XmlDeclHandler = self.on_xml_declaration
        self.parser.StartElementHandler = self.on_start_tag
        self.parser.EndElementHandler = self.on_end_tag
        self.parser.CharacterDataHandler = self.on_text
        self.parser.StartCdataSectionHandler = self.on_cdata_start
        self.parser.EndCdataSectionHandler = self.on_cdata_end
        self.parser.CommentHandler = self.on_other_markup
        self.parser.ProcessingInstructionHandler = self.on_other_markup

    def parse(self) -> list[Paragraph]:
        try:
            self.parser.Parse(self.raw, True)
        except expat.ExpatError as error:
            raise XmlInputError(f"Input is not well-formed XML: {error}") from error
        return self.paragraphs

    def on_xml_declaration(self, version: str, encoding: str | None, standalone: int) -> None:
        self.declared_encoding = encoding

    def on_start_tag(self, name: str, attributes: dict[str, str]) -> None:
        self.end_text()
        self.depth += 1
        if self.paragraph_depth is None and _local_name(name) == "p":
            self.paragraph_depth = self.depth
            self.paragraphs.append(Paragraph(line=self.parser.CurrentLineNumber))

    def on_end_tag(self, name: str) -> None:
        self.end_text()
        if self.depth == self.paragraph_depth:
            self.paragraph_depth = None
        self.depth -= 1

    def on_text(self, text: str) -> None:
        self.finish_pending_chunk()
        if self.paragraph_depth is None:
            return
        chunk = Chunk(
            text=text,
            start=self.parser.CurrentByteIndex,
            run=self.current_run,
            in_cdata=self.in_cdata,
        )
        self.paragraphs[-1].chunks.append(chunk)
        self.pending = chunk

    def on_cdata_start(self) -> None:
        self.end_text()
        self.in_cdata = True

    def on_cdata_end(self) -> None:
        self.end_text()
        self.in_cdata = False

    def on_other_markup(self, *details: str) -> None:
        """Comments and processing instructions only matter as boundaries."""
        self.end_text()

    def end_text(self) -> None:
        """Every event other than text closes the pending chunk and starts a new run."""
        self.finish_pending_chunk()
        self.current_run += 1

    def finish_pending_chunk(self) -> None:
        """The current event begins exactly where the pending chunk's raw bytes end."""
        if self.pending is None:
            return
        chunk = self.pending
        chunk.end = self.parser.CurrentByteIndex
        chunk.plain = self.raw[chunk.start : chunk.end] == chunk.text.encode("utf-8")
        self.pending = None


def _byte_offset(chunk: Chunk, offset: int) -> int | None:
    """Byte position of character `offset` within `chunk`, or None if it has none."""
    if offset == 0:
        return chunk.start
    if offset == len(chunk.text):
        return chunk.end
    if chunk.plain:
        return chunk.start + len(chunk.text[:offset].encode("utf-8"))
    return None


def _local_name(tag: str) -> str:
    """The tag name without its namespace prefix: <p> and <h:p> both give "p"."""
    return tag.rpartition(":")[2]
