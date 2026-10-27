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
    """The input is not well-formed XML, is not UTF-8, or has paragraph text it cannot read."""


@dataclass
class Chunk:
    """One piece of paragraph text, exactly as expat reported it.

    Attributes:
        text: The decoded text, with entities replaced and every line break
            turned into a single line feed.
        start: Byte offset in the file where this text's raw bytes begin.
        run: Chunks share a run number when no tag, comment or CDATA marker
            separates them.
        in_cdata: True inside <![CDATA[...]]>, where an inserted tag would
            become literal text.
        end: Byte offset just past the raw bytes. Set when the next parser
            event arrives.
        plain: True when the raw bytes are exactly the text encoded as UTF-8,
            so every character has its own byte position.
    """

    text: str
    start: int
    run: int
    in_cdata: bool
    end: int = 0
    plain: bool = False


@dataclass
class Paragraph:
    """One <p> element and its text, including text inside child elements.

    Attributes:
        line: Line number of the <p> start tag, for log messages.
        chunks: The paragraph's text, in the pieces expat reported.
    """

    line: int
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The paragraph's full text: what the model proofreads."""
        return "".join(chunk.text for chunk in self.chunks)

    def byte_span(self, char_start: int, char_end: int) -> tuple[int, int] | None:
        """Map a slice of the paragraph's text to byte offsets in the file.

        Args:
            char_start: Index in text of the slice's first character.
            char_end: Index in text just past the slice's last character.

        Returns:
            The (start, end) byte offsets of the slice's raw bytes, or None
            when tags cannot go around the slice without changing the
            document: the slice is empty or out of range, crosses a tag,
            comment or CDATA marker, lies inside CDATA, or starts or ends in
            the middle of an entity that stands for several characters.
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
        """Find the chunk that holds a position in the paragraph's text.

        A position exactly between two chunks goes to the later chunk when it
        starts a span and to the earlier chunk when it ends one, so a span
        never swallows a tag that sits between them.

        Args:
            index: The position in text.
            is_end: True when the position ends a span rather than starts it.

        Returns:
            The chunk, and the position's offset inside that chunk.

        Raises:
            IndexError: If the position is beyond the end of the text. This
                cannot happen when called from byte_span, which checks the
                range first.
        """
        position = 0
        for chunk in self.chunks:
            chunk_end = position + len(chunk.text)
            if index < chunk_end or (is_end and index == chunk_end):
                return chunk, index - position
            position = chunk_end
        raise IndexError(index)


def find_paragraphs(raw: bytes) -> list[Paragraph]:
    """Find every <p> element in a document, in document order.

    Paragraphs are found at any depth and with any namespace prefix. A <p>
    inside another <p> counts as part of the outer one, so no text is
    proofread twice. Empty paragraphs are included, and callers skip them.

    Args:
        raw: The XML document, as the bytes read from the file.

    Returns:
        The paragraphs, with byte offsets for every piece of their text.

    Raises:
        XmlInputError: If the document is not well-formed XML, is not
            encoded as UTF-8, or uses an entity in a paragraph whose
            definition is outside the file.
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
        """Prepare a parser for one document and connect its events to methods.

        Args:
            raw: The XML document, as the bytes read from the file.
        """
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
        self.parser.SkippedEntityHandler = self.on_skipped_entity
        self.parser.StartCdataSectionHandler = self.on_cdata_start
        self.parser.EndCdataSectionHandler = self.on_cdata_end
        self.parser.CommentHandler = self.on_other_markup
        self.parser.ProcessingInstructionHandler = self.on_other_markup

    def parse(self) -> list[Paragraph]:
        """Run expat over the whole document.

        Returns:
            Every paragraph found, in document order.

        Raises:
            XmlInputError: If expat finds the document is not well-formed, or
                a paragraph uses an entity whose definition is outside the file.
        """
        try:
            self.parser.Parse(self.raw, True)
        except expat.ExpatError as error:
            raise XmlInputError(f"Input is not well-formed XML: {error}") from error
        return self.paragraphs

    def on_xml_declaration(self, version: str, encoding: str | None, standalone: int) -> None:
        """Remember the encoding named in the <?xml ...?> declaration.

        Args:
            version: The XML version. Unused, but expat always passes it.
            encoding: The declared encoding, or None if none is declared.
            standalone: The standalone flag. Unused, but expat always passes it.
        """
        self.declared_encoding = encoding

    def on_start_tag(self, name: str, attributes: dict[str, str]) -> None:
        """Go one level deeper, and open a paragraph at the outermost <p>.

        Args:
            name: The tag name, including any namespace prefix.
            attributes: The tag's attributes. Unused, but expat always passes them.
        """
        self.end_text()
        self.depth += 1
        if self.paragraph_depth is None and _local_name(name) == "p":
            self.paragraph_depth = self.depth
            self.paragraphs.append(Paragraph(line=self.parser.CurrentLineNumber))

    def on_end_tag(self, name: str) -> None:
        """Go one level up, closing the paragraph if its <p> just ended.

        Args:
            name: The tag name. Unused, because depth alone tells which tag ended.
        """
        self.end_text()
        if self.depth == self.paragraph_depth:
            self.paragraph_depth = None
        self.depth -= 1

    def on_text(self, text: str) -> None:
        """Record a piece of text, if it is inside a paragraph.

        Args:
            text: The decoded text that expat reported.
        """
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

    def on_skipped_entity(self, name: str, is_parameter_entity: bool) -> None:
        """Refuse an entity in a paragraph whose definition expat could not read.

        This happens when the entity is defined in an external DTD, which
        expat does not load. Expat then leaves the entity out of the text, so
        the model would proofread "Don&rsquo;t" as "Dont". Outside paragraphs
        the entity does no harm, because that text is never read or changed.

        Args:
            name: The entity's name, such as rsquo.
            is_parameter_entity: Whether the entity belongs to the DTD rather
                than the text. Unused, but expat always passes it.

        Raises:
            XmlInputError: If the entity is inside a paragraph.
        """
        if self.paragraph_depth is not None:
            raise XmlInputError(
                f"Line {self.parser.CurrentLineNumber}: the entity &{name}; is defined "
                "outside this file, so its text is unknown."
            )

    def on_cdata_start(self) -> None:
        """Mark the start of a CDATA section, which also ends the current run."""
        self.end_text()
        self.in_cdata = True

    def on_cdata_end(self) -> None:
        """Mark the end of a CDATA section, which also ends the current run."""
        self.end_text()
        self.in_cdata = False

    def on_other_markup(self, *details: str) -> None:
        """Treat a comment or processing instruction as a boundary between runs.

        Args:
            *details: The comment text, or the instruction's target and data.
                Unused, because only the boundary matters.
        """
        self.end_text()

    def end_text(self) -> None:
        """Close the pending chunk and start a new run, for any event except text."""
        self.finish_pending_chunk()
        self.current_run += 1

    def finish_pending_chunk(self) -> None:
        """Give the pending chunk its end offset and decide whether it is plain.

        The current event begins exactly where the pending chunk's raw bytes
        end, which is the only way to learn a chunk's end from expat.
        """
        if self.pending is None:
            return
        chunk = self.pending
        chunk.end = self.parser.CurrentByteIndex
        chunk.plain = self.raw[chunk.start : chunk.end] == chunk.text.encode("utf-8")
        self.pending = None


def _byte_offset(chunk: Chunk, offset: int) -> int | None:
    """Find the byte position of a character position inside a chunk.

    Args:
        chunk: The chunk that holds the position.
        offset: The character position inside the chunk's text.

    Returns:
        The byte offset in the file, or None when the position falls inside
        an entity whose bytes and characters do not line up.
    """
    if offset == 0:
        return chunk.start
    if offset == len(chunk.text):
        return chunk.end
    if chunk.plain:
        return chunk.start + len(chunk.text[:offset].encode("utf-8"))
    return None


def _local_name(tag: str) -> str:
    """Remove the namespace prefix from a tag name.

    Args:
        tag: A tag name such as "p" or "h:p".

    Returns:
        The part after the last colon, so "p" and "h:p" both give "p".
    """
    return tag.rpartition(":")[2]
