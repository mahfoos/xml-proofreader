"""Measure a run: elapsed time, peak memory and the API's token use."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


def peak_memory_mb() -> float | None:
    """Measure the most memory this process has held so far.

    ru_maxrss is counted in bytes on macOS but in kilobytes on Linux, so the
    divisor depends on the platform.

    Returns:
        The peak in megabytes, or None on Windows, which has no resource
        module.
    """
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


@dataclass(frozen=True)
class RunMetrics:
    """The numbers reported at the end of a run.

    Attributes:
        file: The input file.
        seconds: Wall-clock time for the whole file.
        peak_memory_mb: Peak memory in megabytes, or None where unavailable.
        paragraphs: How many <p> elements the document has.
        texts_proofread: How many distinct, non-blank texts were sent.
        errors_marked: How many <error> tags were added.
        findings_rejected: How many findings were refused as unsafe.
        paragraphs_failed: How many paragraphs were left unchanged.
        api_calls: How many replies the API returned.
        input_tokens: Input tokens billed at the normal price.
        output_tokens: Output tokens, including the model's thinking.
        cache_read_tokens: Input tokens read from the prompt cache.
        cache_write_tokens: Input tokens written to the prompt cache.
    """

    file: str
    seconds: float
    peak_memory_mb: float | None
    paragraphs: int
    texts_proofread: int
    errors_marked: int
    findings_rejected: int
    paragraphs_failed: int
    api_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

    def summary(self) -> str:
        """Summarise the run in one line for the log.

        Returns:
            Every number in a short, readable sentence.
        """
        memory = "n/a" if self.peak_memory_mb is None else f"{self.peak_memory_mb:.1f} MB"
        return (
            f"{self.file}: {self.seconds:.1f} s, peak memory {memory}, "
            f"{self.paragraphs} paragraphs, {self.errors_marked} errors marked, "
            f"{self.findings_rejected} findings rejected, "
            f"{self.paragraphs_failed} paragraphs failed, "
            f"{self.api_calls} API calls, tokens in {self.input_tokens} out {self.output_tokens}, "
            f"cache read {self.cache_read_tokens} write {self.cache_write_tokens}"
        )

    def write_json(self, path: Path) -> None:
        """Save every number to a JSON file, for scripts and dashboards.

        Args:
            path: Where to write the file. An existing file is replaced.
        """
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
