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
    """The most memory this process has held so far, in megabytes.

    Returns None on Windows, which has no resource module. Elsewhere,
    ru_maxrss is counted in bytes on macOS but in kilobytes on Linux.
    """
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


@dataclass(frozen=True)
class RunMetrics:
    """The numbers reported at the end of a run."""

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
        """One line for the log."""
        memory = "n/a" if self.peak_memory_mb is None else f"{self.peak_memory_mb:.1f} MB"
        return (
            f"{self.file}: {self.seconds:.1f} s, peak memory {memory}, "
            f"{self.paragraphs} paragraphs, {self.errors_marked} errors marked, "
            f"{self.findings_rejected} findings rejected, {self.paragraphs_failed} paragraphs failed, "
            f"{self.api_calls} API calls, tokens in {self.input_tokens} out {self.output_tokens}, "
            f"cache read {self.cache_read_tokens} write {self.cache_write_tokens}"
        )

    def write_json(self, path: Path) -> None:
        """Save every number to a JSON file, for scripts and dashboards."""
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
