"""Tests for the run measurements."""

import json
from pathlib import Path

from xml_proofreader.metrics import RunMetrics, peak_memory_mb


def sample_metrics() -> RunMetrics:
    return RunMetrics(
        file="input.xml",
        seconds=12.34,
        peak_memory_mb=56.7,
        paragraphs=14,
        texts_proofread=14,
        errors_marked=40,
        findings_rejected=1,
        paragraphs_failed=0,
        api_calls=14,
        input_tokens=700,
        output_tokens=3000,
        cache_read_tokens=75000,
        cache_write_tokens=5773,
    )


def test_peak_memory_is_reported_in_megabytes() -> None:
    memory = peak_memory_mb()
    assert memory is not None
    assert 1 < memory < 4096


def test_the_summary_line_contains_the_key_numbers() -> None:
    line = sample_metrics().summary()
    assert "input.xml: 12.3 s, peak memory 56.7 MB" in line
    assert "40 errors marked" in line
    assert "cache read 75000 write 5773" in line


def test_the_metrics_file_holds_every_number(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    sample_metrics().write_json(path)
    saved = json.loads(path.read_text())
    assert saved["seconds"] == 12.34
    assert saved["errors_marked"] == 40
    assert len(saved) == 13
