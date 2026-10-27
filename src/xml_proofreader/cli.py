"""Command line: xml-proofread INPUT --lang TAG --style-guide GUIDE.

Reads the XML file, proofreads its <p> elements with Claude, and writes the
annotated copy next to the input as NAME.corrected.xml. The API key comes from
ANTHROPIC_API_KEY, which can also be set in a .env file in the current folder.

Exit codes: 0 when every paragraph was proofread; 3 when the output was
written but some paragraphs could not be proofread; 1 when nothing was
written; 2 when the command line itself is wrong; 130 when interrupted.
"""

import argparse
import asyncio
import logging
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .locator import XmlInputError
from .metrics import RunMetrics, peak_memory_mb
from .model_client import DEFAULT_EFFORT, DEFAULT_MODEL, ClaudeProofreader
from .pipeline import Correction, correct_document
from .style_guide import StyleGuideError, load_style_guide
from .verifier import VerificationError

log = logging.getLogger("xml_proofreader")

EFFORTS = ["low", "medium", "high", "xhigh", "max"]
LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*")
API_TIMEOUT_SECONDS = 120.0
API_MAX_RETRIES = 4

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130


def main(argv: list[str] | None = None) -> int:
    """Run the tool and return the process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    load_dotenv(Path.cwd() / ".env")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        log.error("No API key found. Set ANTHROPIC_API_KEY, or put it in a .env file (see .env.example).")
        return EXIT_FAILED

    try:
        return asyncio.run(run(args))
    except (OSError, XmlInputError, StyleGuideError, VerificationError) as error:
        log.error("%s", error)
    except anthropic.AuthenticationError:
        log.error("The API rejected the key. Check ANTHROPIC_API_KEY.")
    except anthropic.APIStatusError as error:
        log.error("The API refused the request (%s): %s", error.status_code, error.message)
    except KeyboardInterrupt:
        log.error("Interrupted. Nothing was written.")
        return EXIT_INTERRUPTED
    return EXIT_FAILED


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line. argparse prints usage and exits with code 2 on mistakes."""
    parser = argparse.ArgumentParser(
        prog="xml-proofread",
        description="Proofread the <p> elements of an XML file with Claude and mark each error.",
    )
    parser.add_argument("input", type=Path, help="the XML file to proofread")
    parser.add_argument("--lang", required=True, type=language_tag, help="BCP 47 tag, such as en or en-GB")
    parser.add_argument("--style-guide", required=True, type=Path, help="the style guide .docx")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default {DEFAULT_MODEL})")
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, choices=EFFORTS, help=f"thinking effort (default {DEFAULT_EFFORT})"
    )
    parser.add_argument(
        "--concurrency", default=8, type=positive_int, help="paragraphs proofread at once (default 8)"
    )
    parser.add_argument("--metrics-json", type=Path, help="also write the run's metrics to this file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def language_tag(value: str) -> str:
    """Accept tags shaped like BCP 47, such as en, en-GB, fr-CA or zh-Hant-TW."""
    if not LANGUAGE_TAG.fullmatch(value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a language tag such as en or en-GB")
    return value


def positive_int(value: str) -> int:
    """Accept whole numbers of 1 or more."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


async def run(args: argparse.Namespace) -> int:
    """Correct one file, write the result next to it, and report what happened."""
    started = time.perf_counter()
    raw = args.input.read_bytes()
    style_guide = load_style_guide(args.style_guide)

    async with anthropic.AsyncAnthropic(
        timeout=API_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES
    ) as client:
        proofreader = ClaudeProofreader(client, style_guide, args.lang, args.model, args.effort)
        log.info("Proofreading %s with %s at %s effort", args.input, args.model, args.effort)
        correction = await correct_document(raw, proofreader.proofread, args.concurrency)

    destination = output_path(args.input)
    write_atomically(destination, correction.output)
    log_problems(correction)
    log.info("Wrote %s", destination)

    usage = proofreader.usage
    metrics = RunMetrics(
        file=str(args.input),
        seconds=time.perf_counter() - started,
        peak_memory_mb=peak_memory_mb(),
        paragraphs=correction.paragraph_count,
        texts_proofread=correction.texts_proofread,
        errors_marked=len(correction.annotation.placements),
        findings_rejected=len(correction.annotation.rejections),
        paragraphs_failed=len(correction.failures),
        api_calls=usage.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    )
    log.info("%s", metrics.summary())
    if args.metrics_json:
        metrics.write_json(args.metrics_json)
    return EXIT_PARTIAL if correction.failures else EXIT_OK


def output_path(input_path: Path) -> Path:
    """input.xml becomes input.corrected.xml, in the same folder."""
    return input_path.with_name(f"{input_path.stem}.corrected.xml")


def write_atomically(path: Path, data: bytes) -> None:
    """Write a temporary file, then rename it, so a crash never leaves half a file behind."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def log_problems(correction: Correction) -> None:
    """Warn about every finding that was skipped and every paragraph left unchanged."""
    for rejection in correction.annotation.rejections:
        finding = rejection.finding
        log.warning(
            "Line %d: skipped %r (correction %r): %s",
            rejection.line,
            finding.original,
            finding.correction,
            rejection.problem,
        )
    for failure in correction.failures:
        log.warning("Line %d: paragraph left unchanged: %s", failure.line, failure.problem)
