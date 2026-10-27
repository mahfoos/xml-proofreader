"""Command line: xml-proofread INPUT --lang TAG --style-guide GUIDE.

Reads the XML file, proofreads its <p> elements with Claude, and writes the
annotated copy next to the input as NAME.corrected.xml. The API key comes from
ANTHROPIC_API_KEY, which can also be set in a .env file in the current folder.

Exit codes: 0 when every paragraph was proofread; 3 when the output was
written but some paragraphs could not be proofread; 1 when nothing was
written, including when no paragraph at all could be proofread; 2 when the
command line itself is wrong; 130 when interrupted.
"""

import argparse
import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import get_args

import anthropic
from dotenv import load_dotenv

from .locator import XmlInputError
from .metrics import RunMetrics, peak_memory_mb
from .model_client import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    ClaudeProofreader,
    Effort,
    TokenUsage,
    describe_api_error,
)
from .pipeline import Correction, correct_document
from .style_guide import StyleGuideError, load_style_guide
from .verifier import VerificationError

log = logging.getLogger("xml_proofreader")

EFFORTS = list(get_args(Effort))
LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*")
API_TIMEOUT_SECONDS = 120.0
CONNECT_TIMEOUT_SECONDS = 10.0
API_MAX_RETRIES = 4

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130


def main(argv: list[str] | None = None) -> int:
    """Run the tool once, turning every expected error into a message and exit code.

    Args:
        argv: The command-line arguments, or None to read them from sys.argv.

    Returns:
        The process exit code, as listed in the module docstring.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    library_level = logging.DEBUG if args.log_level == "DEBUG" else logging.WARNING
    for library in ("anthropic", "httpx2"):
        logging.getLogger(library).setLevel(library_level)
    load_dotenv(Path.cwd() / ".env")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        log.error(
            "No API key found. Set ANTHROPIC_API_KEY, or put it in a .env file (see .env.example)."
        )
        return EXIT_FAILED

    try:
        return asyncio.run(run(args))
    except (OSError, XmlInputError, StyleGuideError, VerificationError) as error:
        log.error("%s", error)
    except anthropic.AuthenticationError:
        log.error("The API rejected the key. Check ANTHROPIC_API_KEY.")
    except anthropic.APIStatusError as error:
        log.error(
            "The API refused the request (%s): %s", error.status_code, describe_api_error(error)
        )
    except KeyboardInterrupt:
        log.error("Interrupted. Nothing was written.")
        return EXIT_INTERRUPTED
    return EXIT_FAILED


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read and validate the command line.

    Args:
        argv: The command-line arguments, or None to read them from sys.argv.

    Returns:
        The parsed arguments.

    Raises:
        SystemExit: With code 2, after argparse prints the usage, if the
            command line is wrong.
    """
    parser = argparse.ArgumentParser(
        prog="xml-proofread",
        description="Proofread the <p> elements of an XML file with Claude and mark each error.",
    )
    parser.add_argument("input", type=Path, help="the XML file to proofread")
    parser.add_argument(
        "--lang", required=True, type=language_tag, help="BCP 47 tag, such as en or en-GB"
    )
    parser.add_argument("--style-guide", required=True, type=Path, help="the style guide .docx")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Claude model (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=EFFORTS,
        help=f"thinking effort (default {DEFAULT_EFFORT})",
    )
    parser.add_argument(
        "--concurrency",
        default=8,
        type=positive_int,
        help="paragraphs proofread at once (default 8)",
    )
    parser.add_argument(
        "--metrics-json", type=Path, help="also write the run's metrics to this file"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def language_tag(value: str) -> str:
    """Check that a value is shaped like a BCP 47 language tag.

    Only the shape is checked, so en, en-GB, fr-CA and zh-Hant-TW all pass.

    Args:
        value: The text given after --lang.

    Returns:
        The tag, unchanged.

    Raises:
        argparse.ArgumentTypeError: If the value is not shaped like a tag.
    """
    if not LANGUAGE_TAG.fullmatch(value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a language tag such as en or en-GB")
    return value


def positive_int(value: str) -> int:
    """Check that a value is a whole number of 1 or more.

    Args:
        value: The text given after --concurrency.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If the number is less than 1.
        ValueError: If the value is not a whole number. argparse reports
            both errors the same way.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


async def run(args: argparse.Namespace) -> int:
    """Correct one file, write the result next to it, and report what happened.

    Args:
        args: The parsed command line.

    Returns:
        The process exit code: 0, 1 or 3.

    Raises:
        OSError: If a file cannot be read, or the output folder is not
            writable. Other errors from the steps below pass through to main.
    """
    started = time.perf_counter()
    raw = args.input.read_bytes()
    style_guide = load_style_guide(args.style_guide)
    destination = output_path(args.input)
    if not os.access(destination.parent, os.W_OK):
        raise OSError(f"Cannot write to the folder {destination.parent}, so nothing was sent.")

    timeout = anthropic.Timeout(API_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    async with anthropic.AsyncAnthropic(timeout=timeout, max_retries=API_MAX_RETRIES) as client:
        proofreader = ClaudeProofreader(client, style_guide, args.lang, args.model, args.effort)
        log.info("Proofreading %s with %s at %s effort", args.input, args.model, args.effort)
        correction = await correct_document(raw, proofreader.proofread, args.concurrency)

    log_problems(correction)
    if correction.texts_proofread and correction.texts_failed == correction.texts_proofread:
        log.error("No paragraph could be proofread, so nothing was written.")
        return EXIT_FAILED
    write_atomically(destination, correction.output)
    log.info("Wrote %s", destination)

    metrics = collect_metrics(args.input, started, correction, proofreader.usage)
    log.info("%s", metrics.summary())
    if args.metrics_json:
        metrics.write_json(args.metrics_json)
    return EXIT_PARTIAL if correction.failures else EXIT_OK


def collect_metrics(
    input_path: Path, started: float, correction: Correction, usage: TokenUsage
) -> RunMetrics:
    """Gather the numbers reported at the end of a run.

    Args:
        input_path: The file that was corrected.
        started: The time.perf_counter() reading taken when the run began.
        correction: The result of the run.
        usage: The token totals from the model client.

    Returns:
        The run's metrics, with the elapsed time and peak memory measured now.
    """
    return RunMetrics(
        file=str(input_path),
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


def output_path(input_path: Path) -> Path:
    """Name the output file after the input file.

    Args:
        input_path: The input file, such as input.xml.

    Returns:
        The output file in the same folder, such as input.corrected.xml.
    """
    return input_path.with_name(f"{input_path.stem}.corrected.xml")


def write_atomically(path: Path, data: bytes) -> None:
    """Write a file so that a crash never leaves half of it behind.

    The data goes to a temporary file first, which is then renamed. On one
    filesystem, a rename happens all at once.

    Args:
        path: The file to write. An existing file is replaced.
        data: The bytes to write.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def log_problems(correction: Correction) -> None:
    """Warn about every finding that was skipped and every paragraph left unchanged.

    Args:
        correction: The result of the run.
    """
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
