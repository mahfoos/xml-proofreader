# xml-proofreader

Command-line tool that proofreads the `<p>` elements of an XML file with Claude and marks each error with an `<error>` tag. Everything outside the tags is left byte-for-byte unchanged. The result is written next to the input as `<name>.corrected.xml`.

```xml
<p>The <error type="spelling" correction="committee" reason="Misspelled word.">committe</error> has decided ...</p>
```

## Setup

Requires Python 3.11+ and an Anthropic API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put your key in `.env` after `ANTHROPIC_API_KEY=`, or export it in your shell.

## Usage

```bash
xml-proofread examples/sample_input.xml --lang en --style-guide tests/fixtures/style_guide.docx
```

This writes `examples/sample_input.corrected.xml` and logs a summary with the time, peak memory and token usage. Add `--metrics-json metrics.json` to save those numbers to a file.

Other options: `--model` (default `claude-opus-5-5`), `--effort` (default `high`), `--concurrency` (default 8) and `--log-level`. See `xml-proofread --help`.

Exit codes: 0 on success, 3 if some paragraphs could not be proofread (the output is still written), 1 if no output was written, 2 for invalid arguments.

## Examples

[`examples/`](examples/) contains the two files from the brief with their outputs and metrics:

- `example_input.xml`: all 10 expected errors are found, with the same types and corrections as `example_output.xml`.
- `sample_input.xml`: 45 errors marked in about 10 seconds.

## How it works

- Paragraphs are located with Python's expat parser, which reports byte offsets. The document is never re-serialized, so tags, attributes, namespaces and CDATA stay exactly as they were.
- Each paragraph is sent to Claude with the full style guide in a cached system prompt. The model returns structured findings (original text, correction, type, reason). It never rewrites the text itself.
- The tags are inserted around the original bytes. Before writing, the output is checked: it must be well-formed, every paragraph's text must be unchanged, and nothing outside the tags may differ from the input.
- Paragraphs are processed concurrently. A paragraph that fails is left unchanged and logged, and the rest are still corrected.

## Tests

```bash
pytest
ruff check .
ruff format --check .
pyright
```

The tests run offline and need no API key. One live test that calls the API is skipped unless `XML_PROOFREADER_LIVE=1` is set.

## Notes

- Only `<p>` elements are proofread, as the brief asks. Headings are left as they are.
- Input files must be UTF-8.
- `--lang en` is treated as US English, which matches the style guide. Use `en-GB` for British spelling.
- The tool was built and tested with Claude. All model calls go through one class, `ClaudeProofreader` in `model_client.py`, so another provider can be added there.
