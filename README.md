# xml-proofreader

A command-line tool that proofreads the `<p>` elements of an XML file with a GenAI model and wraps each error in an `<error>` tag, leaving every other byte of the file unchanged.

> Work in progress: usage, design notes and performance metrics will be added as the tool is built.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```
