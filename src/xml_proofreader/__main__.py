"""Lets `python -m xml_proofreader` run the same command as `xml-proofread`."""

from .cli import main

raise SystemExit(main())
