"""Every statistic quoted in the submission write-up must be a value the notebook prints or defines."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.]*\d)")
GROUPED_INTEGER = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\w,])")


def notebook_text() -> str:
    """Code-cell sources plus every text output of the executed notebook."""
    parts = []
    for cell in json.loads((ROOT / "notebooks" / "frame_first.ipynb").read_text())[
        "cells"
    ]:
        if cell["cell_type"] != "code":
            continue
        parts.append("".join(cell["source"]))
        for output in cell["outputs"]:
            parts.append("".join(output.get("text", [])))
            parts.append("".join(output.get("data", {}).get("text/plain", [])))
    return "\n".join(parts)


def writeup_claims() -> str:
    """The write-up's prose and tables, without its reference list, links and image paths."""
    body = (
        (ROOT / "docs" / "SUBMISSION_WRITEUP.md").read_text().split("## References")[0]
    )
    return re.sub(r"\]\([^)]*\)", "]", body)


def test_every_decimal_in_the_writeup_comes_from_the_notebook() -> None:
    """A decimal quoted in the write-up must appear, digit for digit, in the notebook's code or output."""
    printed = set(DECIMAL.findall(notebook_text()))
    quoted = set(DECIMAL.findall(writeup_claims()))
    assert quoted, "the write-up should quote statistics"
    assert not sorted(quoted - printed)


def test_every_large_count_in_the_writeup_comes_from_the_notebook() -> None:
    """A comma-grouped count quoted in the write-up must appear in the notebook's code or output."""
    printed = set(GROUPED_INTEGER.findall(notebook_text()))
    quoted = set(GROUPED_INTEGER.findall(writeup_claims()))
    assert not sorted(quoted - printed)
