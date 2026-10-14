"""Read every public notebook for the competition through the Kaggle API and cache plain-text digests.

Notebook source is parsed as JSON and never executed. Digests are written to ``data/`` (git-ignored) because
they are other people's work and must not be redistributed. The API rate-limits after about 50 reads, so the
script is resumable: run it again after a few minutes until every row in ``index.json`` reads ``ok``.

Run from the repository root: ``uv run python docs/research/direction-probes/survey_fetch_notebooks.py``.
"""

import json
import logging
import re
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

COMPETITION = "nfl-big-data-bowl-2027"
OUT = Path("data/processed/writeup_survey")
PAGE_SIZE = 100
PAUSE_SECONDS = 4.0
CODE_CHARS = 12000


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    api = KaggleApi()
    api.authenticate()
    (OUT / "nb").mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    if not index_path.exists():
        index_path.write_text(json.dumps(list_notebooks(api), indent=1))
    rows = json.loads(index_path.read_text())
    for i, row in enumerate(rows):
        if row["ok"]:
            continue
        rows[i] = digest(api, row)
        index_path.write_text(json.dumps(rows, indent=1))
        if "429" in rows[i].get("error", ""):
            logging.info("rate limited at %s; run again in a few minutes", row["ref"])
            break
        time.sleep(PAUSE_SECONDS)
    logging.info("%d of %d notebooks cached", sum(r["ok"] for r in rows), len(rows))


def list_notebooks(api: KaggleApi) -> list[dict]:
    """List every public notebook attached to the competition, most-upvoted first."""
    rows = []
    page = 1
    while batch := api.kernels_list(
        competition=COMPETITION, page=page, page_size=PAGE_SIZE, sort_by="voteCount"
    ):
        rows.extend(
            {
                "ref": k.ref,
                "title": k.title,
                "author": k.author,
                "votes": k.total_votes,
                "ok": False,
            }
            for k in batch
        )
        page += 1
    return rows


def digest(api: KaggleApi, row: dict) -> dict:
    """Fetch one notebook's source and write its markdown and leading code to a text file."""
    owner, slug = row["ref"].split("/")
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    try:
        with api.build_kaggle_client() as kaggle:
            source = (
                kaggle.kernels.kernels_api_client.get_kernel(request).blob.source or ""
            )
        cells = json.loads(source)["cells"]
    except Exception as exc:  # noqa: BLE001 - any API or parse failure is recorded and retried on the next run
        return {**row, "error": f"{type(exc).__name__}: {exc}"[:200]}
    joined = {
        kind: [
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in cells
            if c["cell_type"] == kind
        ]
        for kind in ("markdown", "code")
    }
    text = "\n\n".join(joined["markdown"])
    code = "\n".join(joined["code"])
    (OUT / "nb" / f"{owner}__{slug}.txt").write_text(
        f"REF: {row['ref']}\nTITLE: {row['title']}\nVOTES: {row['votes']}\n\n"
        f"===== MARKDOWN =====\n{text}\n\n===== CODE (first {CODE_CHARS} chars) =====\n{code[:CODE_CHARS]}\n"
    )
    return {
        **row,
        "ok": True,
        "error": "",
        "md_words": len(text.split()),
        "code_lines": code.count("\n") + 1,
        "uses_game_tracking": "game_tracking" in code,
        "filters_ball": bool(re.search(r"entity_type|PLAYER", code)),
    }


if __name__ == "__main__":
    main()
