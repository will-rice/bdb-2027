"""Keyword census over the cached notebook digests: which positions, methods and framings the field uses.

Each pattern is matched against a notebook's markdown plus the first 12,000 characters of its code, so counts
are lower bounds. Run ``survey_fetch_notebooks.py`` first, then from the repository root:
``uv run python docs/research/direction-probes/survey_census.py``.
"""

import json
import logging
import re
import statistics
from pathlib import Path

CACHE = Path("data/processed/writeup_survey")
THEMES = {
    "WR/TE routes, cuts, separation": r"separation_at_pass_forward|route_ran|skill_drills_wr|gauntlet",
    "DL / edge pass rush": r"skill_drills_dl|run_the_hoop|player_get_off|quick_pressure|time_to_pressure\b",
    "OL pass protection": r"skill_drills_ol|pressure_allowed|pass_pro|mirror",
    "DB coverage": r"skill_drills_db|w_drill|coverage_assignment|back_pedal",
    "uses game tracking": r"game_tracking",
    "controls for weight": r"combine_weight",
    "controls for draft slot": r"draft_overall_pick|draft_round",
    "repeatability / reliability check": r"split.half|test.retest|repeatab|reliab|spearman.brown|icc",
    "held-out draft class or permutation test": r"leave.one.(draft|class|year)|lodo|held.out|permutation",
    "reports a null or negative result": (
        r"null result|no detectable|does not (transfer|predict|carry)|did not (transfer|predict|demonstrate)"
        r"|negative result|doesn't (transfer|carry)|no signal"
    ),
    "quadrant / archetype matrix": r"quadrant|archetype",
    "'hidden' gems / sleepers / steals": r"hidden (mover|talent|gem)|sleeper|draft steal",
    "'beyond the stopwatch' framing": r"beyond the stopwatch|beyond the 40",
    "two tracked players on one play (matchups)": r"who.faced.whom|rookie.vs.rookie|matchup layer",
    "2026 season out-of-time test": r"2026 (regular.season|season|out.of.time)|stats_player_week_2026|snap_counts_2026",
    "external data (nflverse etc.)": r"nflverse|nflreadr|nfl_data_py|pro-football-reference|pfr",
    "animation": r"funcanimation|animate|\.gif",
    "ties receiver size to separation in prose": (
        r"(heav|light|weight|bigger|smaller|size|frame|height|taller)[^.\n]{0,120}separat"
        r"|separat[^.\n]{0,120}(heav|light|weight|bigger|smaller|body mass|frame|height|taller)"
    ),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = json.loads((CACHE / "index.json").read_text())
    cached = [r for r in rows if r["ok"]]
    texts = [
        (CACHE / "nb" / f"{r['ref'].replace('/', '__')}.txt").read_text().lower()
        for r in cached
    ]
    logging.info(
        "%d public notebooks listed, %d readable as notebooks", len(rows), len(cached)
    )
    for theme, pattern in THEMES.items():
        hits = sum(bool(re.search(pattern, text)) for text in texts)
        logging.info("%4d (%3.0f%%)  %s", hits, 100 * hits / len(cached), theme)
    words = [r["md_words"] for r in cached]
    logging.info(
        "markdown words: median %.0f | at least 1,000 words: %d | under 50 words: %d",
        statistics.median(words),
        sum(w >= 1000 for w in words),
        sum(w < 50 for w in words),
    )


if __name__ == "__main__":
    main()
