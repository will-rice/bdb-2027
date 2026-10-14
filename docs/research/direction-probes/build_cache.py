"""Convert the three game-tracking CSVs to parquet so the probes can re-read them quickly.

Run from the repository root before any probe: ``uv run python docs/research/direction-probes/build_cache.py``.
"""

import logging
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
CACHE = Path("data/processed/direction_probes")
SEASONS = (2023, 2024, 2025)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    CACHE.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        (
            pl.scan_csv(
                RAW / f"game_tracking_{season}.csv",
                null_values=["NA", ""],
                schema_overrides={"event": pl.Utf8, "o": pl.Float64, "dir": pl.Float64},
            )
            .with_columns(
                pl.col("time").str.to_datetime(strict=False),
                pl.lit(season).alias("season"),
            )
            .sink_parquet(CACHE / f"gt_{season}.parquet", compression="zstd")
        )
    rows = pl.scan_parquet(CACHE / "gt_*.parquet").select(pl.len()).collect().item()
    logging.info("cached %d tracking frames in %s", rows, CACHE)


if __name__ == "__main__":
    main()
