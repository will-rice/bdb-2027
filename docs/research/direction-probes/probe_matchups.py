"""Probe: how many receiver-vs-defender reps can be confirmed between two tracked cohort players?"""

import logging
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = "data/processed/direction_probes/"
D = "data/raw/"
TOLERANCE_YARDS = 0.15
pp = pl.read_csv(
    D + "player_play.csv",
    null_values=["NA", ""],
    infer_schema_length=200000,
    schema_overrides={
        "sack": pl.Float64,
        "sack_allowed": pl.Float64,
        "extended_sack_allowed": pl.Float64,
        "pressure_allowed": pl.Utf8,
        "quick_pressure": pl.Utf8,
        "unblocked_pressure": pl.Utf8,
        "lined_up_position": pl.Utf8,
        "blitzing": pl.Utf8,
        "target": pl.Utf8,
        "in_motion_at_ball_snap": pl.Utf8,
    },
)
games = pl.read_csv(D + "games.csv")
players = pl.read_csv(D + "players.csv", null_values=["NA"])
pp = pp.join(
    players.select(["nfl_id", "nfl_position", "draft_year", "display_name"]),
    on="nfl_id",
).join(games.select(["game_id", "season", "season_type", "week"]), on="game_id")
pp = pp.with_columns(
    (pl.col("team_abbr") == pl.col("possession_team")).alias("on_offense")
)
pp.write_parquet(P + "pp.parquet")
frames = (
    pl.scan_parquet(P + "gt_*.parquet")
    .filter(pl.col("event") == "pass_forward")
    .select(["game_id", "play_id", "nfl_id", "x", "y", "s", "dir"])
    .collect()
)
logging.info(
    "%s %s %s %s",
    "pass_forward frames:",
    frames.height,
    "| plays:",
    frames.select(["game_id", "play_id"]).n_unique(),
)
receivers = (
    pp.filter(
        pl.col("route_ran").is_not_null()
        & pl.col("separation_at_pass_forward").is_not_null()
    )
    .select(
        [
            "game_id",
            "play_id",
            "nfl_id",
            "team_abbr",
            "route_ran",
            "target",
            "separation_at_pass_forward",
            "team_coverage_man_zone",
            "season",
            "season_type",
            "nfl_position",
            "cushion",
        ]
    )
    .join(frames, on=["game_id", "play_id", "nfl_id"])
)
defenders = (
    pp.filter(~pl.col("on_offense"))
    .select(
        [
            "game_id",
            "play_id",
            pl.col("nfl_id").alias("def_id"),
            pl.col("team_abbr").alias("def_team"),
            pl.col("nfl_position").alias("def_pos"),
            "coverage_assignment",
            "blitzing",
        ]
    )
    .join(
        frames.rename(
            {"nfl_id": "def_id", "x": "dx", "y": "dy", "s": "ds", "dir": "ddir"}
        ),
        on=["game_id", "play_id", "def_id"],
    )
)
logging.info(
    "%s %s %s %s",
    "receiver route reps with tracking at the throw:",
    receivers.height,
    "| defender-plays at the throw:",
    defenders.height,
)
pairs = receivers.join(defenders, on=["game_id", "play_id"]).filter(
    pl.col("team_abbr") != pl.col("def_team")
)
pairs = pairs.with_columns(
    ((pl.col("x") - pl.col("dx")) ** 2 + (pl.col("y") - pl.col("dy")) ** 2)
    .sqrt()
    .alias("dist")
)
pairs = pairs.with_columns(
    (pl.col("dist") - pl.col("separation_at_pass_forward")).alias("gap")
)
logging.info(
    "%s %s", "receiver x tracked-defender pairs on the same play:", pairs.height
)
logging.info(
    "%s %s",
    "gap = tracked distance - recorded separation; quantiles:",
    pairs.select(
        [
            pl.col("gap").quantile(q).round(2).alias(f"q{int(q * 100)}")
            for q in (0.01, 0.05, 0.25, 0.5, 0.75)
        ]
    ).row(0),
)
logging.info(
    "%s %s",
    "pairs with gap < -0.15 (tracked defender closer than recorded nearest defender; should be ~0):",
    pairs.filter(pl.col("gap") < -TOLERANCE_YARDS).height,
)
matched = pairs.filter(pl.col("gap").abs() <= TOLERANCE_YARDS)
logging.info(
    "%s %s",
    "\nCONFIRMED matchups (tracked defender IS the recorded nearest defender):",
    matched.height,
)
logging.info(
    "%s %s %s %s %s %s %s %s",
    " unique receivers:",
    matched["nfl_id"].n_unique(),
    "| unique defenders:",
    matched["def_id"].n_unique(),
    "| unique pairs:",
    matched.select(["nfl_id", "def_id"]).n_unique(),
    "| targeted:",
    (matched["target"] == "TRUE").sum(),
)
logging.info("%s", matched.group_by("season_type").len())
logging.info(
    "%s",
    matched.group_by("def_pos")
    .agg(pl.len().alias("reps"), pl.col("def_id").n_unique().alias("defenders"))
    .sort("reps", descending=True),
)
logging.info(
    "%s",
    matched.group_by("nfl_position")
    .agg(pl.len().alias("reps"), pl.col("nfl_id").n_unique().alias("receivers"))
    .sort("reps", descending=True),
)
logging.info("%s", matched.group_by("team_coverage_man_zone").len())
logging.info(
    "%s",
    matched.group_by("coverage_assignment").len().sort("len", descending=True).head(8),
)
reps_per_def = matched.group_by("def_id").len()
reps_per_rec = matched.group_by("nfl_id").len()
for name, df in (("defender", reps_per_def), ("receiver", reps_per_rec)):
    logging.info(
        "%s",
        f" reps per {name}: median {df['len'].median()}, >=20: {(df['len'] >= 20).sum()}, >=50: {(df['len'] >= 50).sum()}, >=100: {(df['len'] >= 100).sum()}",
    )
matched.write_parquet(P + "matched_at_throw.parquet")
