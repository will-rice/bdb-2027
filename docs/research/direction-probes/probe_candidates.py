"""Feasibility probes for three candidate directions."""

import logging
import numpy as np
import polars as pl
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = "data/processed/direction_probes/"
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(30)
m = pl.read_parquet(P + "matched_at_throw.parquet").filter(
    pl.col("season_type") == "REG"
)
pp = pl.read_parquet(P + "pp.parquet")
players = pl.read_csv("data/raw/players.csv", null_values=["NA"])
name = dict(zip(players["nfl_id"], players["display_name"]))
logging.info("%s %s %s", "=" * 30, "A. defender-side coverage outcome", "=" * 30)
cb = m.filter(pl.col("def_pos") == "CB")
logging.info(
    "%s %s %s %s",
    "CB reps:",
    cb.height,
    "| MAN-assignment CB reps:",
    cb.filter(pl.col("coverage_assignment") == "MAN").height,
)
cb = cb.with_columns((pl.col("coverage_assignment") == "MAN").alias("is_man"))
cb = cb.with_columns(
    (
        pl.col("separation_at_pass_forward")
        - pl.col("separation_at_pass_forward").mean().over(["route_ran", "is_man"])
    ).alias("sep_resid")
)
rng = np.random.default_rng(0)


def split_half(df, key, value, min_reps):
    rs = []
    for _ in range(200):
        d = df.with_columns(pl.Series("h", rng.integers(0, 2, df.height)))
        a = d.group_by([key, "h"]).agg(
            pl.col(value).mean().alias("v"), pl.len().alias("n")
        )
        w = a.pivot(values=["v", "n"], index=key, on="h").drop_nulls()
        w = w.filter((pl.col("n_0") >= min_reps / 2) & (pl.col("n_1") >= min_reps / 2))
        if w.height > 5:
            rs.append(np.corrcoef(w["v_0"], w["v_1"])[0, 1])
    r = float(np.mean(rs))
    return (r, 2 * r / (1 + r), w.height)


for label, df in (("all CB reps", cb), ("MAN only", cb.filter(pl.col("is_man")))):
    for min_reps in (20, 40):
        r, sb, n = split_half(df, "def_id", "sep_resid", min_reps)
        logging.info(
            "%s",
            f"  split-half r of route-adjusted separation allowed, {label}, >= {min_reps} reps: r = {r:.2f}, Spearman-Brown = {sb:.2f}, defenders = {n}",
        )
r, sb, n = split_half(
    m.filter(pl.col("nfl_position") == "WR").with_columns(
        (
            pl.col("separation_at_pass_forward")
            - pl.col("separation_at_pass_forward").mean().over(["route_ran"])
        ).alias("sep_resid")
    ),
    "nfl_id",
    "sep_resid",
    20,
)
logging.info(
    "%s",
    f"  for comparison, receiver side in the same reps (WR, >= 20 reps): r = {r:.2f}, SB = {sb:.2f}, receivers = {n}",
)
top_pairs = (
    m.group_by(["nfl_id", "def_id"])
    .agg(
        pl.len().alias("reps"),
        pl.col("separation_at_pass_forward").mean().round(2).alias("mean_sep"),
        (pl.col("coverage_assignment") == "MAN").sum().alias("man_reps"),
    )
    .sort("reps", descending=True)
    .head(12)
)
logging.info(
    "%s",
    top_pairs.with_columns(
        pl.col("nfl_id").replace_strict(name, default="?").alias("receiver"),
        pl.col("def_id").replace_strict(name, default="?").alias("defender"),
    ).select(["receiver", "defender", "reps", "man_reps", "mean_sep"]),
)
logging.info(
    "%s %s %s",
    "=" * 30,
    "B. year-2 headroom: does Combine speed add beyond year-1 game speed?",
    "=" * 30,
)
gt = pl.scan_parquet(P + "gt_*.parquet")
games = (
    pl.read_csv("data/raw/games.csv")
    .filter(pl.col("season_type") == "REG")
    .select(["game_id"])
)
per_play = (
    gt.join(games.lazy(), on="game_id")
    .group_by(["season", "nfl_id", "game_id", "play_id"])
    .agg(pl.col("s").max().alias("play_max"))
    .collect()
)
season = (
    per_play.group_by(["season", "nfl_id"])
    .agg(
        pl.len().alias("plays"),
        pl.col("play_max").quantile(0.98).alias("game_speed"),
        pl.col("play_max").max().alias("game_max"),
    )
    .filter(pl.col("plays") >= 100)
)
ct = pl.scan_csv("data/raw/combine_tracking.csv").filter(
    (pl.col("entity_type") == "PLAYER") & (pl.col("drill_type") == "FORTY_YARD_DASH")
)
top = ct.group_by("nfl_id").agg(pl.col("s").max().alias("combine_speed")).collect()
base = players.select(["nfl_id", "nfl_position", "draft_year"]).join(top, on="nfl_id")
grp = {
    "WR": "WR/TE",
    "TE": "WR/TE",
    "CB": "DB",
    "SS": "DB",
    "FS": "DB",
    "DB": "DB",
    "T": "OL",
    "G": "OL",
    "C": "OL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "OLB": "DL",
}
base = base.with_columns(
    pl.col("nfl_position").replace_strict(grp, default="other").alias("grp")
).filter(pl.col("grp") != "other")
y1 = season.rename({"game_speed": "y1_speed", "plays": "y1_plays"}).select(
    ["season", "nfl_id", "y1_speed", "y1_plays"]
)
y2 = season.rename({"game_speed": "y2_speed", "plays": "y2_plays"}).select(
    ["season", "nfl_id", "y2_speed", "y2_plays"]
)
d = (
    base.join(y1, left_on=["nfl_id", "draft_year"], right_on=["nfl_id", "season"])
    .with_columns((pl.col("draft_year") + 1).alias("next"))
    .join(y2, left_on=["nfl_id", "next"], right_on=["nfl_id", "season"])
)
for c in ("combine_speed", "y1_speed", "y2_speed"):
    d = d.with_columns(
        (
            (pl.col(c) - pl.col(c).mean().over("grp")) / pl.col(c).std().over("grp")
        ).alias(c + "_z")
    )
df = d.to_pandas()
logging.info(
    "%s %s %s",
    "players with Combine 40 tracking, year-1 and year-2 game tracking (>=100 plays each):",
    len(df),
    dict(df["grp"].value_counts()),
)
logging.info(
    "%s",
    f"  within-group r: Combine speed vs Y1 game speed = {stats.pearsonr(df.combine_speed_z, df.y1_speed_z)[0]:.2f}; Y1 vs Y2 game speed = {stats.pearsonr(df.y1_speed_z, df.y2_speed_z)[0]:.2f}; Combine vs Y2 = {stats.pearsonr(df.combine_speed_z, df.y2_speed_z)[0]:.2f}",
)
import statsmodels.formula.api as smf

m1 = smf.ols("y2_speed_z ~ y1_speed_z", df).fit()
m2 = smf.ols("y2_speed_z ~ y1_speed_z + combine_speed_z", df).fit()
logging.info(
    "%s",
    f"  Y2 ~ Y1: R2 = {m1.rsquared:.3f} | Y2 ~ Y1 + Combine: R2 = {m2.rsquared:.3f}; Combine coef = {m2.params['combine_speed_z']:+.3f} (se {m2.bse['combine_speed_z']:.3f}, p = {m2.pvalues['combine_speed_z']:.2g}); Y1 coef = {m2.params['y1_speed_z']:+.3f}",
)
df["headroom"] = df.combine_speed_z - df.y1_speed_z
df["change"] = df.y2_speed_z - df.y1_speed_z
logging.info(
    "%s",
    f"  headroom (Combine z - Y1 z) vs Y1->Y2 change: r = {stats.pearsonr(df.headroom, df.change)[0]:.2f} (note: shares -Y1 on both sides, so regression to the mean inflates this)",
)
df["raw_ratio_y1"] = df.y1_speed / df.combine_speed
df["raw_ratio_y2"] = df.y2_speed / df.combine_speed
logging.info("%s", "  game speed as a share of Combine top speed, by group (Y1, Y2):")
logging.info("%s", df.groupby("grp")[["raw_ratio_y1", "raw_ratio_y2"]].mean().round(3))
logging.info("%s %s %s", "=" * 30, "C. trench rookie-vs-rookie reps", "=" * 30)
ol = pp.filter(
    pl.col("pass_rushers_encountered").is_not_null() & (pl.col("season_type") == "REG")
).select(
    [
        "game_id",
        "play_id",
        pl.col("nfl_id").alias("ol_id"),
        pl.col("team_abbr").alias("ol_team"),
        "pressure_allowed",
        "time_to_pressure_allowed",
        pl.col("nfl_position").alias("ol_pos"),
    ]
)
dl = pp.filter(
    pl.col("player_get_off").is_not_null()
    & pl.col("pass_result").is_not_null()
    & ~pl.col("on_offense")
    & (pl.col("season_type") == "REG")
).select(
    [
        "game_id",
        "play_id",
        pl.col("nfl_id").alias("dl_id"),
        pl.col("team_abbr").alias("dl_team"),
        "time_to_pressure",
        "quick_pressure",
        "sack",
        pl.col("nfl_position").alias("dl_pos"),
    ]
)
j = ol.join(dl, on=["game_id", "play_id"]).filter(
    pl.col("ol_team") != pl.col("dl_team")
)
logging.info(
    "%s %s %s %s",
    "pass plays with a tracked rookie blocker and a tracked rookie rusher on opposite sides (pairs):",
    j.height,
    "| plays:",
    j.select(["game_id", "play_id"]).n_unique(),
)
both = j.filter(
    (pl.col("pressure_allowed") == "TRUE") & pl.col("time_to_pressure").is_not_null()
)
exact = both.filter(
    (pl.col("time_to_pressure") - pl.col("time_to_pressure_allowed")).abs() < 0.05
)
logging.info(
    "%s %s %s %s",
    "pairs where the blocker allowed pressure and the rusher got pressure:",
    both.height,
    "| with identical pressure times (confirmed blocker-rusher matchup):",
    exact.height,
)
snap1 = (
    gt.filter(pl.col("event") == "ball_snap")
    .select(["game_id", "play_id", "nfl_id", "x", "y"])
    .collect()
)
jj = (
    j.join(
        snap1.rename({"nfl_id": "ol_id", "x": "ox", "y": "oy"}),
        on=["game_id", "play_id", "ol_id"],
    )
    .join(
        snap1.rename({"nfl_id": "dl_id", "x": "dx", "y": "dy"}),
        on=["game_id", "play_id", "dl_id"],
    )
    .with_columns(
        ((pl.col("ox") - pl.col("dx")) ** 2 + (pl.col("oy") - pl.col("dy")) ** 2)
        .sqrt()
        .alias("snap_dist")
    )
)
logging.info(
    "%s %s %s %s %s %s",
    "pairs lined up within 2.5 yards at the snap (likely direct matchups):",
    jj.filter(pl.col("snap_dist") <= 2.5).height,
    "| unique blockers:",
    jj.filter(pl.col("snap_dist") <= 2.5)["ol_id"].n_unique(),
    "| unique rushers:",
    jj.filter(pl.col("snap_dist") <= 2.5)["dl_id"].n_unique(),
)
if exact.height:
    e = exact.join(
        jj.select(["game_id", "play_id", "ol_id", "dl_id", "snap_dist"]),
        on=["game_id", "play_id", "ol_id", "dl_id"],
        how="left",
    )
    logging.info(
        "%s %s",
        "  snap distance for confirmed pressure matchups, quantiles (10/50/90):",
        [round(e["snap_dist"].quantile(q), 2) for q in (0.1, 0.5, 0.9)],
    )
