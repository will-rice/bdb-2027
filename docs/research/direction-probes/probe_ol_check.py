"""Check the OL in-game result within position and without same-play circularity."""

import logging
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = "data/processed/direction_probes/"
pp = pl.read_parquet(P + "pp.parquet").filter(pl.col("season_type") == "REG")
ol = pp.filter(
    pl.col("pass_rushers_encountered").is_not_null()
    & pl.col("nfl_position").is_in(["T", "G", "C"])
).select(
    [
        "game_id",
        "play_id",
        "nfl_id",
        "nfl_position",
        "play_direction",
        "peak_pressure_probability_allowed",
        "pressure_allowed",
        "season",
        "draft_year",
    ]
)
gt = pl.scan_parquet(P + "gt_*.parquet").select(
    ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"]
)
tr = gt.join(
    ol.lazy().select(["game_id", "play_id", "nfl_id"]),
    on=["game_id", "play_id", "nfl_id"],
).collect()
snap = (
    tr.filter(pl.col("event") == "ball_snap")
    .select(
        [
            "game_id",
            "play_id",
            "nfl_id",
            pl.col("time").alias("t0"),
            pl.col("x").alias("x0"),
            pl.col("y").alias("y0"),
        ]
    )
    .unique(["game_id", "play_id", "nfl_id"])
)
tr = tr.join(snap, on=["game_id", "play_id", "nfl_id"]).with_columns(
    ((pl.col("time") - pl.col("t0")).dt.total_milliseconds() / 1000).round(1).alias("t")
)
play = (
    tr.filter((pl.col("t") >= 0) & (pl.col("t") <= 1.5))
    .group_by(["game_id", "play_id", "nfl_id"])
    .agg(
        pl.col("s").max().alias("peak_speed"),
        pl.col("x").filter(pl.col("t") == 1.0).first().alias("x1"),
        pl.col("x0").first(),
    )
    .drop_nulls()
)
play = play.join(ol, on=["game_id", "play_id", "nfl_id"]).with_columns(
    pl.when(pl.col("play_direction") == "right")
    .then(pl.col("x0") - pl.col("x1"))
    .otherwise(pl.col("x1") - pl.col("x0"))
    .alias("depth"),
    (pl.col("play_id") % 2).alias("h"),
    pl.col("peak_pressure_probability_allowed").alias("ppp"),
)
h = play.group_by(["nfl_id", "h"]).agg(
    pl.len().alias("n"),
    pl.col("depth").mean(),
    pl.col("peak_speed").mean(),
    pl.col("ppp").mean(),
    pl.col("nfl_position").first().alias("pos"),
)
w = (
    h.pivot(values=["n", "depth", "peak_speed", "ppp"], index=["nfl_id", "pos"], on="h")
    .drop_nulls()
    .filter((pl.col("n_0") >= 100) & (pl.col("n_1") >= 100))
    .to_pandas()
)
logging.info(
    "%s %s %s",
    "players with >= 100 odd and >= 100 even pass-pro snaps:",
    len(w),
    dict(w.pos.value_counts()),
)
for c in ("depth", "peak_speed", "ppp"):
    for half in ("0", "1"):
        w[f"{c}_{half}_c"] = w[f"{c}_{half}"] - w.groupby("pos")[
            f"{c}_{half}"
        ].transform("mean")
    logging.info(
        "%s",
        f"  within-position split-half r of {c}: {np.corrcoef(w[c + '_0_c'], w[c + '_1_c'])[0, 1]:.2f}",
    )
logging.info(
    "%s",
    "  no shared plays: set trait on odd plays vs pressure probability allowed on even plays, centred within position",
)
for c in ("depth", "peak_speed"):
    r, p = stats.pearsonr(w[c + "_0_c"], w["ppp_1_c"])
    logging.info("%s", f"    {c:10s} r = {r:+.2f} (p = {p:.4f})")
for pos in ("T", "G"):
    s = w[w.pos == pos]
    for c in ("depth", "peak_speed"):
        r, p = stats.pearsonr(s[c + "_0"], s["ppp_1"])
        logging.info(
            "%s", f"    {pos} only: {c:10s} r = {r:+.2f} (p = {p:.3f}, n = {len(s)})"
        )
order = play.sort(["nfl_id", "game_id", "play_id"]).with_columns(
    pl.int_range(pl.len()).over("nfl_id").alias("k")
)
early = (
    order.filter(pl.col("k") < 50)
    .group_by("nfl_id")
    .agg(
        pl.col("peak_speed").mean().alias("e_speed"),
        pl.col("depth").mean().alias("e_depth"),
        pl.col("ppp").mean().alias("e_ppp"),
        pl.col("nfl_position").first().alias("pos"),
    )
)
late = (
    order.filter(pl.col("k") >= 50)
    .group_by("nfl_id")
    .agg(pl.col("ppp").mean().alias("l_ppp"), pl.len().alias("n_late"))
    .filter(pl.col("n_late") >= 150)
)
e = early.join(late, on="nfl_id").to_pandas()
for c in ("e_speed", "e_depth", "e_ppp", "l_ppp"):
    e[c + "_c"] = e[c] - e.groupby("pos")[c].transform("mean")
logging.info(
    "%s",
    f"  first 50 career pass-pro snaps -> pressure probability allowed on all later snaps (n = {len(e)}, centred within position):",
)
for c in ("e_speed", "e_depth", "e_ppp"):
    r, p = stats.pearsonr(e[c + "_c"], e["l_ppp_c"])
    logging.info("%s", f"    {c:8s} r = {r:+.2f} (p = {p:.4f})")
f = smf.ols("l_ppp_c ~ e_ppp_c + e_speed_c", e).fit()
logging.info(
    "%s",
    f"    later ppp ~ early ppp + early set speed: t(early set speed) = {f.tvalues['e_speed_c']:+.2f} (p = {f.pvalues['e_speed_c']:.4f}), R2 = {f.rsquared:.2f} vs early ppp alone R2 = {smf.ols('l_ppp_c ~ e_ppp_c', e).fit().rsquared:.2f}",
)
