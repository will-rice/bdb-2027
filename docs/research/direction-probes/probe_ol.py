"""Probe: is the offensive-line mirror drill a reliable, weight-independent lead on pass protection?"""

import logging
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = "data/processed/direction_probes/"
pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(50)
pp = pl.read_parquet(P + "pp.parquet").filter(pl.col("season_type") == "REG")
ol = pp.filter(
    pl.col("pass_rushers_encountered").is_not_null()
    & pl.col("nfl_position").is_in(["T", "G", "C"])
)
logging.info(
    "%s %s %s %s",
    "OL pass-protection snaps:",
    ol.height,
    "| players:",
    ol["nfl_id"].n_unique(),
)
ol = ol.with_columns(
    (pl.col("pressure_allowed") == "TRUE").cast(pl.Float64).alias("press"),
    (pl.col("play_id") % 2).alias("h"),
)
h = ol.group_by(["nfl_id", "h"]).agg(
    pl.len().alias("n"),
    pl.col("press").mean().alias("press"),
    pl.col("peak_pressure_probability_allowed").mean().alias("ppp"),
)
for v in ("press", "ppp"):
    w = (
        h.pivot(values=[v, "n"], index="nfl_id", on="h")
        .drop_nulls()
        .filter((pl.col("n_0") >= 100) & (pl.col("n_1") >= 100))
    )
    r = np.corrcoef(w[v + "_0"], w[v + "_1"])[0, 1]
    logging.info(
        "%s",
        f"  split-half r of player {v} (odd/even plays, >= 100 each): {r:.2f}, Spearman-Brown {2 * r / (1 + r):.2f}, n = {w.height}",
    )
out = (
    ol.group_by("nfl_id")
    .agg(
        pl.len().alias("snaps"),
        pl.col("press").mean().alias("press"),
        pl.col("peak_pressure_probability_allowed").mean().alias("ppp"),
        pl.col("nfl_position").first().alias("pos"),
    )
    .filter(pl.col("snaps") >= 200)
)
ct = pl.scan_csv("data/raw/combine_tracking.csv").filter(
    (pl.col("entity_type") == "PLAYER") & (pl.col("drill_type") == "SKILL_DRILLS_OL")
)
logging.info(
    "%s",
    ct.group_by(["drill_name", "draft_year"])
    .agg(
        pl.col("event_id").n_unique().alias("reps"),
        pl.col("nfl_id").n_unique().alias("players"),
    )
    .sort(["drill_name", "draft_year"])
    .collect(),
)
d = (
    ct.select(["nfl_id", "event_id", "draft_year", "drill_name", "time", "x", "y", "s"])
    .collect()
    .sort(["event_id", "time"])
)
d = d.with_columns(
    pl.col("s").rolling_mean(3, center=True).over("event_id").alias("ss")
)
d = d.with_columns(
    (
        (
            pl.col("ss").shift(-1).over("event_id")
            - pl.col("ss").shift(1).over("event_id")
        )
        / 0.2
    ).alias("acc")
)
rep = d.group_by(["nfl_id", "event_id", "draft_year", "drill_name"]).agg(
    pl.len().alias("frames"),
    pl.col("s").max().alias("peak_speed"),
    pl.col("s").quantile(0.9).alias("p90_speed"),
    pl.col("acc").max().alias("peak_accel"),
    (-pl.col("acc").min()).alias("peak_brake"),
    pl.col("acc").abs().mean().alias("mean_abs_acc"),
)
mets = ["peak_speed", "p90_speed", "peak_accel", "peak_brake", "mean_abs_acc"]
for c in mets:
    rep = rep.with_columns(
        (
            (pl.col(c) - pl.col(c).mean().over(["drill_name", "draft_year"]))
            / pl.col(c).std().over(["drill_name", "draft_year"])
        ).alias(c + "_z")
    )
res = pl.read_csv("data/raw/combine_results.csv", null_values=["NA"]).select(
    ["nfl_id", "forty", "combine_weight"]
)
pk = (
    pl.read_csv("data/raw/players.csv", null_values=["NA"])
    .select(["nfl_id", "draft_overall_pick"])
    .with_columns(pl.col("draft_overall_pick").fill_null(300).log().alias("log_pick"))
)


def test(drill_filter, label):
    r_ = rep.filter(drill_filter)
    two = (
        r_.sort("event_id")
        .group_by(["nfl_id", "drill_name"])
        .agg(
            [pl.col(c + "_z").first().alias(c + "_a") for c in mets]
            + [pl.col(c + "_z").last().alias(c + "_b") for c in mets]
            + [pl.len().alias("k")]
        )
        .filter(pl.col("k") >= 2)
    )
    player = r_.group_by("nfl_id").agg(
        [pl.col(c + "_z").mean().alias(c) for c in mets] + [pl.len().alias("reps")]
    )
    j = (
        out.join(player, on="nfl_id")
        .join(res, on="nfl_id")
        .join(pk, on="nfl_id")
        .to_pandas()
    )
    logging.info(
        "%s",
        f"\n{label}: players with outcome = {len(j)}; players with two reps of the same drill = {two.height}",
    )
    for c in mets:
        rel = (
            np.corrcoef(two[c + "_a"], two[c + "_b"])[0, 1]
            if two.height > 10
            else float("nan")
        )
        line = f"  {c:13s} rep-to-rep r = {rel:+.2f}"
        for o in ("ppp", "press"):
            r, p = stats.pearsonr(j[c], j[o])
            f = smf.ols(f"{o} ~ {c} + combine_weight + C(pos) + log_pick", j).fit()
            line += f" | {o}: r = {r:+.2f} (p={p:.3f}); adj. t = {f.tvalues[c]:+.2f} (p={f.pvalues[c]:.3f})"
        logging.info("%s", line)
    return j


j = test(pl.col("drill_name").str.contains("MIRROR"), "PASS PRO MIRROR drill only")
test(pl.col("drill_name").str.contains("MIRROR|WAVE"), "MIRROR + WAVE drills")
test(pl.lit(True), "all OL position drills")
jj = out.join(res, on="nfl_id").join(pk, on="nfl_id").to_pandas()
for f in ("combine_weight", "forty", "log_pick"):
    s = jj.dropna(subset=[f])
    logging.info(
        "%s",
        f"  baseline {f:15s}: ppp r = {stats.pearsonr(s[f], s.ppp)[0]:+.2f}, press r = {stats.pearsonr(s[f], s.press)[0]:+.2f} (n = {len(s)})",
    )
