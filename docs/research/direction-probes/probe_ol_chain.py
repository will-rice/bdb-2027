"""Probe: an offensive-line chain. Combine pass-set drill -> in-game pass set -> pressure probability allowed."""

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
        "dropback_duration",
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
win = tr.filter((pl.col("t") >= 0) & (pl.col("t") <= 2.0))
play = (
    win.group_by(["game_id", "play_id", "nfl_id"])
    .agg(
        pl.col("s").filter(pl.col("t") <= 1.5).max().alias("peak_speed_15"),
        pl.col("x").filter(pl.col("t") == 1.0).first().alias("x1"),
        pl.col("y").filter(pl.col("t") == 1.0).first().alias("y1"),
        pl.col("x0").first(),
        pl.col("y0").first(),
    )
    .drop_nulls()
)
play = play.join(ol, on=["game_id", "play_id", "nfl_id"]).with_columns(
    ((pl.col("x1") - pl.col("x0")) ** 2 + (pl.col("y1") - pl.col("y0")) ** 2)
    .sqrt()
    .alias("disp_1s"),
    pl.when(pl.col("play_direction") == "right")
    .then(pl.col("x0") - pl.col("x1"))
    .otherwise(pl.col("x1") - pl.col("x0"))
    .alias("depth_1s"),
    (pl.col("y1") - pl.col("y0")).abs().alias("width_1s"),
    (pl.col("play_id") % 2).alias("h"),
)
logging.info(
    "%s %s %s %s",
    "OL pass-pro snaps with a tracked first second:",
    play.height,
    "| players:",
    play["nfl_id"].n_unique(),
)
logging.info(
    "%s",
    play.group_by("nfl_position")
    .agg(
        pl.len().alias("snaps"),
        pl.col("depth_1s").mean().round(2),
        pl.col("width_1s").mean().round(2),
        pl.col("disp_1s").mean().round(2),
        pl.col("peak_speed_15").mean().round(2),
    )
    .sort("nfl_position"),
)
game_mets = ["disp_1s", "depth_1s", "width_1s", "peak_speed_15"]
h = play.group_by(["nfl_id", "h"]).agg(
    [pl.len().alias("n")] + [pl.col(c).mean().alias(c) for c in game_mets]
)
logging.info(
    "%s",
    "in-game pass-set traits: split-half reliability (odd/even plays, >= 100 each)",
)
for c in game_mets:
    w = (
        h.pivot(values=[c, "n"], index="nfl_id", on="h")
        .drop_nulls()
        .filter((pl.col("n_0") >= 100) & (pl.col("n_1") >= 100))
    )
    logging.info(
        "%s",
        f"  {c:14s} r = {np.corrcoef(w[c + '_0'], w[c + '_1'])[0, 1]:.2f} (n = {w.height})",
    )
player = (
    play.group_by("nfl_id")
    .agg(
        [
            pl.len().alias("snaps"),
            pl.col("nfl_position").first().alias("pos"),
            pl.col("peak_pressure_probability_allowed").mean().alias("ppp"),
        ]
        + [pl.col(c).mean().alias("g_" + c) for c in game_mets]
    )
    .filter(pl.col("snaps") >= 200)
)
ct = (
    pl.scan_csv("data/raw/combine_tracking.csv")
    .filter(
        (pl.col("entity_type") == "PLAYER")
        & (pl.col("drill_type") == "SKILL_DRILLS_OL")
    )
    .select(["nfl_id", "event_id", "draft_year", "drill_name", "time", "x", "y", "s"])
    .collect()
)
ct = ct.with_columns(
    pl.col("drill_name").str.replace("PASS_PRO-MIRROR", "PASS_PRO_MIRROR"),
    pl.col("time").str.to_datetime(strict=False),
).sort(["event_id", "time"])
onset = (
    ct.filter(pl.col("s") >= 0.5)
    .group_by("event_id")
    .agg(pl.col("time").min().alias("t_on"))
)
c2 = (
    ct.join(onset, on="event_id")
    .with_columns(
        ((pl.col("time") - pl.col("t_on")).dt.total_milliseconds() / 1000)
        .round(1)
        .alias("t")
    )
    .filter((pl.col("t") >= 0) & (pl.col("t") <= 1.5))
)
rep = (
    c2.group_by(["nfl_id", "event_id", "draft_year", "drill_name"])
    .agg(
        pl.col("s").max().alias("c_peak_speed_15"),
        pl.col("x").filter(pl.col("t") == 0).first().alias("xa"),
        pl.col("y").filter(pl.col("t") == 0).first().alias("ya"),
        pl.col("x").filter(pl.col("t") == 1.0).first().alias("xb"),
        pl.col("y").filter(pl.col("t") == 1.0).first().alias("yb"),
    )
    .drop_nulls()
)
rep = rep.with_columns(
    ((pl.col("xb") - pl.col("xa")) ** 2 + (pl.col("yb") - pl.col("ya")) ** 2)
    .sqrt()
    .alias("c_disp_1s")
)
cm = ["c_disp_1s", "c_peak_speed_15"]
for c in cm:
    rep = rep.with_columns(
        (
            (pl.col(c) - pl.col(c).mean().over(["drill_name", "draft_year"]))
            / pl.col(c).std().over(["drill_name", "draft_year"])
        ).alias(c + "_z")
    )
res = pl.read_csv("data/raw/combine_results.csv", null_values=["NA"]).select(
    ["nfl_id", "forty", "ten_yd_split", "combine_weight"]
)
pk = (
    pl.read_csv("data/raw/players.csv", null_values=["NA"])
    .select(["nfl_id", "draft_overall_pick"])
    .with_columns(pl.col("draft_overall_pick").fill_null(300).log().alias("log_pick"))
)
logging.info(
    "%s",
    "\nCombine first-second burst by drill: rep-to-rep reliability and links (adj. = weight + position + log pick)",
)
for drill in [
    "PASS_RUSH_DROP",
    "PASS_PRO_MIRROR_DRILL",
    "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",
    "ALL_OL_DRILLS",
]:
    r_ = rep if drill == "ALL_OL_DRILLS" else rep.filter(pl.col("drill_name") == drill)
    two = (
        r_.sort("event_id")
        .group_by(["nfl_id", "drill_name"])
        .agg(
            [pl.col(c + "_z").first().alias(c + "_a") for c in cm]
            + [pl.col(c + "_z").last().alias(c + "_b") for c in cm]
            + [pl.len().alias("k")]
        )
        .filter(pl.col("k") >= 2)
    )
    pl_ = r_.group_by("nfl_id").agg([pl.col(c + "_z").mean().alias(c) for c in cm])
    j = (
        player.join(pl_, on="nfl_id")
        .join(res, on="nfl_id")
        .join(pk, on="nfl_id")
        .to_pandas()
    )
    for c in cm:
        rel = (
            np.corrcoef(two[c + "_a"], two[c + "_b"])[0, 1]
            if two.height > 10
            else float("nan")
        )
        line = f"  {drill[:22]:22s} {c:16s} rep-to-rep r = {rel:+.2f} (n={two.height:3d}) | n = {len(j)}"
        for o in ("g_disp_1s", "g_peak_speed_15", "ppp"):
            f = smf.ols(f"{o} ~ {c} + combine_weight + C(pos) + log_pick", j).fit()
            line += f" | {o}: r = {stats.pearsonr(j[c], j[o])[0]:+.2f}, adj. t = {f.tvalues[c]:+.2f} (p={f.pvalues[c]:.3f})"
        logging.info("%s", line)
jp = player.join(res, on="nfl_id").join(pk, on="nfl_id").to_pandas()
logging.info(
    "%s %s %s",
    "\nin-game pass set -> pressure probability allowed (players >= 200 snaps, n =",
    len(jp),
    "), adj. = weight + position + log pick",
)
for c in ["g_" + m for m in game_mets]:
    f = smf.ols(f"ppp ~ {c} + combine_weight + C(pos) + log_pick", jp).fit()
    logging.info(
        "%s",
        f"  {c:18s} r = {stats.pearsonr(jp[c], jp.ppp)[0]:+.2f} | adj. t = {f.tvalues[c]:+.2f} (p = {f.pvalues[c]:.4f})",
    )
s = jp.dropna(subset=["ten_yd_split"])
logging.info(
    "%s",
    f"  official 10-yd split -> game disp_1s: r = {stats.pearsonr(s.ten_yd_split, s.g_disp_1s)[0]:+.2f} (n = {len(s)}); weight -> game disp_1s: r = {stats.pearsonr(jp.combine_weight, jp.g_disp_1s)[0]:+.2f}",
)
