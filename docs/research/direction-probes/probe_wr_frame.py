"""Probe: for receivers, is separation a frame-and-speed story, does it hold out of class, and is it role or skill?"""

import logging
import warnings
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")
P = "data/processed/direction_probes/"
pp = pl.read_parquet(P + "pp.parquet").filter(pl.col("season_type") == "REG")
res = pl.read_csv("data/raw/combine_results.csv", null_values=["NA"]).select(
    ["nfl_id", "forty", "combine_weight", "combine_height"]
)
ct = pl.scan_csv("data/raw/combine_tracking.csv").filter(
    pl.col("entity_type") == "PLAYER"
)
top = (
    ct.filter(pl.col("drill_type") == "FORTY_YARD_DASH")
    .group_by("nfl_id")
    .agg(pl.col("s").max().alias("top_speed"))
    .collect()
)
routes = (
    ct.filter(pl.col("drill_name").str.contains("ROUTE"))
    .select(["nfl_id", "event_id", "time", "s"])
    .collect()
    .sort(["event_id", "time"])
)
routes = routes.with_columns(
    (
        (pl.col("s").shift(-1).over("event_id") - pl.col("s").shift(1).over("event_id"))
        / 0.2
    ).alias("ds")
)
brake = (
    routes.group_by(["nfl_id", "event_id"])
    .agg(pl.col("ds").min().alias("b"), pl.col("s").max().alias("v"))
    .group_by("nfl_id")
    .agg(
        pl.col("b").mean().alias("route_brake"), pl.col("v").mean().alias("route_speed")
    )
)
feat = res.join(top, on="nfl_id", how="left").join(brake, on="nfl_id", how="left")
wr = pp.filter(
    (pl.col("nfl_position") == "WR")
    & pl.col("route_ran").is_not_null()
    & pl.col("separation_at_pass_forward").is_not_null()
)
wr = wr.with_columns(
    (pl.col("lined_up_position") == "SLOT_WR").cast(pl.Int8).alias("slot"),
    (pl.col("team_coverage_man_zone") == "MAN_COVERAGE").cast(pl.Int8).alias("man"),
    (
        pl.col("separation_at_pass_forward")
        - pl.col("separation_at_pass_forward")
        .mean()
        .over(["route_ran", "lined_up_position", "team_coverage_man_zone"])
    ).alias("sep_adj"),
)
per = (
    wr.group_by(["nfl_id", "draft_year"])
    .agg(
        pl.len().alias("routes"),
        pl.col("separation_at_pass_forward").mean().alias("sep"),
        pl.col("sep_adj").mean().alias("sep_adj"),
        pl.col("slot").mean().alias("slot_share"),
        pl.col("route_ran")
        .is_in(["GO", "POST", "CORNER"])
        .mean()
        .alias("vertical_share"),
    )
    .filter(pl.col("routes") >= 100)
    .join(feat, on="nfl_id")
    .to_pandas()
)
logging.info(
    "%s",
    f"WRs with >= 100 routes with a recorded separation: {len(per)} | by class: {dict(per.draft_year.value_counts().sort_index())}",
)
logging.info(
    "%s",
    "player-level correlations (raw separation | separation adjusted for route x alignment x man/zone | slot share):",
)
for f in (
    "combine_weight",
    "combine_height",
    "top_speed",
    "forty",
    "route_brake",
    "route_speed",
):
    s = per.dropna(subset=[f])
    out = f"  {f:15s} n={len(s):2d}"
    for o in ("sep", "sep_adj", "slot_share", "vertical_share"):
        r, p = stats.pearsonr(s[f], s[o])
        out += f" | {o}: {r:+.2f} (p={p:.3f})"
    logging.info("%s", out)
s = per.dropna(subset=["top_speed", "route_brake"]).copy()
logging.info(
    "%s", f"\nleave-one-draft-class-out prediction of player separation (n = {len(s)}):"
)


def lodo(formula, target):
    pred = np.zeros(len(s))
    for y in sorted(s.draft_year.unique()):
        tr, te = (s[s.draft_year != y], s[s.draft_year == y])
        pred[(s.draft_year == y).values] = (
            smf.ols(f"{target} ~ {formula}", tr).fit().predict(te)
        )
    r = stats.pearsonr(pred, s[target])[0]
    r2 = (
        1
        - ((s[target] - pred) ** 2).sum() / ((s[target] - s[target].mean()) ** 2).sum()
    )
    return (r, r2)


for target in ("sep", "sep_adj"):
    for label, formula in (
        ("weight + height", "combine_weight + combine_height"),
        ("+ tracked top speed", "combine_weight + combine_height + top_speed"),
        (
            "+ route-drill braking and speed",
            "combine_weight + combine_height + top_speed + route_brake + route_speed",
        ),
        ("official 40 only", "forty"),
    ):
        ss = s.dropna(subset=["forty"]) if "forty" in formula else s
        r, r2 = lodo(formula, target) if "forty" not in formula else (np.nan, np.nan)
        if "forty" in formula:
            pred = np.zeros(len(ss))
            for y in sorted(ss.draft_year.unique()):
                tr, te = (ss[ss.draft_year != y], ss[ss.draft_year == y])
                pred[(ss.draft_year == y).values] = (
                    smf.ols(f"{target} ~ forty", tr).fit().predict(te)
                )
            r = stats.pearsonr(pred, ss[target])[0]
            r2 = (
                1
                - ((ss[target] - pred) ** 2).sum()
                / ((ss[target] - ss[target].mean()) ** 2).sum()
            )
        logging.info(
            "%s",
            f"  {target:8s} ~ {label:32s} out-of-class r = {r:+.2f}, R2 = {r2:+.3f}",
        )
f = smf.ols("sep_adj ~ combine_weight + combine_height + top_speed", s).fit()
logging.info(
    "%s %s %s",
    "\nin-sample, role-adjusted separation ~ weight + height + top speed:",
    {
        k: f"{f.tvalues[k]:+.2f} (p={f.pvalues[k]:.3f})"
        for k in ("combine_weight", "combine_height", "top_speed")
    },
    f"R2 = {f.rsquared:.2f}",
)
logging.info(
    "%s %s %s %s %s %s %s",
    "correlation weight-height:",
    round(stats.pearsonr(s.combine_weight, s.combine_height)[0], 2),
    "| SD of player sep:",
    round(per.sep.std(), 2),
    "yd | SD of role-adjusted:",
    round(per.sep_adj.std(), 2),
    "yd",
)
