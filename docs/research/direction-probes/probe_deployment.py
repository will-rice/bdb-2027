"""Probe: do Combine traits show up in deployment (routes, cushion, matchups) rather than raw outcomes?"""

import logging
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = "data/processed/direction_probes/"
pp = pl.read_parquet(P + "pp.parquet").filter(pl.col("season_type") == "REG")
res = pl.read_csv("data/raw/combine_results.csv", null_values=["NA"]).select(
    ["nfl_id", "forty", "ten_yd_split", "combine_weight", "combine_height"]
)
players = pl.read_csv("data/raw/players.csv", null_values=["NA"]).select(
    ["nfl_id", "draft_overall_pick"]
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
feat = res.join(top, on="nfl_id", how="left").join(players, on="nfl_id", how="left")
logging.info("%s %s %s", "=" * 25, "WR deployment", "=" * 25)
wr = pp.filter((pl.col("nfl_position") == "WR") & pl.col("route_ran").is_not_null())
logging.info(
    "%s %s",
    "WR lined_up_position values:",
    wr.group_by("lined_up_position").len().sort("len", descending=True).head(8).rows(),
)
VERT = ["GO", "POST", "CORNER"]
per = (
    wr.group_by("nfl_id")
    .agg(
        pl.len().alias("routes"),
        pl.col("route_ran").is_in(VERT).mean().alias("vertical_share"),
        (pl.col("route_ran") == "GO").mean().alias("go_share"),
        pl.col("route_ran")
        .is_in(["HITCH", "SCREEN", "FLAT"])
        .mean()
        .alias("short_share"),
        pl.col("cushion").mean().alias("cushion"),
        pl.col("lined_up_position").str.contains("SLOT").mean().alias("slot_share"),
        pl.col("separation_at_pass_forward").mean().alias("sep"),
        (pl.col("team_coverage_man_zone") == "MAN_COVERAGE").mean().alias("man_faced"),
    )
    .filter(pl.col("routes") >= 100)
    .join(feat, on="nfl_id")
    .to_pandas()
)
logging.info("%s", f"WRs with >= 100 routes: {len(per)}")
for outcome in (
    "vertical_share",
    "go_share",
    "short_share",
    "cushion",
    "slot_share",
    "sep",
):
    line = f"  {outcome:15s}"
    for f in ("top_speed", "forty", "combine_weight", "combine_height"):
        s = per.dropna(subset=[f, outcome])
        r, p = stats.pearsonr(s[f], s[outcome])
        line += f" | {f} r = {r:+.2f} (p={p:.3f}, n={len(s)})"
    logging.info("%s", line)
w2 = wr.with_columns((pl.col("play_id") % 2).alias("h"))
h = w2.group_by(["nfl_id", "h"]).agg(
    pl.len().alias("n"),
    pl.col("route_ran").is_in(VERT).mean().alias("vertical_share"),
    pl.col("cushion").mean().alias("cushion"),
    pl.col("separation_at_pass_forward").mean().alias("sep"),
)
for v in ("vertical_share", "cushion", "sep"):
    w = (
        h.pivot(values=[v, "n"], index="nfl_id", on="h")
        .drop_nulls()
        .filter((pl.col("n_0") >= 50) & (pl.col("n_1") >= 50))
    )
    logging.info(
        "%s",
        f"  split-half r of player {v}: {np.corrcoef(w[v + '_0'], w[v + '_1'])[0, 1]:.2f} (n = {w.height})",
    )
logging.info(
    "%s",
    "\n--- play level: does speed 'show up' in separation once deployment is held fixed? (cluster by receiver)",
)
d = (
    wr.filter(
        pl.col("separation_at_pass_forward").is_not_null()
        & pl.col("cushion").is_not_null()
    )
    .join(feat, on="nfl_id")
    .to_pandas()
    .dropna(subset=["top_speed", "combine_weight"])
)
d["sep"] = d["separation_at_pass_forward"]
d["speed_z"] = (d.top_speed - d.top_speed.mean()) / d.top_speed.std()
d["weight_z"] = (d.combine_weight - d.combine_weight.mean()) / d.combine_weight.std()
d["man"] = (d.team_coverage_man_zone == "MAN_COVERAGE").astype(int)


def fit(formula, label):
    m = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d["nfl_id"]})
    logging.info(
        "%s",
        f"  {label:58s} speed_z coef {m.params['speed_z']:+.3f} (se {m.bse['speed_z']:.3f}, p = {m.pvalues['speed_z']:.3f}) | n = {int(m.nobs)}, R2 = {m.rsquared:.3f}",
    )


fit("sep ~ speed_z", "raw")
fit("sep ~ speed_z + weight_z", "+ weight")
fit("sep ~ speed_z + weight_z + C(route_ran)", "+ weight + route")
fit(
    "sep ~ speed_z + weight_z + C(route_ran) + cushion + man",
    "+ weight + route + cushion + man/zone",
)
m = smf.ols("cushion ~ speed_z + weight_z + C(route_ran) + man", d).fit(
    cov_type="cluster", cov_kwds={"groups": d["nfl_id"]}
)
logging.info(
    "%s",
    f"  cushion ~ speed_z + weight + route + man: speed_z coef {m.params['speed_z']:+.3f} yd per SD (se {m.bse['speed_z']:.3f}, p = {m.pvalues['speed_z']:.4f})",
)
logging.info("%s", f"  1 SD of WR top speed = {d.top_speed.std():.2f} yd/s")
logging.info(
    "%s %s %s",
    "\n" + "=" * 25,
    "matchup assignment: does speed get matched with speed?",
    "=" * 25,
)
mm = pl.read_parquet(P + "matched_at_throw.parquet").filter(
    (pl.col("season_type") == "REG")
    & (pl.col("def_pos") == "CB")
    & (pl.col("nfl_position") == "WR")
)
mm = mm.join(top.rename({"top_speed": "wr_speed"}), on="nfl_id").join(
    top.rename({"nfl_id": "def_id", "top_speed": "cb_speed"}), on="def_id"
)
for label, sub in (
    ("all confirmed WR-CB reps", mm),
    ("MAN assignment", mm.filter(pl.col("coverage_assignment") == "MAN")),
):
    pairs = sub.group_by(["nfl_id", "def_id"]).agg(
        pl.len().alias("reps"), pl.col("wr_speed").first(), pl.col("cb_speed").first()
    )
    r, p = stats.pearsonr(sub["wr_speed"], sub["cb_speed"])
    logging.info(
        "%s",
        f"  {label}: reps = {sub.height}, pairs = {pairs.height}, rep-level r(WR speed, CB speed) = {r:+.3f}",
    )
logging.info("%s %s %s", "\n" + "=" * 25, "CB and DL deployment", "=" * 25)
cb = pp.filter(
    (pl.col("nfl_position") == "CB")
    & pl.col("coverage_assignment").is_not_null()
    & (pl.col("coverage_assignment") != "NO_ASSIGNMENT")
)
cbp = (
    cb.group_by("nfl_id")
    .agg(
        pl.len().alias("snaps"),
        (pl.col("coverage_assignment") == "MAN").mean().alias("man_share"),
        pl.col("coverage_assignment")
        .str.contains("DEEP|QUARTERS")
        .mean()
        .alias("deep_share"),
    )
    .filter(pl.col("snaps") >= 100)
    .join(feat, on="nfl_id")
    .to_pandas()
)
for outcome in ("man_share", "deep_share"):
    line = f"  CB {outcome:11s}"
    for f in ("top_speed", "forty", "combine_weight"):
        s = cbp.dropna(subset=[f])
        r, p = stats.pearsonr(s[f], s[outcome])
        line += f" | {f} r = {r:+.2f} (p={p:.3f}, n={len(s)})"
    logging.info("%s", line)
dl = pp.filter(
    pl.col("nfl_position").is_in(["DE", "DT", "NT", "OLB"])
    & pl.col("pass_result").is_not_null()
)
logging.info(
    "%s %s",
    "  DL lined_up_position values:",
    dl.group_by("lined_up_position").len().sort("len", descending=True).head(10).rows(),
)
