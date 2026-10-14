"""Check the receiver-vs-corner variance split: man coverage only, with alignment in the context, plus catch outcome."""

import logging
import warnings
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")
P = "data/processed/direction_probes/"
m = pl.read_parquet(P + "matched_at_throw.parquet").filter(
    (pl.col("season_type") == "REG")
    & (pl.col("def_pos") == "CB")
    & (pl.col("nfl_position") == "WR")
)
pp = pl.read_parquet(P + "pp.parquet").select(
    ["game_id", "play_id", "nfl_id", "lined_up_position", "pass_result", "pass_length"]
)
d = m.join(pp, on=["game_id", "play_id", "nfl_id"], how="left").to_pandas()
d["sep"] = d["separation_at_pass_forward"]
d["man"] = (d.coverage_assignment == "MAN").astype(int)
d["targeted"] = (d.target == "TRUE").astype(int)
d["slot"] = (d.lined_up_position == "SLOT_WR").astype(int)
d["one"] = 1
vc = {"receiver": "0 + C(nfl_id)", "defender": "0 + C(def_id)"}


def split(data, formula, label, outcome="sep"):
    fit = smf.mixedlm(formula, data, groups="one", vc_formula=vc).fit(reml=True)
    var = dict(zip(fit.model.exog_vc.names, fit.vcomp))
    tot = sum(var.values()) + fit.scale
    logging.info(
        "%s",
        f"{label}: reps = {len(data)}, receivers = {data.nfl_id.nunique()}, corners = {data.def_id.nunique()} | receiver {100 * var['receiver'] / tot:.1f}%, corner {100 * var['defender'] / tot:.1f}%, residual {100 * fit.scale / tot:.1f}%",
    )


split(
    d,
    "sep ~ C(route_ran) + man + targeted + slot",
    "all confirmed WR-CB reps (+ slot alignment)",
)
split(d[d.man == 1], "sep ~ C(route_ran) + targeted + slot", "MAN assignment only")
split(d[d.man == 0], "sep ~ C(route_ran) + targeted + slot", "zone / other assignments")
rng = np.random.default_rng(0)
dm = d[d.man == 1].copy()
shares = []
for _ in range(20):
    dm["def_id_perm"] = rng.permutation(dm["def_id"].values)
    fit = smf.mixedlm(
        "sep ~ C(route_ran) + targeted + slot",
        dm,
        groups="one",
        vc_formula={"receiver": "0 + C(nfl_id)", "defender": "0 + C(def_id_perm)"},
    ).fit(reml=True)
    var = dict(zip(fit.model.exog_vc.names, fit.vcomp))
    shares.append(100 * var["defender"] / (sum(var.values()) + fit.scale))
logging.info(
    "%s",
    f"  MAN reps, corner labels shuffled 20 times: corner share mean {np.mean(shares):.2f}%, max {np.max(shares):.2f}%",
)
t = d[(d.targeted == 1) & d.pass_result.isin(["C", "I", "IN"])].copy()
t["caught"] = (t.pass_result == "C").astype(float)
logging.info(
    "%s %s %s %s",
    "targeted confirmed WR-CB reps with a catch outcome:",
    len(t),
    "| completion rate:",
    round(t.caught.mean(), 3),
)
split(
    t,
    "caught ~ sep + pass_length + man",
    "catch outcome on targets (linear probability)",
)
