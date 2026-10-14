"""Probe: in confirmed rookie-vs-rookie coverage reps, do Combine differentials predict separation?"""

import logging
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = "data/processed/direction_probes/"
m = pl.read_parquet(P + "matched_at_throw.parquet")
res = pl.read_csv("data/raw/combine_results.csv", null_values=["NA"])
ct = pl.scan_csv("data/raw/combine_tracking.csv").filter(
    pl.col("entity_type") == "PLAYER"
)
top = (
    ct.filter(pl.col("drill_type") == "FORTY_YARD_DASH")
    .group_by("nfl_id")
    .agg(pl.col("s").max().alias("top_speed"))
    .collect()
)
feats = res.select(
    [
        "nfl_id",
        "forty",
        "ten_yd_split",
        "combine_weight",
        "combine_height",
        "arm_length",
        "three_cone",
        "short_shuttle",
    ]
).join(top, on="nfl_id", how="left")
logging.info(
    "%s %s %s %s",
    "players with tracked 40 top speed:",
    top.height,
    "| official forty non-null:",
    res["forty"].is_not_null().sum(),
)
d = m.join(
    feats.rename({c: "r_" + c for c in feats.columns if c != "nfl_id"}),
    on="nfl_id",
    how="left",
).join(
    feats.rename({c: "d_" + c for c in feats.columns if c != "nfl_id"}).rename(
        {"nfl_id": "def_id"}
    ),
    on="def_id",
    how="left",
)
d = d.filter(pl.col("season_type") == "REG")
logging.info("%s %s", "regular-season confirmed reps:", d.height)
logging.info(
    "%s",
    d.group_by("route_ran")
    .agg(
        pl.len().alias("reps"),
        pl.col("separation_at_pass_forward").mean().round(2).alias("mean_sep"),
    )
    .sort("reps", descending=True)
    .head(14),
)
df = d.to_pandas()
df["sep"] = df["separation_at_pass_forward"]
df["speed_edge"] = df["r_top_speed"] - df["d_top_speed"]
df["forty_edge"] = df["d_forty"] - df["r_forty"]
df["weight_edge"] = df["r_combine_weight"] - df["d_combine_weight"]
df["height_edge"] = df["r_combine_height"] - df["d_combine_height"]
df["man"] = (df["coverage_assignment"] == "MAN").astype(int)
df["targeted"] = (df["target"] == "TRUE").astype(int)
vertical = {"GO", "POST", "CORNER", "SEAM", "FADE", "DEEP_CROSS", "WHEEL"}
df["vertical"] = df["route_ran"].isin(vertical).astype(int)


def fit(formula, data, label):
    data = data.dropna(
        subset=[
            c
            for c in [
                "sep",
                "speed_edge",
                "forty_edge",
                "weight_edge",
                "r_top_speed",
                "d_top_speed",
            ]
            if c in formula
        ]
    ).copy()
    groups = np.column_stack(
        [
            data["nfl_id"].astype("category").cat.codes,
            data["def_id"].astype("category").cat.codes,
        ]
    )
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    logging.info(
        "%s",
        f"\n--- {label}: n reps = {int(model.nobs)}, receivers = {data['nfl_id'].nunique()}, defenders = {data['def_id'].nunique()}, R2 = {model.rsquared:.3f}",
    )
    for name in model.params.index:
        if name.startswith("C(") or name == "Intercept":
            continue
        logging.info(
            "%s",
            f"   {name:28s} coef {model.params[name]:+.3f}  se {model.bse[name]:.3f}  p {model.pvalues[name]:.4f}",
        )


ctx = "C(route_ran) + man + targeted"
fit(
    f"sep ~ speed_edge + {ctx}",
    df,
    "all reps: tracked top-speed edge (yd/s), two-way clustered",
)
fit(
    f"sep ~ r_top_speed + d_top_speed + {ctx}",
    df,
    "all reps: each side's top speed separately",
)
fit(f"sep ~ speed_edge + weight_edge + {ctx}", df, "all reps: speed edge + weight edge")
fit("sep ~ speed_edge * vertical + man + targeted", df, "speed edge x vertical route")
fit(
    "sep ~ speed_edge + C(route_ran) + targeted",
    df[df["man"] == 1],
    "MAN-assignment reps only",
)
fit(
    "sep ~ speed_edge + targeted",
    df[(df["man"] == 1) & (df["vertical"] == 1)],
    "MAN reps on vertical routes",
)
fit(f"sep ~ forty_edge + {ctx}", df, "all reps: official 40-time edge (s)")
logging.info(
    "%s %s %s %s %s",
    "\nSD of speed_edge:",
    round(df["speed_edge"].std(), 3),
    "yd/s | SD of sep:",
    round(df["sep"].std(), 2),
    "yd",
)
