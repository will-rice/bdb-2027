"""Probe: a defender-side, tracking-only outcome for DBs: closing on a tracked rookie receiver's catch point during ball flight."""

import logging
import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = "data/processed/direction_probes/"
pp = pl.read_parquet(P + "pp.parquet").filter(pl.col("season_type") == "REG")
gt = pl.scan_parquet(P + "gt_*.parquet")
fr = (
    gt.filter(pl.col("event").is_in(["pass_forward", "pass_arrived"]))
    .select(["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "dir", "event"])
    .collect()
)
thrown = fr.filter(pl.col("event") == "pass_forward").drop("event")
arrived = fr.filter(pl.col("event") == "pass_arrived").drop("event")
keys = ["game_id", "play_id"]
rec = (
    pp.filter((pl.col("target") == "TRUE") & pl.col("on_offense"))
    .select(
        keys
        + [
            pl.col("nfl_id").alias("rec_id"),
            "route_ran",
            "pass_length",
            "pass_result",
            "team_coverage_man_zone",
        ]
    )
    .join(
        arrived.rename({"nfl_id": "rec_id", "x": "cx", "y": "cy", "time": "t1"}).select(
            keys + ["rec_id", "cx", "cy", "t1"]
        ),
        on=keys + ["rec_id"],
    )
)
dbs = (
    pp.filter(
        ~pl.col("on_offense") & pl.col("nfl_position").is_in(["CB", "SS", "FS", "DB"])
    )
    .select(
        keys
        + [
            pl.col("nfl_id").alias("def_id"),
            pl.col("nfl_position").alias("def_pos"),
            "coverage_assignment",
            "season",
        ]
    )
    .join(
        thrown.rename(
            {
                "nfl_id": "def_id",
                "x": "x0",
                "y": "y0",
                "s": "s0",
                "dir": "dir0",
                "time": "t0",
            }
        ),
        on=keys + ["def_id"],
    )
    .join(
        arrived.rename({"nfl_id": "def_id", "x": "x1", "y": "y1"}).select(
            keys + ["def_id", "x1", "y1"]
        ),
        on=keys + ["def_id"],
    )
)
d = dbs.join(rec, on=keys)
d = d.with_columns(
    ((pl.col("t1") - pl.col("t0")).dt.total_milliseconds() / 1000).alias("air"),
    ((pl.col("x0") - pl.col("cx")) ** 2 + (pl.col("y0") - pl.col("cy")) ** 2)
    .sqrt()
    .alias("d0"),
    ((pl.col("x1") - pl.col("cx")) ** 2 + (pl.col("y1") - pl.col("cy")) ** 2)
    .sqrt()
    .alias("d1"),
)
d = d.filter(
    (pl.col("air") >= 0.5)
    & (pl.col("air") <= 4.0)
    & (pl.col("d0") >= 2.0)
    & (pl.col("d0") <= 20.0)
)
bearing = pl.arctan2(pl.col("cx") - pl.col("x0"), pl.col("cy") - pl.col("y0"))
d = d.with_columns(
    ((pl.col("d0") - pl.col("d1")) / pl.col("air")).alias("closing"),
    (pl.col("s0") * (pl.col("dir0").radians() - bearing).cos()).alias("v_toward"),
)
logging.info(
    "%s %s %s %s %s %s",
    "DB reps with a tracked rookie receiver targeted on the play (2-20 yd away at the throw):",
    d.height,
    "| defenders:",
    d["def_id"].n_unique(),
    "| receivers:",
    d["rec_id"].n_unique(),
)
logging.info(
    "%s",
    d.group_by("def_pos").agg(
        pl.len().alias("reps"),
        pl.col("def_id").n_unique().alias("players"),
        pl.col("closing").mean().round(2),
        pl.col("d0").mean().round(1),
    ),
)
df = d.to_pandas()
X = df[["d0", "air", "s0", "v_toward", "pass_length"]].fillna(0)
model = HistGradientBoostingRegressor(
    max_depth=4, learning_rate=0.05, max_iter=300, random_state=0
)
df["expected"] = cross_val_predict(
    model, X, df["closing"], groups=df["def_id"], cv=GroupKFold(5)
)
df["coe"] = df["closing"] - df["expected"]
logging.info(
    "%s",
    f"context model (defender-grouped CV) R2 for closing speed: {1 - ((df.closing - df.expected) ** 2).sum() / ((df.closing - df.closing.mean()) ** 2).sum():.3f}",
)
rng = np.random.default_rng(0)


def split_half(frame, value, min_reps):
    rs, n = ([], 0)
    for _ in range(300):
        h = rng.integers(0, 2, len(frame))
        a = (
            frame.assign(h=h)
            .groupby(["def_id", "h"])[value]
            .agg(["mean", "size"])
            .unstack()
        )
        a = a.dropna()
        a = a[(a["size", 0] >= min_reps / 2) & (a["size", 1] >= min_reps / 2)]
        if len(a) > 8:
            rs.append(np.corrcoef(a["mean", 0], a["mean", 1])[0, 1])
            n = len(a)
    r = float(np.mean(rs))
    return (r, 2 * r / (1 + r), n)


for value in ("closing", "coe"):
    for min_reps in (20, 40):
        r, sb, n = split_half(df, value, min_reps)
        logging.info(
            "%s",
            f"  split-half reliability of player mean {value:8s} (>= {min_reps} reps): r = {r:.2f}, Spearman-Brown = {sb:.2f}, defenders = {n}",
        )
per = (
    df.groupby("def_id")
    .agg(
        reps=("coe", "size"),
        coe=("coe", "mean"),
        closing=("closing", "mean"),
        pos=("def_pos", "first"),
    )
    .reset_index()
)
logging.info(
    "%s %s %s %s %s %s",
    "reps per defender: median",
    per.reps.median(),
    "| >=20:",
    (per.reps >= 20).sum(),
    "| >=40:",
    (per.reps >= 40).sum(),
)
res = pl.read_csv("data/raw/combine_results.csv", null_values=["NA"]).select(
    ["nfl_id", "forty", "combine_weight", "draft_year"]
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
drills = (
    ct.filter(pl.col("drill_type") == "SKILL_DRILLS_DB")
    .select(["nfl_id", "event_id", "draft_year", "drill_name", "time", "s"])
    .collect()
    .sort(["event_id", "time"])
)
drills = drills.with_columns(
    pl.col("s").rolling_mean(3, center=True).over("event_id").alias("ss")
)
drills = drills.with_columns(
    (
        (
            pl.col("ss").shift(-1).over("event_id")
            - pl.col("ss").shift(1).over("event_id")
        )
        / 0.2
    ).alias("acc")
)
rep = drills.group_by(["nfl_id", "event_id", "draft_year", "drill_name"]).agg(
    pl.col("s").max().alias("peak_speed"),
    pl.col("acc").max().alias("peak_accel"),
    pl.col("acc").min().alias("peak_decel"),
)
rep = rep.filter(
    ~(
        (pl.col("drill_name") == "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")
        & (pl.col("draft_year") == 2024)
    )
)
for c in ("peak_speed", "peak_accel", "peak_decel"):
    rep = rep.with_columns(
        (
            (pl.col(c) - pl.col(c).mean().over(["drill_name", "draft_year"]))
            / pl.col(c).std().over(["drill_name", "draft_year"])
        ).alias(c + "_z")
    )
drill_player = rep.group_by("nfl_id").agg(
    pl.col("peak_speed_z").mean().alias("drill_speed"),
    pl.col("peak_accel_z").mean().alias("drill_accel"),
    (-pl.col("peak_decel_z")).mean().alias("drill_brake"),
    pl.len().alias("drill_reps"),
)
feat = (
    res.join(top, on="nfl_id", how="left")
    .join(drill_player, on="nfl_id", how="left")
    .to_pandas()
    .rename(columns={"nfl_id": "def_id"})
)
j = per[per.reps >= 20].merge(feat, on="def_id")
logging.info(
    "%s",
    f"\nCombine links, defenders with >= 20 reps (n = {len(j)}); partial = controlling weight and safety/corner",
)
import statsmodels.formula.api as smf

j["is_cb"] = (j.pos == "CB").astype(int)
for outcome in ("coe", "closing"):
    for f in ("top_speed", "forty", "drill_speed", "drill_accel", "drill_brake"):
        s = j.dropna(subset=[f, "combine_weight"])
        r, p = stats.pearsonr(s[f], s[outcome])
        fit = smf.ols(f"{outcome} ~ {f} + combine_weight + is_cb", s).fit()
        logging.info(
            "%s",
            f"  {outcome:8s} ~ {f:12s}: r = {r:+.2f} (p = {p:.3f}, n = {len(s)}) | partial t = {fit.tvalues[f]:+.2f}, p = {fit.pvalues[f]:.3f}",
        )
