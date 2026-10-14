"""Combine data loading and sensor tracking feature engineering."""

from pathlib import Path

import polars as pl

from bdb_2027.config import Config
from bdb_2027.datasets.kinematics import compute_heading_transitions

# The W-drill chains three sharp hip reversals (backpedal -> drive -> backpedal ->
# drive) and is recorded end-to-end in every draft class. The other back-pedal drills
# are unusable for latency: the 45-degree reaction reps stop recording before the
# break in 2024, and the 90-degree break is a rounded speed turn with no plant.
DB_TRANSITION_DRILL = "W_DRILL"
# Reversals per rep. Anything earlier in the recording is pre-rep movement.
W_DRILL_TRANSITIONS = 3


class CombineFeatureExtractor:
    """Extract kinematic and mechanical features from NFL Combine sensor tracking."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def load_combine_results(self, path: Path | None = None) -> pl.DataFrame:
        """Load standardized Combine physical testing results."""
        file_path = path or self.config.paths.combine_results_file
        if not file_path.exists():
            raise FileNotFoundError(f"Combine results not found at {file_path}")
        return pl.read_csv(
            file_path,
            null_values=["NA", "null", ""],
            infer_schema_length=None,
        )

    def load_combine_tracking(self, path: Path | None = None) -> pl.DataFrame:
        """Load 10 Hz high-frequency Combine sensor tracking frames."""
        file_path = path or self.config.paths.combine_tracking_file
        if not file_path.exists():
            raise FileNotFoundError(f"Combine tracking not found at {file_path}")
        return pl.read_csv(
            file_path,
            null_values=["NA", "null", ""],
            infer_schema_length=10000,
        )

    def extract_drill_features(self, tracking_df: pl.DataFrame) -> pl.DataFrame:
        """Extract kinematic features across drill attempts.

        Calculates:
        - Peak speed (s_max)
        - Peak acceleration (a_max)
        - Peak deceleration (a_min / braking force)
        - Average speed (s_mean)
        - Duration / total distance
        - Speed maintenance (speed in second half of drill / peak speed)
        - Rate of acceleration change (jerk approximation)
        """
        # Filter to player frames only (avoid ball tracking frames)
        player_frames = tracking_df.filter(pl.col("entity_type") == "PLAYER")

        # Calculate tangential acceleration per rep by differentiating speed
        partition_cols = (
            ["nfl_id", "event_id"]
            if "event_id" in player_frames.columns
            else ["nfl_id"]
        )
        player_frames = player_frames.with_columns(
            [
                (
                    (pl.col("s") - pl.col("s").shift(1))
                    * self.config.features.tracking_hz
                )
                .over(partition_cols)
                .fill_null(0.0)
                .alias("tangential_accel")
            ]
        )

        # Aggregate drill-level kinematic metrics
        drill_features = (
            player_frames.group_by(["nfl_id", "draft_year", "drill_type", "drill_name"])
            .agg(
                [
                    pl.col("s").max().alias("peak_speed"),
                    pl.col("s").mean().alias("mean_speed"),
                    pl.col("s").quantile(0.90).alias("speed_p90"),
                    pl.col("a").max().alias("peak_accel"),
                    # True horizontal braking deceleration: maximum magnitude of negative tangential acceleration
                    pl.col("tangential_accel")
                    .filter(pl.col("tangential_accel") < 0)
                    .abs()
                    .max()
                    .fill_null(0.0)
                    .alias("peak_decel"),
                    pl.col("a").std().alias("accel_std"),
                    pl.col("dis").sum().alias("total_distance"),
                    pl.len().alias("frame_count"),
                ]
            )
            .with_columns(
                [
                    (pl.col("frame_count") / self.config.features.tracking_hz).alias(
                        "drill_duration_sec"
                    )
                ]
            )
        )

        # Centering by (drill_type, draft_year) to prevent cross-year sensor artifacts
        if self.config.features.center_by_draft_year:
            drill_features = drill_features.with_columns(
                [
                    (
                        pl.col("peak_speed")
                        - pl.col("peak_speed").mean().over(["drill_type", "draft_year"])
                    ).alias("peak_speed_centered"),
                    (
                        pl.col("peak_accel")
                        - pl.col("peak_accel").mean().over(["drill_type", "draft_year"])
                    ).alias("peak_accel_centered"),
                    (
                        pl.col("peak_decel")
                        - pl.col("peak_decel").mean().over(["drill_type", "draft_year"])
                    ).alias("peak_decel_centered"),
                ]
            )

        return drill_features

    def extract_40yd_features(self, tracking_df: pl.DataFrame) -> pl.DataFrame:
        """Extract granular sprint mechanics for the 40-yard dash.

        Analyzes the drive phase (0-10y), transition phase (10-20y),
        and maximum velocity phase (20-40y).
        """
        sprint_frames = tracking_df.filter(
            (pl.col("entity_type") == "PLAYER")
            & (pl.col("drill_type") == "FORTY_YARD_DASH")
        )

        if sprint_frames.is_empty():
            return pl.DataFrame()

        # Track acceleration profiles per attempt
        sprint_agg = (
            sprint_frames.group_by(["nfl_id", "draft_year", "attempt"])
            .agg(
                [
                    pl.col("s").max().alias("forty_peak_speed"),
                    pl.col("a").max().alias("forty_peak_accel"),
                    pl.col("s").mean().alias("forty_avg_speed"),
                    pl.len().alias("forty_frames"),
                ]
            )
            .group_by(["nfl_id", "draft_year"])
            .agg(
                [
                    pl.col("forty_peak_speed").max().alias("best_40_peak_speed"),
                    pl.col("forty_peak_accel").max().alias("best_40_peak_accel"),
                    pl.col("forty_avg_speed").max().alias("best_40_avg_speed"),
                ]
            )
        )

        return sprint_agg

    def extract_wr_route_kinematics(self, tracking_df: pl.DataFrame) -> pl.DataFrame:
        """Extract route cutting deceleration, exit burst, and Gauntlet efficiency for WRs."""
        wr_frames = tracking_df.filter(
            (pl.col("entity_type") == "PLAYER")
            & (pl.col("drill_type") == "SKILL_DRILLS_WR")
        )
        if wr_frames.is_empty():
            return pl.DataFrame()

        # Tangential acceleration per rep
        wr_frames = wr_frames.with_columns(
            [
                (
                    (pl.col("s") - pl.col("s").shift(1))
                    * self.config.features.tracking_hz
                )
                .over(["nfl_id", "event_id"])
                .fill_null(0.0)
                .alias("tangential_accel")
            ]
        )

        breaking_drills = [
            "COMEBACK_ROUTE_RIGHT",
            "CURL_ROUTE_RIGHT",
            "SLANT_ROUTE_LEFT",
            "SPEED_OUT_LEFT",
            "SPEED_OUT_ROUTE_LEFT",
            "POST_CORNER_ROUTE_LEFT",
        ]

        route_reps = (
            wr_frames.filter(pl.col("drill_name").is_in(breaking_drills))
            .group_by(["nfl_id", "event_id", "draft_year", "drill_name"])
            .agg(
                [
                    pl.col("s").max().alias("rep_peak_speed"),
                    pl.col("tangential_accel")
                    .filter(pl.col("tangential_accel") < 0)
                    .abs()
                    .max()
                    .fill_null(0.0)
                    .alias("rep_braking_decel"),
                    pl.col("tangential_accel")
                    .filter(pl.col("tangential_accel") > 0)
                    .max()
                    .fill_null(0.0)
                    .alias("rep_exit_accel"),
                ]
            )
        )

        # Standardize within (drill_name, draft_year)
        route_norm = route_reps.with_columns(
            [
                (
                    (
                        pl.col("rep_peak_speed")
                        - pl.col("rep_peak_speed")
                        .mean()
                        .over(["drill_name", "draft_year"])
                    )
                    / (
                        pl.col("rep_peak_speed")
                        .std()
                        .over(["drill_name", "draft_year"])
                        + 1e-6
                    )
                ).alias("wr_route_speed_z"),
                (
                    (
                        pl.col("rep_braking_decel")
                        - pl.col("rep_braking_decel")
                        .mean()
                        .over(["drill_name", "draft_year"])
                    )
                    / (
                        pl.col("rep_braking_decel")
                        .std()
                        .over(["drill_name", "draft_year"])
                        + 1e-6
                    )
                ).alias("wr_route_decel_z"),
            ]
        )

        wr_route_agg = route_norm.group_by("nfl_id").agg(
            [
                pl.col("wr_route_speed_z").mean(),
                pl.col("wr_route_decel_z").mean(),
                pl.col("rep_braking_decel").mean().alias("raw_wr_braking_decel"),
            ]
        )

        # Gauntlet drill extraction
        gauntlet = (
            wr_frames.filter(pl.col("drill_name") == "GAUNTLET_DRILL")
            .filter(pl.col("s") > 1.5)
            .group_by(["nfl_id", "draft_year"])
            .agg(
                [
                    pl.col("s").mean().alias("gauntlet_mean_speed"),
                    pl.col("s").max().alias("gauntlet_peak_speed"),
                    (pl.col("s").std() / (pl.col("s").mean() + 1e-6)).alias(
                        "gauntlet_stutter_cv"
                    ),
                ]
            )
        )

        if not gauntlet.is_empty():
            gauntlet_norm = gauntlet.with_columns(
                [
                    (
                        (
                            pl.col("gauntlet_mean_speed")
                            - pl.col("gauntlet_mean_speed").mean().over("draft_year")
                        )
                        / (
                            pl.col("gauntlet_mean_speed").std().over("draft_year")
                            + 1e-6
                        )
                    ).alias("gauntlet_speed_z"),
                    (
                        (
                            pl.col("gauntlet_stutter_cv")
                            - pl.col("gauntlet_stutter_cv").mean().over("draft_year")
                        )
                        / (
                            pl.col("gauntlet_stutter_cv").std().over("draft_year")
                            + 1e-6
                        )
                    ).alias("gauntlet_stutter_z"),
                ]
            ).select(["nfl_id", "gauntlet_speed_z", "gauntlet_stutter_z"])

            wr_route_agg = wr_route_agg.join(gauntlet_norm, on="nfl_id", how="left")

        return wr_route_agg

    def extract_dl_hoop_kinematics(self, tracking_df: pl.DataFrame) -> pl.DataFrame:
        """Extract Hoop Drill centripetal acceleration and bend efficiency for DL/EDGE."""
        hoop = tracking_df.filter(
            (pl.col("entity_type") == "PLAYER")
            & (pl.col("drill_name") == "RUN_THE_HOOP_DRILL")
        )
        if hoop.is_empty():
            return pl.DataFrame()

        hoop_agg = (
            hoop.group_by(["nfl_id", "draft_year"])
            .agg(
                [
                    pl.col("s").max().alias("hoop_peak_speed"),
                    pl.col("s").mean().alias("hoop_mean_speed"),
                    pl.col("a").max().alias("hoop_peak_accel"),
                ]
            )
            .with_columns(
                [
                    (
                        (
                            pl.col("hoop_peak_speed")
                            - pl.col("hoop_peak_speed").mean().over("draft_year")
                        )
                        / (pl.col("hoop_peak_speed").std().over("draft_year") + 1e-6)
                    ).alias("hoop_speed_z"),
                    (
                        (
                            pl.col("hoop_peak_accel")
                            - pl.col("hoop_peak_accel").mean().over("draft_year")
                        )
                        / (pl.col("hoop_peak_accel").std().over("draft_year") + 1e-6)
                    ).alias("hoop_accel_z"),
                ]
            )
            .select(["nfl_id", "hoop_speed_z", "hoop_accel_z", "hoop_peak_speed"])
        )
        return hoop_agg

    def extract_ol_mirror_kinematics(self, tracking_df: pl.DataFrame) -> pl.DataFrame:
        """Extract lateral agility and mirror recovery acceleration for Offensive Linemen."""
        ol_drills = tracking_df.filter(
            (pl.col("entity_type") == "PLAYER")
            & pl.col("drill_name").str.contains("MIRROR|WAVE")
        )
        if ol_drills.is_empty():
            return pl.DataFrame()

        ol_agg = (
            ol_drills.group_by(["nfl_id", "draft_year"])
            .agg(
                [
                    pl.col("s").max().alias("mirror_max_speed"),
                    pl.col("s").mean().alias("mirror_mean_speed"),
                    pl.col("a").max().alias("mirror_max_accel"),
                ]
            )
            .with_columns(
                [
                    (
                        (
                            pl.col("mirror_max_accel")
                            - pl.col("mirror_max_accel").mean().over("draft_year")
                        )
                        / (pl.col("mirror_max_accel").std().over("draft_year") + 1e-6)
                    ).alias("mirror_accel_z"),
                ]
            )
            .select(
                ["nfl_id", "mirror_max_accel", "mirror_accel_z", "mirror_max_speed"]
            )
        )
        return ol_agg

    def extract_db_hip_transition_kinematics(
        self, tracking_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract hip transition latency and plant-and-drive kinetics for Defensive Backs.

        Times the last `W_DRILL_TRANSITIONS` heading reversals of every W-drill rep.
        Each reversal is standardized against the same reversal (counted back from
        the end of the rep, which is always recorded) in the same draft class, so
        drill geometry and per-year sensor exports cannot leak into the player score.

        Returns:
            One row per player with `db_hip_transition_latency` (mean seconds),
            `db_transition_count`, and the z-scores `db_hip_latency_z`,
            `db_brake_decel_z` and `db_drive_accel_z`.
        """
        rep_keys = ["nfl_id", "event_id", "draft_year"]
        reps = tracking_df.filter(
            (pl.col("entity_type") == "PLAYER")
            & (pl.col("drill_name") == DB_TRANSITION_DRILL)
        ).sort("time")
        if reps.is_empty():
            return pl.DataFrame()

        rows = []
        for key, rep in reps.partition_by(rep_keys, as_dict=True).items():
            turns = compute_heading_transitions(
                rep["s"].to_numpy(),
                rep["dir"].to_numpy(),
                self.config.features.tracking_hz,
            )
            for turn_from_end, turn in enumerate(
                reversed(turns[-W_DRILL_TRANSITIONS:])
            ):
                rows.append(
                    dict(zip(rep_keys, key, strict=True))
                    | {"turn_from_end": turn_from_end}
                    | turn
                )

        cohort = ["draft_year", "turn_from_end"]
        z_scores = {
            "hip_transition_latency": "db_hip_latency_z",
            "brake_decel": "db_brake_decel_z",
            "drive_accel": "db_drive_accel_z",
        }
        transitions = pl.DataFrame(rows).with_columns(
            [
                (
                    (pl.col(raw) - pl.col(raw).mean().over(cohort))
                    / (pl.col(raw).std().over(cohort) + 1e-6)
                ).alias(z)
                for raw, z in z_scores.items()
            ]
        )

        return transitions.group_by("nfl_id").agg(
            [
                pl.col("hip_transition_latency")
                .mean()
                .alias("db_hip_transition_latency"),
                pl.len().alias("db_transition_count"),
                *[pl.col(z).mean() for z in z_scores.values()],
            ]
        )
