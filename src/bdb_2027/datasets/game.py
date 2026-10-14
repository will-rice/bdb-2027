"""Game performance and player-play outcome extraction."""

from pathlib import Path

import polars as pl

from bdb_2027.config import Config


class GameOutcomeExtractor:
    """Aggregate on-field regular season performance metrics per player."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def load_player_play(self, path: Path | None = None) -> pl.DataFrame:
        """Load granular player-play Next Gen Stats."""
        file_path = path or self.config.paths.player_play_file
        if not file_path.exists():
            raise FileNotFoundError(f"Player play data not found at {file_path}")
        return pl.read_csv(
            file_path,
            null_values=["NA", "null", ""],
            infer_schema_length=None,
            schema_overrides={
                "sack_allowed": pl.Float64,
                "sack": pl.Float64,
            },
        )

    def load_games(self, path: Path | None = None) -> pl.DataFrame:
        """Load schedule and matchup metadata."""
        file_path = path or self.config.paths.games_file
        if not file_path.exists():
            raise FileNotFoundError(f"Games data not found at {file_path}")
        return pl.read_csv(file_path)

    def extract_receiver_metrics(self, player_play_df: pl.DataFrame) -> pl.DataFrame:
        """Extract route running, breaking separation, and YACOE outcomes for WRs/TEs/RBs."""
        receivers = player_play_df.filter(
            pl.col("lined_up_position").is_in(["WR", "SLOT_WR", "TE", "RB"])
        )

        has_route = "route_ran" in receivers.columns
        breaking_routes = ["SLANT", "OUT", "IN", "COMEBACK", "HITCH", "CROSS"]

        agg_exprs = [
            pl.len().alias("total_snaps"),
            pl.col("target").sum().alias("total_targets"),
            pl.col("separation_at_pass_forward").mean().alias("avg_separation_at_pass"),
            pl.col("separation_at_pass_forward")
            .quantile(0.75)
            .alias("p75_separation_at_pass"),
        ]

        if has_route:
            agg_exprs.extend(
                [
                    pl.col("separation_at_pass_forward")
                    .filter(pl.col("route_ran").is_in(breaking_routes))
                    .mean()
                    .alias("breaking_route_separation"),
                    pl.col("separation_at_pass_forward")
                    .filter(pl.col("route_ran") == "GO")
                    .mean()
                    .alias("go_route_separation"),
                    pl.col("target")
                    .filter(pl.col("route_ran").is_in(breaking_routes))
                    .sum()
                    .alias("breaking_targets_count"),
                ]
            )
        else:
            agg_exprs.extend(
                [
                    pl.lit(None).cast(pl.Float64).alias("breaking_route_separation"),
                    pl.lit(None).cast(pl.Float64).alias("go_route_separation"),
                    pl.lit(0).cast(pl.UInt32).alias("breaking_targets_count"),
                ]
            )

        agg_exprs.extend(
            [
                pl.col("cushion").mean().alias("avg_cushion_faced"),
                pl.col("rec_yards").sum().alias("total_receiving_yards"),
                pl.col("yards_after_catch").mean().alias("avg_yac"),
                (pl.col("yards_after_catch") - pl.col("expected_yards_after_catch"))
                .mean()
                .alias("avg_yac_above_expected"),
            ]
        )

        if "cushion" in receivers.columns:
            agg_exprs.extend(
                [
                    pl.col("target")
                    .filter(pl.col("cushion") <= 3.0)
                    .sum()
                    .alias("press_targets_count"),
                    pl.col("separation_at_pass_forward")
                    .filter(pl.col("cushion") <= 3.0)
                    .mean()
                    .alias("press_separation"),
                ]
            )

        if "in_motion_at_ball_snap" in receivers.columns:
            agg_exprs.extend(
                [
                    pl.col("target")
                    .filter(pl.col("in_motion_at_ball_snap"))
                    .sum()
                    .alias("in_motion_targets_count"),
                    pl.col("separation_at_pass_forward")
                    .filter(pl.col("in_motion_at_ball_snap"))
                    .mean()
                    .alias("in_motion_separation"),
                ]
            )

        if "yardline_number" in receivers.columns:
            agg_exprs.extend(
                [
                    pl.col("target")
                    .filter(pl.col("yardline_number") <= 10)
                    .sum()
                    .alias("inside_10_targets_count"),
                    pl.col("separation_at_pass_forward")
                    .filter(pl.col("yardline_number") <= 10)
                    .mean()
                    .alias("inside_10_separation"),
                ]
            )

        return receivers.group_by("nfl_id").agg(agg_exprs)

    def extract_pass_rusher_metrics(self, player_play_df: pl.DataFrame) -> pl.DataFrame:
        """Extract trench burst and disruption metrics for DL/EDGE/OLB."""
        pass_rushers = player_play_df.filter(
            pl.col("lined_up_position").is_in(["EDGE", "INTERIOR_LINE", "OLB"])
        )

        agg_exprs = [
            pl.len().alias("total_rush_snaps"),
            pl.col("player_get_off").mean().alias("avg_get_off_time"),
            pl.col("player_get_off").min().alias("fastest_get_off_time"),
            pl.col("time_to_pressure").mean().alias("avg_time_to_pressure"),
            pl.col("quick_pressure").sum().alias("quick_pressures_count"),
            pl.col("quick_pressure")
            .cast(pl.Float64)
            .mean()
            .alias("quick_pressure_rate"),
            pl.col("sack").sum().alias("total_sacks"),
            pl.col("tackle_for_loss").sum().alias("total_tfl"),
        ]

        if "unblocked_pressure" in pass_rushers.columns:
            agg_exprs.append(
                pl.col("unblocked_pressure")
                .cast(pl.Float64)
                .sum()
                .alias("unblocked_pressures_count")
            )

        return pass_rushers.group_by("nfl_id").agg(agg_exprs)

    def extract_offensive_line_metrics(
        self, player_play_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract pass protection win rates and pressure allowed for OL."""
        ol = player_play_df.filter(pl.col("lined_up_position").is_in(["T", "G", "C"]))

        agg_exprs = [
            pl.len().alias("total_pass_pro_snaps"),
            pl.col("pressure_allowed")
            .cast(pl.Float64)
            .mean()
            .alias("pressure_rate_allowed"),
            pl.col("sack_allowed").sum().alias("total_sacks_allowed"),
            pl.col("peak_pressure_probability_allowed")
            .mean()
            .alias("avg_peak_pressure_prob_allowed"),
            pl.col("time_to_pressure_allowed")
            .mean()
            .alias("avg_time_to_pressure_allowed"),
        ]

        if "pass_rushers_encountered" in ol.columns:
            agg_exprs.append(
                (pl.col("pass_rushers_encountered") >= 2)
                .cast(pl.Float64)
                .sum()
                .alias("stunts_encountered_count")
            )

        if "extended_sack_allowed" in ol.columns:
            agg_exprs.append(
                pl.col("extended_sack_allowed")
                .cast(pl.Float64)
                .sum()
                .alias("extended_sacks_allowed")
            )

        if "dropback_duration" in ol.columns:
            agg_exprs.extend(
                [
                    (pl.col("dropback_duration") > 3.2)
                    .cast(pl.Float64)
                    .sum()
                    .alias("extended_dropbacks_count"),
                    pl.col("pressure_allowed")
                    .filter(pl.col("dropback_duration") > 3.2)
                    .cast(pl.Float64)
                    .mean()
                    .alias("extended_pressure_rate_allowed"),
                ]
            )

        return ol.group_by("nfl_id").agg(agg_exprs)

    def extract_defensive_back_metrics(
        self, player_play_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract coverage impact and pass completion allowed for DBs/CBs/Safeties."""
        dbs = player_play_df.filter(
            pl.col("lined_up_position").is_in(
                ["CB", "SLOT_CB", "HIGH_SAFETY", "BOX_SAFETY", "DB"]
            )
            & pl.col("pass_result").is_not_null()
            & ~pl.col("blitzing")
        )

        agg_exprs = [
            pl.len().alias("total_coverage_snaps"),
            pl.col("expected_points_added")
            .cast(pl.Float64)
            .mean()
            .alias("epa_per_snap_allowed"),
            (pl.col("pass_result") == "C")
            .cast(pl.Float64)
            .mean()
            .alias("completion_rate_allowed"),
            (pl.col("coverage_assignment") == "MAN")
            .cast(pl.Float64)
            .mean()
            .alias("man_coverage_rate"),
        ]

        if "team_coverage_man_zone" in dbs.columns:
            agg_exprs.extend(
                [
                    (pl.col("team_coverage_man_zone") == "MAN_COVERAGE")
                    .cast(pl.Float64)
                    .sum()
                    .alias("man_coverage_snaps"),
                    (pl.col("pass_result") == "C")
                    .filter(pl.col("team_coverage_man_zone") == "MAN_COVERAGE")
                    .cast(pl.Float64)
                    .mean()
                    .alias("man_completion_rate_allowed"),
                    pl.col("pass_result")
                    .filter(pl.col("team_coverage_man_zone") == "MAN_COVERAGE")
                    .is_in(["I", "IN"])
                    .cast(pl.Float64)
                    .sum()
                    .alias("man_forced_incompletions"),
                ]
            )

        if "cushion" in dbs.columns:
            agg_exprs.extend(
                [
                    pl.col("cushion").mean().alias("avg_cushion_allowed"),
                    (pl.col("cushion") <= 3.0)
                    .cast(pl.Float64)
                    .sum()
                    .alias("press_coverage_snaps"),
                ]
            )

        return dbs.group_by("nfl_id").agg(agg_exprs)
