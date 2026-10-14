"""Unit tests for NFL Big Data Bowl 2027 pipeline."""

import numpy as np
import polars as pl
import pytest

from bdb_2027.config import Config
from bdb_2027.datasets.combine import CombineFeatureExtractor
from bdb_2027.datasets.game import GameOutcomeExtractor
from bdb_2027.datasets.kinematics import (
    compute_ball_tracking_stability,
    compute_break_on_ball_efficiency,
    compute_career_longevity_residuals,
    compute_centripetal_mechanics,
    compute_compressed_2x1_run_key_inertia_metrics,
    compute_contested_catch_separation_cliff,
    compute_cornerback_boundary_slot_leverage,
    compute_cover0_allout_blitz_cb_leverage_metrics,
    compute_cover0_blitz_temporal_compression,
    compute_cover6_split_field_turnover_trap_metrics,
    compute_cover_2_man_speed_rush_metrics,
    compute_coverage_rotation_seam_advantage,
    compute_day3_special_teams_survival_index,
    compute_deep_ball_size_leverage_index,
    compute_deep_pass_coverage_shell_risk_metrics,
    compute_deep_perimeter_cb_stature_leverage_metrics,
    compute_defensive_back_blitz_timing_metrics,
    compute_defensive_back_cushion_elasticity,
    compute_defensive_tackle_penetration_archetype,
    compute_disguised_coverage_rotation_metrics,
    compute_disruption_hierarchy_penalty,
    compute_down_and_distance_pressure_escalation,
    compute_dt_scramble_containment_burst_metrics,
    compute_edge_disruption_index,
    compute_edge_pressure_latency_chain,
    compute_empty_protection_breakdown_hazard,
    compute_flat_coverage_index,
    compute_formation_getoff_spectrum_dynamics,
    compute_formation_spatial_leverage_advantage,
    compute_fourth_quarter_ol_fatigue_decay_metrics,
    compute_fourth_quarter_pass_rush_acceleration_metrics,
    compute_gauntlet_yac_translation,
    compute_goal_line_formation_penalty_metrics,
    compute_goalline_fade_cliff_dynamics,
    compute_heading_transitions,
    compute_heavy_formation_te_efficiency_metrics,
    compute_in_vs_out_breaking_route_dynamics,
    compute_inside_10_target_efficiency,
    compute_interior_defensive_line_burst_penetration_metrics,
    compute_interior_dt_penetration_index,
    compute_interior_impulse_index,
    compute_interior_offensive_line_leverage_metrics,
    compute_interior_ol_lever_paradox_metrics,
    compute_man_coverage_reach_advantage,
    compute_money_down_distance_regime_metrics,
    compute_motion_cross_vector_amplification,
    compute_motion_vector_advantage,
    compute_overload_dual_rusher_breakdown_hazard,
    compute_overload_protection_metrics,
    compute_pass_length_depth_spectrum_metrics,
    compute_pass_protection_longevity_index,
    compute_pass_rush_quick_pressure_index,
    compute_pocket_halflife_degradation,
    compute_pocket_lifespan_efficiency_curve_metrics,
    compute_presnap_motion_advantage,
    compute_presnap_motion_coverage_rupture_metrics,
    compute_presnap_motion_vector_surplus,
    compute_press_disengagement_metrics,
    compute_press_release_polar_reversal,
    compute_receiver_alignment_structural_density_metrics,
    compute_red_zone_compression_metrics,
    compute_red_zone_spatial_compression_metrics,
    compute_route_curvature_efficiency,
    compute_route_stem_kinematic_smoothness,
    compute_run_and_club_efficiency,
    compute_rusher_getoff_trench_rupture,
    compute_safety_spin_down_coverage_risk_metrics,
    compute_scramble_drill_breakdown_metrics,
    compute_shuttle_bilateral_kinematics,
    compute_simulated_pressure_edge_coverage_metrics,
    compute_single_high_safety_range_metrics,
    compute_situational_datr,
    compute_slot_alignment_separation_surplus,
    compute_slot_blitz_unblocked_pursuit_metrics,
    compute_slot_vs_boundary_cb_anthropometric_divergence_metrics,
    compute_slot_vs_boundary_cb_risk,
    compute_sprint_mechanics,
    compute_stunt_overload_protection_hazard,
    compute_stunt_twist_disruption_metrics,
    compute_tackle_extended_lever_penalty,
    compute_tackle_pocket_protection_decay,
    compute_third_down_separation_dynamics,
    compute_tight_end_dual_role_index,
    compute_tight_end_inline_momentum_cliff_metrics,
    compute_tight_end_inline_vs_detached_dynamics,
    compute_trailing_game_script_rush_acceleration_metrics,
    compute_trench_momentum,
    compute_trips_3x1_alignment_asymmetry_metrics,
    compute_two_high_run_defense_concession_metrics,
    compute_unblocked_rusher_timing_rupture_metrics,
    compute_under_center_play_action_decoupling_metrics,
    compute_unified_translation_composite,
    compute_weight_inertia_separation_impact,
    compute_wide_receiver_boundary_inertia_metrics,
    compute_wide_receiver_press_suffocation_metrics,
    compute_yacoe_momentum_index,
    compute_zone_break_efficiency,
    compute_zone_coverage_efficiency,
)
from bdb_2027.modeling.pipeline import TranslationPipeline


def test_config_initialization() -> None:
    """Test Config defaults and path construction."""
    config = Config()
    assert config.competition_slug == "nfl-big-data-bowl-2027"
    assert config.features.tracking_hz == 10.0
    assert config.cv_strategy == "group_kfold"
    assert config.group_col == "nfl_id"
    assert config.n_splits == 5


def test_combine_feature_extractor() -> None:
    """Test CombineFeatureExtractor kinematic computations with synthetic data."""
    config = Config()
    extractor = CombineFeatureExtractor(config)

    # Synthetic tracking frames
    tracking_data = pl.DataFrame(
        {
            "draft_year": [2024, 2024, 2024, 2024],
            "nfl_id": [101, 101, 101, 101],
            "entity_type": ["PLAYER", "PLAYER", "PLAYER", "BALL"],
            "drill_type": [
                "FORTY_YARD_DASH",
                "FORTY_YARD_DASH",
                "FORTY_YARD_DASH",
                "FORTY_YARD_DASH",
            ],
            "drill_name": ["40_YARD", "40_YARD", "40_YARD", "40_YARD"],
            "attempt": [1, 1, 1, 1],
            "x": [0.0, 1.0, 2.5, 3.0],
            "y": [25.0, 25.0, 25.0, 25.0],
            "s": [2.0, 5.0, 8.0, 7.0],
            "a": [1.5, 3.0, 2.0, 0.0],
            "dis": [0.2, 0.5, 0.8, 0.7],
        }
    )

    drill_features = extractor.extract_drill_features(tracking_data)
    assert drill_features.height == 1
    assert "peak_speed" in drill_features.columns
    assert "peak_accel" in drill_features.columns
    assert drill_features["peak_speed"][0] == 8.0
    assert drill_features["peak_accel"][0] == 3.0


def test_game_outcome_extractor() -> None:
    """Test GameOutcomeExtractor receiver outcome computations."""
    config = Config()
    extractor = GameOutcomeExtractor(config)

    player_play_data = pl.DataFrame(
        {
            "game_id": [2024010100, 2024010100],
            "play_id": [1, 2],
            "nfl_id": [101, 101],
            "lined_up_position": ["WR", "WR"],
            "target": [True, True],
            "separation_at_pass_forward": [3.2, 2.8],
            "cushion": [5.5, 6.0],
            "rec_yards": [15.0, 20.0],
            "yards_after_catch": [4.0, 6.0],
            "expected_yards_after_catch": [3.0, 4.0],
        }
    )

    receiver_metrics = extractor.extract_receiver_metrics(player_play_data)
    assert receiver_metrics.height == 1
    assert receiver_metrics["total_targets"][0] == 2
    assert pytest.approx(receiver_metrics["avg_separation_at_pass"][0], 0.01) == 3.0


def test_translation_pipeline_cv() -> None:
    """Test TranslationPipeline cross-validation with GroupKFold."""
    config = Config(n_splits=2, lgb_n_estimators=10)
    pipeline = TranslationPipeline(config)

    # Synthetic dataset with 6 distinct players across 20 observations
    synthetic_df = pl.DataFrame(
        {
            "nfl_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "combine_speed": [
                7.5,
                7.6,
                8.1,
                8.0,
                6.9,
                7.1,
                8.5,
                8.4,
                7.8,
                7.9,
                8.2,
                8.3,
            ],
            "combine_accel": [
                2.5,
                2.6,
                3.1,
                3.0,
                2.1,
                2.2,
                3.5,
                3.4,
                2.8,
                2.9,
                3.2,
                3.3,
            ],
            "nfl_separation": [
                2.8,
                2.9,
                3.4,
                3.3,
                2.2,
                2.3,
                3.8,
                3.7,
                3.0,
                3.1,
                3.5,
                3.6,
            ],
        }
    )

    results = pipeline.train_cv(
        features_df=synthetic_df,
        feature_cols=["combine_speed", "combine_accel"],
        target_col="nfl_separation",
        group_col="nfl_id",
    )

    assert "overall_r2" in results
    assert "pearson_r" in results
    assert "feature_importance" in results
    assert len(results["fold_metrics"]) == 2


def test_wr_route_kinematics() -> None:
    """Test WR route kinematic feature extraction and normalization."""
    config = Config()
    extractor = CombineFeatureExtractor(config)

    # Synthetic tracking for two WR route reps
    synthetic_wr = pl.DataFrame(
        {
            "draft_year": [2024] * 6,
            "event_id": ["E1", "E1", "E1", "E2", "E2", "E2"],
            "nfl_id": [201, 201, 201, 202, 202, 202],
            "entity_type": ["PLAYER"] * 6,
            "drill_type": ["SKILL_DRILLS_WR"] * 6,
            "drill_name": ["SLANT_ROUTE_LEFT"] * 6,
            "attempt": [1] * 6,
            "x": [0.0, 2.0, 4.0, 0.0, 2.0, 4.0],
            "y": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "s": [4.0, 7.0, 3.0, 3.5, 6.5, 2.5],
            "a": [2.0, 1.0, 4.0, 2.0, 1.0, 4.0],
            "dis": [0.4, 0.7, 0.3, 0.35, 0.65, 0.25],
        }
    )

    wr_feats = extractor.extract_wr_route_kinematics(synthetic_wr)
    assert wr_feats.height == 2
    assert "wr_route_speed_z" in wr_feats.columns
    assert "wr_route_decel_z" in wr_feats.columns
    assert "raw_wr_braking_decel" in wr_feats.columns


def test_dl_hoop_kinematics() -> None:
    """Test DL hoop drill kinematics extraction."""
    config = Config()
    extractor = CombineFeatureExtractor(config)

    synthetic_dl = pl.DataFrame(
        {
            "draft_year": [2024] * 4,
            "nfl_id": [301, 301, 302, 302],
            "entity_type": ["PLAYER"] * 4,
            "drill_type": ["SKILL_DRILLS_DL"] * 4,
            "drill_name": ["RUN_THE_HOOP_DRILL"] * 4,
            "attempt": [1] * 4,
            "s": [3.0, 6.0, 2.5, 5.5],
            "a": [1.0, 4.0, 1.0, 3.5],
        }
    )

    dl_feats = extractor.extract_dl_hoop_kinematics(synthetic_dl)
    assert dl_feats.height == 2
    assert "hoop_speed_z" in dl_feats.columns
    assert "hoop_accel_z" in dl_feats.columns


def test_ol_mirror_kinematics() -> None:
    """Test OL pass protection mirror drill kinematics extraction."""
    config = Config()
    extractor = CombineFeatureExtractor(config)

    synthetic_ol = pl.DataFrame(
        {
            "draft_year": [2024] * 4,
            "nfl_id": [401, 401, 402, 402],
            "entity_type": ["PLAYER"] * 4,
            "drill_type": ["SKILL_DRILLS_OL"] * 4,
            "drill_name": ["PASS_PRO_MIRROR_DRILL"] * 4,
            "attempt": [1] * 4,
            "s": [2.0, 4.5, 1.8, 4.0],
            "a": [1.0, 3.2, 0.8, 2.7],
        }
    )

    ol_feats = extractor.extract_ol_mirror_kinematics(synthetic_ol)
    assert ol_feats.height == 2
    assert "mirror_max_accel" in ol_feats.columns
    assert "mirror_accel_z" in ol_feats.columns


def heading_ramp(start: float, stop: float, frames: int) -> list[float]:
    """Hold `start` for 10 frames, rotate linearly to `stop`, hold for 10 frames."""
    return [start] * 10 + list(np.linspace(start, stop, frames + 1)) + [stop] * 10


def test_heading_transition_latency() -> None:
    """A 135 degree reversal at 150 deg/s takes 0.6 s to clear the central 90."""
    direction = np.array(heading_ramp(90.0, 225.0, 9))
    # Brake at 3 yd/s^2 into the plant (frame 15), then drive out at 4 yd/s^2
    speed = np.concatenate([3.0 - 0.3 * np.arange(15), 0.4 * np.arange(15)[1:] - 1.2])
    speed = np.concatenate([speed, np.full(len(direction) - len(speed), speed[-1])])

    transitions = compute_heading_transitions(speed + 3.0, direction)

    assert len(transitions) == 1
    assert transitions[0]["hip_transition_latency"] == pytest.approx(0.6)
    assert transitions[0]["turn_angle"] == pytest.approx(135.0)
    assert transitions[0]["brake_decel"] == pytest.approx(3.0)
    assert transitions[0]["drive_accel"] > 3.0


def test_heading_transition_rejects_artifacts() -> None:
    """Wrap-around, stationary jitter, truncated and shallow turns are rejected."""
    moving = np.full(30, 5.0)
    wrap = np.tile([358.0, 2.0], 15)
    assert compute_heading_transitions(moving, wrap) == []

    jitter = np.tile([90.0, 250.0], 15)
    assert compute_heading_transitions(np.full(30, 0.3), jitter) == []

    truncated = np.linspace(90.0, 225.0, 10)
    assert compute_heading_transitions(np.full(10, 5.0), truncated) == []

    shallow = np.array(heading_ramp(90.0, 150.0, 4))
    assert compute_heading_transitions(np.full(len(shallow), 5.0), shallow) == []


def test_db_hip_transition_kinematics() -> None:
    """Test W-drill latency extraction ranks a fluid DB ahead of a stiff one."""
    config = Config()
    extractor = CombineFeatureExtractor(config)

    reps = []
    for nfl_id, break_frames in [(501, 6), (502, 12), (503, 9)]:
        # backpedal -> drive, drive -> backpedal, backpedal -> drive
        direction = (
            heading_ramp(90.0, 225.0, break_frames)
            + heading_ramp(225.0, 90.0, break_frames)
            + heading_ramp(90.0, 225.0, break_frames)
        )
        reps.append(
            pl.DataFrame(
                {
                    "draft_year": 2024,
                    "event_id": f"E{nfl_id}",
                    "nfl_id": nfl_id,
                    "entity_type": "PLAYER",
                    "drill_type": "SKILL_DRILLS_DB",
                    "drill_name": "W_DRILL",
                    "time": [
                        f"2024-03-01T10:00:{i / 10:04.1f}"
                        for i in range(len(direction))
                    ],
                    "s": 3.0,
                    "dir": direction,
                }
            )
        )

    db_feats = extractor.extract_db_hip_transition_kinematics(pl.concat(reps)).sort(
        "nfl_id"
    )
    assert db_feats["nfl_id"].to_list() == [501, 502, 503]
    assert db_feats["db_transition_count"].to_list() == [3, 3, 3]
    assert db_feats["db_hip_transition_latency"].to_list() == pytest.approx(
        [0.4, 0.8, 0.6]
    )
    assert db_feats["db_hip_latency_z"].to_list() == pytest.approx(
        [-1.0, 1.0, 0.0], abs=1e-3
    )


def test_pass_rusher_metrics() -> None:
    """Test GameOutcomeExtractor DL pass rush metrics extraction."""
    config = Config()
    extractor = GameOutcomeExtractor(config)

    dl_data = pl.DataFrame(
        {
            "game_id": [2024010100, 2024010100],
            "play_id": [10, 20],
            "nfl_id": [601, 601],
            "lined_up_position": ["EDGE", "EDGE"],
            "player_get_off": [0.72, 0.68],
            "time_to_pressure": [2.4, 2.1],
            "quick_pressure": [True, False],
            "sack": [1.0, 0.0],
            "tackle_for_loss": [1, 0],
        }
    )

    dl_metrics = extractor.extract_pass_rusher_metrics(dl_data)
    assert dl_metrics.height == 1
    assert dl_metrics["total_rush_snaps"][0] == 2
    assert dl_metrics["total_sacks"][0] == 1.0
    assert pytest.approx(dl_metrics["avg_get_off_time"][0], 0.01) == 0.70


def test_offensive_line_metrics() -> None:
    """Test GameOutcomeExtractor OL pass protection metrics extraction."""
    config = Config()
    extractor = GameOutcomeExtractor(config)

    ol_data = pl.DataFrame(
        {
            "game_id": [2024010100, 2024010100],
            "play_id": [30, 40],
            "nfl_id": [701, 701],
            "lined_up_position": ["T", "T"],
            "pressure_allowed": [False, True],
            "sack_allowed": [0.0, 1.0],
            "peak_pressure_probability_allowed": [0.15, 0.85],
            "time_to_pressure_allowed": [3.2, 2.3],
        }
    )

    ol_metrics = extractor.extract_offensive_line_metrics(ol_data)
    assert ol_metrics.height == 1
    assert ol_metrics["total_pass_pro_snaps"][0] == 2
    assert ol_metrics["total_sacks_allowed"][0] == 1.0
    assert pytest.approx(ol_metrics["pressure_rate_allowed"][0], 0.01) == 0.50


def test_defensive_back_metrics() -> None:
    """Test GameOutcomeExtractor DB coverage outcome extraction."""
    config = Config()
    extractor = GameOutcomeExtractor(config)

    db_data = pl.DataFrame(
        {
            "game_id": [2024010100, 2024010100],
            "play_id": [50, 60],
            "nfl_id": [801, 801],
            "lined_up_position": ["CB", "CB"],
            "pass_result": ["C", "IN"],
            "blitzing": [False, False],
            "expected_points_added": [0.45, -0.65],
            "coverage_assignment": ["MAN", "MAN"],
        }
    )

    db_metrics = extractor.extract_defensive_back_metrics(db_data)
    assert db_metrics.height == 1
    assert db_metrics["total_coverage_snaps"][0] == 2
    assert pytest.approx(db_metrics["completion_rate_allowed"][0], 0.01) == 0.50
    assert pytest.approx(db_metrics["man_coverage_rate"][0], 0.01) == 1.0


def test_sprint_mechanics_and_kinetic_energy() -> None:
    """Test sprint mechanics decomposition and kinetic energy calculation."""
    speed = np.array([0.0, 2.0, 5.0, 8.0, 10.0, 10.2, 10.1, 9.9, 9.8, 9.7])
    res = compute_sprint_mechanics(speed, fps=10.0, weight_lbs=200.0)

    assert pytest.approx(res["v_max"], 0.01) == 10.2
    assert pytest.approx(res["time_to_v_max"], 0.01) == 0.5
    assert res["dist_to_v_max"] > 0
    assert res["kinetic_energy_peak"] > 0
    # Mass = 200 * 0.45359237 = 90.718 kg
    # V = 10.2 * 0.9144 = 9.32688 m/s
    # Ek = 0.5 * 90.718 * (9.32688^2) = ~3945 Joules
    assert pytest.approx(res["kinetic_energy_peak"], 10.0) == 3945.7


def test_trench_momentum() -> None:
    """Test trench momentum calculation p = mv."""
    # 300 lbs lineman moving at 6.0 yd/s
    momentum = compute_trench_momentum(weight_lbs=300.0, peak_speed_yds=6.0)
    # Mass = 300 * 0.45359237 = 136.077 kg
    # V = 6.0 * 0.9144 = 5.4864 m/s
    # p = 136.077 * 5.4864 = 746.52 kg * m/s
    assert pytest.approx(momentum, 1.0) == 746.5


def test_break_on_ball_efficiency() -> None:
    """Test Break-on-Ball Efficiency (BOBE) calculation from DB transitions."""
    transitions = [
        {"brake_decel": 4.5, "drive_accel": 3.0, "hip_transition_latency": 0.4},
        {"brake_decel": 5.5, "drive_accel": 3.5, "hip_transition_latency": 0.5},
    ]
    bobe_res = compute_break_on_ball_efficiency(transitions)
    assert pytest.approx(bobe_res["mean_brake_decel"], 0.01) == 5.0
    assert pytest.approx(bobe_res["mean_drive_accel"], 0.01) == 3.25
    assert pytest.approx(bobe_res["mean_hip_latency"], 0.01) == 0.45
    # bobe = 3.25 / (0.45 + 0.1) = 3.25 / 0.55 = 5.909
    assert pytest.approx(bobe_res["break_on_ball_efficiency"], 0.01) == 5.91


def test_centripetal_mechanics() -> None:
    """Test centripetal acceleration and trajectory curvature calculation."""
    # Circular arc trajectory: radius R = 5 yards, speed v = 5 yd/s
    theta = np.linspace(0, np.pi, 20)
    x = 5.0 * np.cos(theta)
    y = 5.0 * np.sin(theta)
    res = compute_centripetal_mechanics(x, y, fps=10.0, trim_boundary_frames=2)

    assert res["peak_curvature"] > 0.1
    assert res["peak_centripetal"] > 1.0
    assert res["mean_bend_speed"] > 0.5


def test_zone_coverage_efficiency() -> None:
    """Test Zone Coverage Reaction Index (ZCRI) calculation."""
    w_transitions = [
        {"brake_decel": 4.0, "drive_accel": 3.0, "hip_transition_latency": 0.4}
    ]
    box_transitions = [
        {"brake_decel": 5.0, "drive_accel": 4.0, "hip_transition_latency": 0.5}
    ]
    res = compute_zone_coverage_efficiency(w_transitions, box_transitions)

    assert pytest.approx(res["mean_plant_brake"], 0.01) == 4.5
    assert pytest.approx(res["mean_drive_accel"], 0.01) == 3.5
    assert pytest.approx(res["mean_hip_latency"], 0.01) == 0.45
    # zcri = (4.5 + 3.5) / (0.45 + 0.15) = 8.0 / 0.60 = 13.333
    assert pytest.approx(res["zone_coverage_reaction_index"], 0.01) == 13.33


def test_flat_coverage_index() -> None:
    """Test Flat Coverage Reaction Index (FCRI) calculation."""
    res = compute_flat_coverage_index(
        plant_decel=4.5, drive_burst=3.5, transition_latency=0.35
    )
    assert pytest.approx(res["plant_decel"], 0.01) == 4.5
    assert pytest.approx(res["drive_burst"], 0.01) == 3.5
    assert pytest.approx(res["transition_latency"], 0.01) == 0.35
    # fcri = (4.5 + 3.5) / (0.35 + 0.10) = 8.0 / 0.45 = 17.7778
    assert pytest.approx(res["flat_coverage_reaction_index"], 0.01) == 17.78


def test_edge_disruption_index() -> None:
    """Test Edge Disruption Index (EDI) combining bend and burst."""
    res = compute_edge_disruption_index(
        peak_centripetal=4.0, peak_burst_accel=3.0, weight_lbs=253.53
    )
    # base_edi = 0.60 * 4.0 + 0.40 * 3.0 = 2.4 + 1.2 = 3.6
    assert pytest.approx(res["edge_disruption_index"], 0.01) == 3.60
    # Mass = 253.53 * 0.45359237 = 115.0 kg -> scaling = 1.0
    assert pytest.approx(res["mass_scaled_edi"], 0.05) == 3.60

    # With club efficiency
    res_club = compute_edge_disruption_index(
        peak_centripetal=4.0,
        peak_burst_accel=3.0,
        club_efficiency=1.2,
        weight_lbs=253.53,
    )
    # base_edi = 0.40 * 4.0 + 0.35 * (1.2 * 3.0) + 0.25 * 3.0 = 1.6 + 1.26 + 0.75 = 3.61
    assert pytest.approx(res_club["edge_disruption_index"], 0.01) == 3.61
    assert pytest.approx(res_club["club_efficiency"], 0.01) == 1.20


def test_situational_datr() -> None:
    """Test Situational DATR under high leverage downs and red zone."""
    res_base = compute_situational_datr(braking_decel=6.0, burst_accel=3.0)
    assert pytest.approx(res_base["raw_datr"], 0.01) == 2.0
    assert pytest.approx(res_base["situational_multiplier"], 0.01) == 1.0

    res_money = compute_situational_datr(
        braking_decel=6.0, burst_accel=3.0, is_money_down=True
    )
    assert pytest.approx(res_money["situational_multiplier"], 0.01) == 1.25
    assert pytest.approx(res_money["situational_datr"], 0.01) == 2.50

    res_rz = compute_situational_datr(
        braking_decel=6.0, burst_accel=3.0, is_money_down=True, is_red_zone=True
    )
    assert pytest.approx(res_rz["situational_multiplier"], 0.01) == 1.25 * 1.15


def test_ball_tracking_stability() -> None:
    """Test ball tracking stability and speed CV calculation."""
    # Constant speed = perfect stability (CV = 0)
    flat_speed = np.array([8.0] * 15)
    res_flat = compute_ball_tracking_stability(flat_speed)
    assert pytest.approx(res_flat["tracking_speed_mean"], 0.01) == 8.0
    assert pytest.approx(res_flat["tracking_speed_std"], 0.01) == 0.0
    assert pytest.approx(res_flat["tracking_speed_cv"], 0.01) == 0.0
    assert pytest.approx(res_flat["tracking_stability_score"], 0.01) == 1.0

    # Erratic stutter speed = high CV, low stability
    erratic_speed = np.array([8.0, 4.0, 9.0, 3.0, 8.0, 2.0, 7.0, 3.0, 8.0, 4.0])
    res_erratic = compute_ball_tracking_stability(erratic_speed)
    assert res_erratic["tracking_speed_cv"] > 0.30
    assert res_erratic["tracking_stability_score"] < 0.60


def test_run_and_club_efficiency() -> None:
    """Test contact deceleration and shed burst calculation in DL drills."""
    # Tangential accel with deceleration (-4.0) into dummy and burst (+5.0) after
    tang = np.array([1.0, 2.0, 1.0, -2.0, -4.0, -1.0, 3.0, 5.0, 2.0, 1.0, 0.0, 0.0])
    norm = np.array([0.5, 0.5, 1.0, 2.0, 2.5, 3.5, 3.0, 2.0, 1.0, 0.5, 0.0, 0.0])
    res = compute_run_and_club_efficiency(tang, norm, trim_boundary_frames=1)

    assert pytest.approx(res["contact_decel"], 0.01) == 4.0
    assert pytest.approx(res["shed_burst"], 0.01) == 5.0
    # club_efficiency = 5.0 / (4.0 + 0.5) = 5.0 / 4.5 = 1.111
    assert pytest.approx(res["club_efficiency"], 0.01) == 1.11
    assert pytest.approx(res["peak_bend"], 0.01) == 3.50


def test_translation_pipeline_multi_model() -> None:
    """Test TranslationPipeline across diverse model backends (rf, ridge, gb, ensemble)."""
    from bdb_2027.modeling.pipeline import TranslationPipeline

    synthetic_df = pl.DataFrame(
        {
            "nfl_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "combine_speed": [
                4.4,
                4.5,
                4.8,
                4.9,
                4.3,
                4.4,
                5.1,
                5.0,
                4.6,
                4.7,
                4.5,
                4.6,
            ],
            "combine_accel": [
                2.5,
                2.6,
                3.1,
                3.0,
                2.1,
                2.2,
                3.5,
                3.4,
                2.8,
                2.9,
                3.2,
                3.3,
            ],
            "nfl_outcome": [2.8, 2.9, 3.4, 3.3, 2.2, 2.3, 3.8, 3.7, 3.0, 3.1, 3.5, 3.6],
        }
    )

    pipeline = TranslationPipeline(Config(n_splits=2))
    for mtype in ["rf", "ridge", "gb", "ensemble"]:
        res = pipeline.train_cv(
            features_df=synthetic_df,
            feature_cols=["combine_speed", "combine_accel"],
            target_col="nfl_outcome",
            group_col="nfl_id",
            model_type=mtype,
        )
        assert "overall_r2" in res
        assert "pearson_r" in res
        assert "feature_importance" in res
        assert len(res["fold_metrics"]) == 2


def test_interior_impulse_index() -> None:
    """Test IDL explosive lower-body power and snap get-off prediction."""
    # Test valid interior lineman: Broad jump 110 inches, weight 305 lbs
    res = compute_interior_impulse_index(
        broad_jump_in=110.0, combine_weight_lbs=305.0, ten_yd_split_s=1.75
    )
    # pred_get_off = 1.2438 - 0.0038 * 110 + 0.0009 * 305 = 1.2438 - 0.418 + 0.2745 = 1.1003
    assert pytest.approx(res["predicted_snap_get_off"], 0.01) == 1.100
    assert (
        pytest.approx(res["concentric_power_index"], 0.01) == (110.0 * 305.0) / 1000.0
    )
    assert res["quick_penetration_index"] > 0.0

    # Test edge case with zero / invalid values
    res_zero = compute_interior_impulse_index(broad_jump_in=0.0, combine_weight_lbs=0.0)
    assert res_zero["predicted_snap_get_off"] == 0.0
    assert res_zero["concentric_power_index"] == 0.0
    assert res_zero["quick_penetration_index"] == 0.0


def test_shuttle_bilateral_kinematics() -> None:
    """Test short shuttle turning detection, high-speed braking efficiency, and asymmetry."""
    # 50 frames oscillating trajectory
    t = np.linspace(0, 5, 50)
    x = 5.0 * np.sin(t)
    y = np.zeros(50)
    speed = 5.0 * np.abs(np.cos(t))
    accel = 5.0 * np.abs(np.sin(t))

    res = compute_shuttle_bilateral_kinematics(x, y, speed, accel)
    assert res["plant1_accel"] > 4.5
    assert res["plant2_accel"] > 4.5
    assert res["plant2_entry_speed"] > 4.0
    assert res["plant2_braking_efficiency"] > 1.0
    assert res["bilateral_asymmetry"] >= 0.0

    # Short array edge case
    short_res = compute_shuttle_bilateral_kinematics(
        x[:10], y[:10], speed[:10], accel[:10]
    )
    assert short_res["plant1_accel"] == 0.0
    assert short_res["bilateral_asymmetry"] == 0.0


def test_pass_protection_longevity_index() -> None:
    """Test pass protection longevity index and fatigue penalty on extended dropbacks."""
    # Test valid interior lineman: Weight 325 lbs, 10y split 1.82s, 32 bench reps
    res = compute_pass_protection_longevity_index(
        combine_weight_lbs=325.0, ten_yd_split_s=1.82, bench_reps=32.0
    )
    # pred_prob = 0.0032 + 0.0017 * 325 - 0.1272 * 1.82 = 0.0032 + 0.5525 - 0.2315 = 0.3242
    assert pytest.approx(res["predicted_extended_pressure_prob"], 0.01) == 0.324
    assert pytest.approx(res["quick_drop_anchor_score"], 0.01) == 32.0 / 30.0
    assert (
        pytest.approx(res["extended_fatigue_penalty"], 0.001)
        == (325.0 - 315.0) * 0.0017
    )

    # Test edge case with zero / invalid weight
    res_zero = compute_pass_protection_longevity_index(combine_weight_lbs=0.0)
    assert res_zero["predicted_extended_pressure_prob"] == 0.0
    assert res_zero["quick_drop_anchor_score"] == 0.0
    assert res_zero["extended_fatigue_penalty"] == 0.0


def test_motion_vector_advantage() -> None:
    """Test pre-snap motion separation boost and inertia neutralization."""
    # Test receiver in motion with 3.0 yards base separation
    res_motion = compute_motion_vector_advantage(
        in_motion_at_snap=True, base_separation=3.0, ten_yd_split_s=1.60
    )
    assert pytest.approx(res_motion["motion_boost_yards"], 0.01) == 1.534
    assert pytest.approx(res_motion["projected_separation"], 0.01) == 4.534
    assert pytest.approx(res_motion["motion_completion_lift"], 0.01) == 0.1496
    assert res_motion["inertia_neutralized"] > 0.0

    # Test receiver static at snap
    res_static = compute_motion_vector_advantage(
        in_motion_at_snap=False, base_separation=3.0
    )
    assert res_static["motion_boost_yards"] == 0.0
    assert res_static["projected_separation"] == 3.0
    assert res_static["motion_completion_lift"] == 0.0
    assert res_static["inertia_neutralized"] == 0.0


def test_career_longevity_residuals() -> None:
    """Test econometric career longevity, draft capital expectations, and kinematic surplus."""
    # Test Day 3 breakout WR: Puka Nacua profile (43 starts, draft 2023, pick 177, decel z = +1.5)
    res_wr = compute_career_longevity_residuals(
        career_games_started=43,
        draft_year=2023,
        draft_overall_pick=177.0,
        position_group="WR",
        kinematic_z_score=1.5,
    )
    assert res_wr["actual_starts_per_year"] == pytest.approx(43.0 / 3.0, 0.01)
    # expected = 24.3223 - 4.2041 * ln(177) = 24.3223 - 4.2041 * 5.17615 = 2.56
    assert res_wr["expected_starts_per_year"] < 5.0
    assert res_wr["draft_capital_residual"] > 9.0
    assert res_wr["outperformance_flag"] == 1.0
    assert res_wr["kinematic_surplus_starts"] == pytest.approx(1.5749 * 1.5, 0.01)

    # Test OL starter profile: (31 starts, draft 2024, pick 62, mirror z = +1.0)
    res_ol = compute_career_longevity_residuals(
        career_games_started=31,
        draft_year=2024,
        draft_overall_pick=62.0,
        position_group="OL",
        kinematic_z_score=1.0,
    )
    assert res_ol["actual_starts_per_year"] == pytest.approx(15.5, 0.01)
    assert res_ol["draft_capital_residual"] > 0.0
    assert res_ol["kinematic_surplus_starts"] == pytest.approx(2.0531, 0.01)

    # Test UDFA bust profile
    res_udfa = compute_career_longevity_residuals(
        career_games_started=0,
        draft_year=2024,
        draft_overall_pick=None,
    )
    assert res_udfa["actual_starts_per_year"] == 0.0
    assert res_udfa["outperformance_flag"] == 0.0


def test_overload_protection_metrics() -> None:
    """Test pass protection overload dynamics and anchor stabilization."""
    # 1-on-1 protection with high mirror acceleration (4.5 yd/s^2)
    res_single = compute_overload_protection_metrics(
        pass_rushers_encountered=1,
        mirror_max_accel_yd_s2=4.5,
    )
    assert res_single["is_overload"] == 0.0
    assert res_single["overload_penalty"] == 0.0
    assert res_single["anchor_stabilization_credit"] == 0.0
    # pred_prob = 0.450 - 0.045 * 4.5 = 0.2475
    assert pytest.approx(res_single["predicted_peak_pressure_prob"], 0.01) == 0.2475

    # Multi-rusher overload with heavy bench reps (32 reps)
    res_overload = compute_overload_protection_metrics(
        pass_rushers_encountered=2,
        mirror_max_accel_yd_s2=4.5,
        bench_reps=32.0,
    )
    assert res_overload["is_overload"] == 1.0
    assert res_overload["overload_penalty"] == 0.156
    # anchor_credit = (32 - 25) * 0.005 = 0.035
    assert pytest.approx(res_overload["anchor_stabilization_credit"], 0.001) == 0.035
    assert pytest.approx(res_overload["predicted_peak_pressure_prob"], 0.01) == 0.395


def test_zone_break_efficiency() -> None:
    """Test defensive back break-on-ball completion suppression in zone coverage."""
    # Test high forward drive CB in Zone Coverage (drive z = +1.5, latency z = -1.0)
    res_zone = compute_zone_break_efficiency(
        db_drive_accel_z=1.5,
        db_hip_latency_z=-1.0,
        is_zone_coverage=True,
    )
    # pred_delta = -0.0211 * 1.5 + 0.0054 * (-1.0) = -0.03165 - 0.0054 = -0.03705
    assert pytest.approx(res_zone["predicted_completion_delta"], 0.005) == -0.037
    assert res_zone["zone_break_index"] > 1.5
    assert res_zone["scheme_moderation_factor"] == 1.0

    # Test Man Coverage where linear drive acceleration is neutral
    res_man = compute_zone_break_efficiency(
        db_drive_accel_z=1.5,
        db_hip_latency_z=-1.0,
        is_zone_coverage=False,
    )
    assert res_man["predicted_completion_delta"] == 0.0
    assert res_man["zone_break_index"] == 0.0
    assert res_man["scheme_moderation_factor"] == 0.0


def test_tight_end_dual_role_index() -> None:
    """Test Tight End dual-role receiving and protection trade-off."""
    # Heavy inline TE (258 lbs, 26 bench reps, 4.2 yards separation)
    res_heavy = compute_tight_end_dual_role_index(
        combine_weight_lbs=258.0,
        route_separation_yards=4.2,
        bench_reps=26.0,
    )
    assert res_heavy["is_heavy_inline_profile"] == 1.0
    assert res_heavy["predicted_protection_pressure_prob"] < 0.28
    assert res_heavy["leak_separation_surplus"] == 0.20
    assert res_heavy["dual_role_utility_score"] > 5.0

    # Lighter move TE (238 lbs, 3.8 yards separation)
    res_move = compute_tight_end_dual_role_index(
        combine_weight_lbs=238.0,
        route_separation_yards=3.8,
    )
    assert res_move["is_heavy_inline_profile"] == 0.0
    assert res_move["predicted_protection_pressure_prob"] > 0.33
    assert res_move["leak_separation_surplus"] == 0.0


def test_gauntlet_yac_translation() -> None:
    """Test Gauntlet cruising velocity translation to NFL Yards After Catch."""
    # Fast cruising WR (speed = 6.2 yd/s, weight = 195 lbs)
    res_fast = compute_gauntlet_yac_translation(
        gauntlet_mean_speed_yd_s=6.2,
        combine_weight_lbs=195.0,
    )
    # pred_yac = -1.2015 + 1.7674 * 6.2 - 0.0237 * 195 = -1.2015 + 10.95788 - 4.6215 = 5.13488
    assert pytest.approx(res_fast["predicted_yac_per_reception"], 0.05) == 5.13
    assert res_fast["hands_in_stride_efficiency"] > 1.0
    assert res_fast["gauntlet_speed_premium"] > 10.0

    # Low cruising speed WR (speed = 4.2 yd/s, weight = 205 lbs)
    res_slow = compute_gauntlet_yac_translation(
        gauntlet_mean_speed_yd_s=4.2,
        combine_weight_lbs=205.0,
    )
    assert res_slow["predicted_yac_per_reception"] < 4.0
    assert res_slow["hands_in_stride_efficiency"] == 0.0


def test_edge_pressure_latency_chain() -> None:
    """Test Edge rusher concentric power to snap get-off to time-to-pressure chain."""
    # Elite explosive Edge rusher (broad jump 128 in, split 1.52s, weight 255 lbs)
    res_elite = compute_edge_pressure_latency_chain(
        broad_jump_in=128.0,
        ten_yd_split_s=1.52,
        combine_weight_lbs=255.0,
    )
    assert res_elite["predicted_snap_get_off_s"] < 0.95
    assert res_elite["predicted_time_to_pressure_s"] < 2.50
    assert res_elite["quick_pressure_threat_flag"] == 1.0

    # Stiff, heavy Edge rusher (broad jump 112 in, split 1.68s, weight 275 lbs)
    res_heavy = compute_edge_pressure_latency_chain(
        broad_jump_in=112.0,
        ten_yd_split_s=1.68,
        combine_weight_lbs=275.0,
    )
    assert res_heavy["predicted_snap_get_off_s"] > 0.95
    assert res_heavy["predicted_time_to_pressure_s"] > 2.50
    assert res_heavy["quick_pressure_threat_flag"] == 0.0


def test_weight_inertia_separation_impact() -> None:
    """Test receiver rotational inertia penalty on separation across route depths."""
    # Light receiver (178 lbs) on short route (4 yards)
    res_light = compute_weight_inertia_separation_impact(
        combine_weight_lbs=178.0,
        target_depth_yards=4.0,
    )
    assert res_light["predicted_separation_yards"] == pytest.approx(5.40, 0.05)
    assert res_light["rotational_inertia_penalty_yards"] == 0.0
    assert res_light["is_light_shifty_archetype"] == 1.0

    # Heavy receiver (218 lbs) on short route (4 yards)
    res_heavy = compute_weight_inertia_separation_impact(
        combine_weight_lbs=218.0,
        target_depth_yards=4.0,
    )
    # penalty = (218 - 180) * 0.018 = 38 * 0.018 = 0.684
    assert pytest.approx(res_heavy["rotational_inertia_penalty_yards"], 0.01) == 0.684
    assert res_heavy["predicted_separation_yards"] < 4.80
    assert res_heavy["is_light_shifty_archetype"] == 0.0


def test_press_disengagement_metrics() -> None:
    """Test receiver separation and completion under press vs off coverage."""
    # Press coverage (cushion = 2.0 yds) with elite braking deceleration (5.5 yd/s^2)
    res_press = compute_press_disengagement_metrics(
        cushion_yards=2.0,
        raw_braking_decel_yd_s2=5.5,
        combine_weight_lbs=190.0,
    )
    assert res_press["is_press_coverage"] == 1.0
    assert res_press["predicted_separation_yards"] > 2.20
    assert res_press["predicted_completion_rate"] > 0.48
    assert res_press["press_separation_compression"] > 0.0

    # Off coverage (cushion = 7.5 yds)
    res_off = compute_press_disengagement_metrics(
        cushion_yards=7.5,
        raw_braking_decel_yd_s2=4.0,
        combine_weight_lbs=190.0,
    )
    assert res_off["is_press_coverage"] == 0.0
    assert res_off["predicted_separation_yards"] > 4.0
    assert res_off["press_separation_compression"] == 0.0


def test_slot_vs_boundary_cb_risk() -> None:
    """Test cornerback completion vulnerability in slot vs boundary alignments."""
    # Slot cornerback with slower 40 time (4.55s)
    res_slot_slow = compute_slot_vs_boundary_cb_risk(
        forty_s=4.55,
        is_slot_alignment=True,
    )
    assert res_slot_slow["is_slot_alignment"] == 1.0
    # comp = 0.580 + 0.80 * (4.55 - 4.45) = 0.580 + 0.080 = 0.660
    assert (
        pytest.approx(res_slot_slow["predicted_completion_rate_allowed"], 0.01) == 0.660
    )
    assert res_slot_slow["open_field_recovery_penalty"] > 0.05

    # Slot cornerback with fast 40 time (4.38s)
    res_slot_fast = compute_slot_vs_boundary_cb_risk(
        forty_s=4.38,
        is_slot_alignment=True,
    )
    assert res_slot_fast["predicted_completion_rate_allowed"] == 0.580
    assert res_slot_fast["open_field_recovery_penalty"] == 0.0

    # Boundary cornerback (40 time independent)
    res_bnd = compute_slot_vs_boundary_cb_risk(
        forty_s=4.55,
        is_slot_alignment=False,
    )
    assert res_bnd["is_slot_alignment"] == 0.0
    assert res_bnd["predicted_completion_rate_allowed"] == 0.540
    assert res_bnd["open_field_recovery_penalty"] == 0.0


def test_tackle_extended_lever_penalty() -> None:
    """Test Offensive Tackle lever-arm recovery penalty on extended dropbacks."""
    # Extended dropback (>3.2s) with very wide wingspan (86.0 inches)
    res_wide = compute_tackle_extended_lever_penalty(
        wing_span_in=86.0,
        dropback_duration_s=3.6,
        mirror_max_accel_yd_s2=3.8,
    )
    assert res_wide["is_extended_dropback"] == 1.0
    # penalty = (86 - 80) * 0.0066 = 0.0396
    assert pytest.approx(res_wide["lever_arm_recovery_penalty"], 0.001) == 0.0396
    assert res_wide["predicted_peak_pressure_prob"] > 0.33

    # Quick rhythm dropback (2.0s)
    res_quick = compute_tackle_extended_lever_penalty(
        wing_span_in=86.0,
        dropback_duration_s=2.0,
    )
    assert res_quick["is_extended_dropback"] == 0.0
    assert res_quick["lever_arm_recovery_penalty"] == 0.0
    assert res_quick["predicted_peak_pressure_prob"] == 0.245


def test_third_down_separation_dynamics() -> None:
    """Test 3rd down target separation compression and weight resilience."""
    # Light receiver on 3rd down (<185 lbs)
    res_light_3rd = compute_third_down_separation_dynamics(
        down=3,
        combine_weight_lbs=180.0,
        baseline_separation_yards=3.90,
    )
    assert res_light_3rd["is_third_down"] == 1.0
    assert res_light_3rd["down_leverage_compression"] > 0.0
    assert res_light_3rd["predicted_completion_rate"] == 0.5898

    # Medium receiver on 3rd down (195 lbs)
    res_med_3rd = compute_third_down_separation_dynamics(
        down=3,
        combine_weight_lbs=195.0,
        baseline_separation_yards=3.90,
    )
    assert res_med_3rd["is_third_down"] == 1.0
    assert res_med_3rd["predicted_completion_rate"] == 0.5620
    # Light receiver retains more separation than medium receiver
    assert (
        res_light_3rd["predicted_separation_yards"]
        > res_med_3rd["predicted_separation_yards"]
    )

    # Early down (1st down)
    res_early = compute_third_down_separation_dynamics(
        down=1,
        combine_weight_lbs=195.0,
        baseline_separation_yards=3.90,
    )
    assert res_early["is_third_down"] == 0.0
    assert res_early["down_leverage_compression"] == 0.0
    assert res_early["predicted_separation_yards"] == 3.90
    assert res_early["predicted_completion_rate"] == 0.6650


def test_pass_rush_quick_pressure_index() -> None:
    """Test pass rush pocket collapse latency and quick pressure rate."""
    # Elite edge rusher with 1.55s 10y split and 125-inch broad jump
    res_elite = compute_pass_rush_quick_pressure_index(
        ten_yd_split_s=1.55,
        broad_jump_in=125.0,
        combine_weight_lbs=252.0,
    )
    assert res_elite["is_elite_burst"] == 1.0
    assert res_elite["predicted_time_to_pressure_s"] < 2.95
    assert res_elite["predicted_quick_pressure_rate"] > 0.35

    # Heavy interior defensive lineman with 1.78s split and 100-inch broad jump
    res_heavy = compute_pass_rush_quick_pressure_index(
        ten_yd_split_s=1.78,
        broad_jump_in=100.0,
        combine_weight_lbs=315.0,
    )
    assert res_heavy["is_elite_burst"] == 0.0
    assert res_heavy["predicted_time_to_pressure_s"] > 3.0
    assert res_heavy["predicted_quick_pressure_rate"] < 0.20


def test_route_curvature_efficiency() -> None:
    """Test route curvature efficiency on speed cuts vs hard breaks."""
    # Speed cut route (curvilinear)
    res_speed = compute_route_curvature_efficiency(
        route_class="speed_cut",
        baseline_separation_yards=3.50,
    )
    assert res_speed["is_speed_cut"] == 1.0
    # Separation is compressed: 3.50 - 0.451 = 3.049
    assert pytest.approx(res_speed["predicted_separation_yards"], 0.01) == 3.049
    assert res_speed["predicted_yac_yards"] > 5.0
    assert res_speed["predicted_epa_per_play"] > 0.25

    # Hard break route (angle cut)
    res_hard = compute_route_curvature_efficiency(
        route_class="hard_break",
        baseline_separation_yards=3.50,
    )
    assert res_hard["is_speed_cut"] == 0.0
    assert res_hard["predicted_separation_yards"] == 3.50
    assert res_hard["predicted_yac_yards"] < 3.5
    assert res_hard["predicted_epa_per_play"] < 0.15


def test_day3_special_teams_survival_index() -> None:
    """Test Day 3 special teams roster survival index."""
    # High ST contributor (>=50 snaps) with fast 40 and explosive broad jump
    res_high = compute_day3_special_teams_survival_index(
        forty_s=4.45,
        broad_jump_in=124.0,
        career_st_snaps=120,
    )
    assert res_high["is_high_st_role"] == 1.0
    assert res_high["predicted_career_games_active"] > 25.0
    assert res_high["special_teams_readiness_score"] > 0.60

    # Low ST contributor (<50 snaps)
    res_low = compute_day3_special_teams_survival_index(
        forty_s=4.58,
        broad_jump_in=110.0,
        career_st_snaps=15,
    )
    assert res_low["is_high_st_role"] == 0.0
    assert res_low["predicted_career_games_active"] < 12.0


def test_red_zone_compression_metrics() -> None:
    """Test red zone spatial compression and contested catch probability."""
    # Red zone target (yardline <= 20)
    res_rz = compute_red_zone_compression_metrics(
        yardline_number=14.0,
        baseline_separation_yards=3.75,
    )
    assert res_rz["is_red_zone"] == 1.0
    assert pytest.approx(res_rz["red_zone_compression_yards"], 0.001) == 0.419
    # 3.75 - 0.419 = 3.331
    assert pytest.approx(res_rz["predicted_separation_yards"], 0.01) == 3.331
    assert res_rz["contested_catch_probability"] > 0.40

    # Open field target (yardline 45)
    res_open = compute_red_zone_compression_metrics(
        yardline_number=45.0,
        baseline_separation_yards=3.75,
    )
    assert res_open["is_red_zone"] == 0.0
    assert res_open["red_zone_compression_yards"] == 0.0
    assert res_open["predicted_separation_yards"] == 3.75
    assert res_open["contested_catch_probability"] < 0.35


def test_coverage_rotation_seam_advantage() -> None:
    """Test offensive passing advantage against defensive coverage assignment rotations."""
    # Defense rotating assignments at snap
    res_rot = compute_coverage_rotation_seam_advantage(
        is_defense_rotating=True,
        baseline_epa=0.115,
    )
    assert res_rot["is_defense_rotating"] == 1.0
    assert res_rot["seam_separation_surplus_yards"] > 0.10
    assert pytest.approx(res_rot["predicted_epa_per_play"], 0.01) == 0.1856

    # Defense static at snap
    res_stat = compute_coverage_rotation_seam_advantage(
        is_defense_rotating=False,
        baseline_epa=0.115,
    )
    assert res_stat["is_defense_rotating"] == 0.0
    assert res_stat["seam_separation_surplus_yards"] == 0.0
    assert res_stat["predicted_epa_per_play"] == 0.115


def test_stunt_twist_disruption_metrics() -> None:
    """Test offensive line pocket collapse acceleration when absorbing defensive stunts."""
    # Guard absorbing a stunt twist (>=2 rushers)
    res_guard_stunt = compute_stunt_twist_disruption_metrics(
        is_interior_lineman=True,
        pass_rushers_encountered=2,
    )
    assert res_guard_stunt["is_stunt_overload"] == 1.0
    assert res_guard_stunt["is_interior_lineman"] == 1.0
    assert res_guard_stunt["predicted_peak_pressure"] == 0.430
    assert pytest.approx(res_guard_stunt["predicted_time_to_pressure_s"], 0.01) == 2.799
    assert res_guard_stunt["time_to_pressure_reduction_s"] > 0.20
    assert res_guard_stunt["predicted_sack_probability"] > 0.03

    # Tackle in standard 1-on-1 protection
    res_ot_1on1 = compute_stunt_twist_disruption_metrics(
        is_interior_lineman=False,
        pass_rushers_encountered=1,
    )
    assert res_ot_1on1["is_stunt_overload"] == 0.0
    assert res_ot_1on1["predicted_peak_pressure"] == 0.274
    assert res_ot_1on1["predicted_time_to_pressure_s"] > 3.0
    assert res_ot_1on1["time_to_pressure_reduction_s"] == 0.0


def test_yacoe_momentum_index() -> None:
    """Test projected YAC Over Expected (YACOE) and total YAC from catch-point momentum."""
    # Heavy Tight End (255 lbs, 6.0 yd/s Gauntlet speed)
    res_te = compute_yacoe_momentum_index(
        combine_weight_lbs=255.0,
        gauntlet_mean_speed_yd_s=6.0,
        expected_yac_yards=4.60,
    )
    assert res_te["is_high_contact_balance"] == 1.0
    assert res_te["predicted_yacoe_yards"] > 1.0
    assert res_te["predicted_total_yac_yards"] > 5.60
    assert res_te["catch_momentum_ratio"] > 1.20

    # Lightweight receiver (175 lbs, 5.2 yd/s)
    res_light = compute_yacoe_momentum_index(
        combine_weight_lbs=175.0,
        gauntlet_mean_speed_yd_s=5.2,
        expected_yac_yards=4.60,
    )
    assert res_light["is_high_contact_balance"] == 0.0
    assert res_light["predicted_yacoe_yards"] < 0.35


def test_rusher_getoff_trench_rupture() -> None:
    """Test pocket collapse and pressure risk from rusher snap get-off latency."""
    # Fast rusher get-off (<0.73s) against Offensive Tackle
    res_fast_ot = compute_rusher_getoff_trench_rupture(
        player_get_off_s=0.65,
        is_tackle=True,
    )
    assert res_fast_ot["is_fast_get_off"] == 1.0
    assert res_fast_ot["get_off_tier"] == 1.0
    assert res_fast_ot["predicted_peak_pressure"] > 0.34
    assert res_fast_ot["predicted_pressure_rate"] > 0.12
    assert res_fast_ot["predicted_sack_rate"] > 0.015
    assert res_fast_ot["predicted_time_to_pressure_s"] < 2.90
    assert res_fast_ot["pocket_collapse_acceleration_s"] > 0.30

    # Slow rusher get-off (>0.87s) against Offensive Tackle
    res_slow_ot = compute_rusher_getoff_trench_rupture(
        player_get_off_s=0.95,
        is_tackle=True,
    )
    assert res_slow_ot["is_fast_get_off"] == 0.0
    assert res_slow_ot["get_off_tier"] == 3.0
    assert res_slow_ot["predicted_peak_pressure"] < 0.28
    assert res_slow_ot["predicted_pressure_rate"] < 0.09
    assert res_slow_ot["predicted_sack_rate"] < 0.008
    assert res_slow_ot["predicted_time_to_pressure_s"] > 3.30

    # Interior Lineman (Guard) against Fast get-off
    res_fast_g = compute_rusher_getoff_trench_rupture(
        player_get_off_s=0.68,
        is_tackle=False,
    )
    assert res_fast_g["predicted_peak_pressure"] > 0.29
    assert res_fast_g["predicted_time_to_pressure_s"] < 2.95


def test_man_coverage_reach_advantage() -> None:
    """Test defensive back man coverage incompletion leverage from arm length."""
    # Long Arm DB (32.5 in, 4.45s 40)
    res_long = compute_man_coverage_reach_advantage(
        arm_length_in=32.5,
        combine_forty_s=4.45,
    )
    assert res_long["is_long_arm_threshold"] == 1.0
    assert res_long["reach_leverage_in"] == 2.0
    assert res_long["predicted_incompletion_rate_forced"] > 0.43
    assert res_long["predicted_completion_rate_allowed"] < 0.49
    assert res_long["predicted_epa_allowed_per_snap"] < -0.15
    assert pytest.approx(res_long["predicted_presnap_cushion_yards"], 0.01) == 5.375

    # Short Arm DB (29.5 in, 4.55s 40)
    res_short = compute_man_coverage_reach_advantage(
        arm_length_in=29.5,
        combine_forty_s=4.55,
    )
    assert res_short["is_long_arm_threshold"] == 0.0
    assert res_short["reach_leverage_in"] == -1.0
    assert res_short["predicted_incompletion_rate_forced"] < 0.39
    assert res_short["predicted_completion_rate_allowed"] > 0.53
    assert res_short["predicted_epa_allowed_per_snap"] > -0.02
    assert pytest.approx(res_short["predicted_presnap_cushion_yards"], 0.01) == 5.625


def test_press_release_polar_reversal() -> None:
    """Test target separation and completion under press vs off coverage based on receiver weight."""
    # Heavy WR (215 lbs) facing Press (2.0 yds cushion)
    res_heavy_press = compute_press_release_polar_reversal(
        combine_weight_lbs=215.0,
        cushion_yards=2.0,
    )
    assert res_heavy_press["is_press_coverage"] == 1.0
    assert res_heavy_press["is_off_coverage"] == 0.0
    assert res_heavy_press["press_mass_leverage_index"] > 1.0
    assert res_heavy_press["predicted_target_separation_yards"] > 2.40
    assert res_heavy_press["predicted_completion_rate"] > 0.53

    # Light WR (175 lbs) facing Press (2.0 yds cushion)
    res_light_press = compute_press_release_polar_reversal(
        combine_weight_lbs=175.0,
        cushion_yards=2.0,
    )
    assert res_light_press["predicted_target_separation_yards"] < 2.30
    assert res_light_press["predicted_completion_rate"] < 0.48

    # Light WR (175 lbs) facing Off Coverage (7.0 yds cushion)
    res_light_off = compute_press_release_polar_reversal(
        combine_weight_lbs=175.0,
        cushion_yards=7.0,
    )
    assert res_light_off["is_off_coverage"] == 1.0
    assert res_light_off["predicted_target_separation_yards"] > 4.20
    assert res_light_off["predicted_completion_rate"] > 0.68

    # Heavy WR (215 lbs) facing Off Coverage (7.0 yds cushion)
    res_heavy_off = compute_press_release_polar_reversal(
        combine_weight_lbs=215.0,
        cushion_yards=7.0,
    )
    assert res_heavy_off["predicted_target_separation_yards"] < 3.75


def test_inside_10_target_efficiency() -> None:
    """Test red zone conversion efficiency and separation inside the 10-yard line."""
    # Tight End (250 lbs) inside 10 (yardline 6)
    res_te = compute_inside_10_target_efficiency(
        combine_weight_lbs=250.0,
        is_tight_end=True,
        yardline_number=6.0,
    )
    assert res_te["is_tight_end"] == 1.0
    assert res_te["predicted_inside_10_separation_yards"] > 3.30
    assert res_te["predicted_completion_rate"] > 0.72
    assert res_te["predicted_epa_per_target"] >= 0.20

    # Outside WR (190 lbs) inside 10 (yardline 6)
    res_wr = compute_inside_10_target_efficiency(
        combine_weight_lbs=190.0,
        is_tight_end=False,
        yardline_number=6.0,
    )
    assert res_wr["is_tight_end"] == 0.0
    assert res_wr["predicted_inside_10_separation_yards"] < 2.45
    assert res_wr["predicted_completion_rate"] < 0.55
    assert res_wr["separation_compression_delta_yards"] > 1.35


def test_presnap_motion_advantage() -> None:
    """Test target separation surplus and completion probability from pre-snap motion."""
    # Receiver In-Motion vs Man Coverage
    res_mot_man = compute_presnap_motion_advantage(
        in_motion_at_snap=True,
        is_man_coverage=True,
    )
    assert res_mot_man["is_in_motion"] == 1.0
    assert res_mot_man["is_man_coverage"] == 1.0
    assert res_mot_man["predicted_target_separation_yards"] > 3.75
    assert res_mot_man["predicted_completion_rate"] > 0.70
    assert res_mot_man["motion_separation_surplus_yards"] > 1.30

    # Receiver Static vs Man Coverage
    res_stat_man = compute_presnap_motion_advantage(
        in_motion_at_snap=False,
        is_man_coverage=True,
    )
    assert res_stat_man["is_in_motion"] == 0.0
    assert res_stat_man["predicted_target_separation_yards"] < 2.50
    assert res_stat_man["predicted_completion_rate"] < 0.55
    assert res_stat_man["motion_separation_surplus_yards"] == 0.0

    # Receiver In-Motion vs Zone Coverage
    res_mot_zone = compute_presnap_motion_advantage(
        in_motion_at_snap=True,
        is_man_coverage=False,
    )
    assert res_mot_zone["predicted_target_separation_yards"] > 5.50
    assert res_mot_zone["predicted_completion_rate"] > 0.77


def test_pocket_halflife_degradation() -> None:
    """Test pass protection degradation and pocket collapse probability over dropback duration."""
    # Quick rhythm drop (2.0s) for Offensive Tackle
    res_quick_t = compute_pocket_halflife_degradation(
        dropback_duration_s=2.0,
        is_tackle=True,
    )
    assert res_quick_t["dropback_duration_tier"] == 1.0
    assert res_quick_t["is_extended_scramble"] == 0.0
    assert res_quick_t["predicted_peak_pressure"] < 0.22
    assert res_quick_t["predicted_pressure_allowed_rate"] < 0.02
    assert res_quick_t["predicted_sack_allowed_rate"] == 0.0
    assert res_quick_t["pocket_pressure_multiplication_factor"] <= 1.0

    # Extended scramble drop (4.0s) for Offensive Tackle
    res_ext_t = compute_pocket_halflife_degradation(
        dropback_duration_s=4.0,
        is_tackle=True,
    )
    assert res_ext_t["dropback_duration_tier"] == 3.0
    assert res_ext_t["is_extended_scramble"] == 1.0
    assert res_ext_t["predicted_peak_pressure"] > 0.38
    assert res_ext_t["predicted_pressure_allowed_rate"] > 0.17
    assert res_ext_t["predicted_sack_allowed_rate"] > 0.025
    assert res_ext_t["pocket_pressure_multiplication_factor"] > 10.0

    # Quick drop for Guard
    res_quick_g = compute_pocket_halflife_degradation(
        dropback_duration_s=2.0,
        is_tackle=False,
    )
    assert res_quick_g["predicted_pressure_allowed_rate"] < 0.025
    assert res_quick_g["predicted_sack_allowed_rate"] == 0.0


def test_disruption_hierarchy_penalty() -> None:
    """Test passing efficiency collapse and sack hazard under hierarchical pocket disruption."""
    # Clean Pocket
    res_clean = compute_disruption_hierarchy_penalty(
        has_quick_pressure=False,
        has_unblocked_pressure=False,
        has_late_pressure=False,
    )
    assert res_clean["disruption_severity_tier"] == 1.0
    assert res_clean["predicted_completion_rate"] > 0.55
    assert res_clean["predicted_sack_probability"] < 0.05
    assert res_clean["predicted_epa_per_play"] > 0.0
    assert pytest.approx(res_clean["disruption_epa_penalty"], 0.01) == 0.0

    # Quick Blocked Pressure (<=2.5s)
    res_quick = compute_disruption_hierarchy_penalty(
        has_quick_pressure=True,
        has_unblocked_pressure=False,
    )
    assert res_quick["disruption_severity_tier"] == 4.0
    assert res_quick["is_timing_ruptured"] == 1.0
    assert res_quick["predicted_completion_rate"] < 0.40
    assert res_quick["predicted_sack_probability"] > 0.20
    assert res_quick["predicted_epa_per_play"] < -0.50
    assert res_quick["disruption_epa_penalty"] > 0.50

    # Unblocked Free Rusher
    res_unblk = compute_disruption_hierarchy_penalty(
        has_quick_pressure=False,
        has_unblocked_pressure=True,
    )
    assert res_unblk["disruption_severity_tier"] == 3.0
    assert res_unblk["predicted_sack_probability"] > 0.15


def test_unified_translation_composite() -> None:
    """Test the unified kinematic translation composite index across all five core physical dimensions."""
    # Elite 5-Tool Prospect (All +1.5 z-score)
    res_elite = compute_unified_translation_composite(
        concentric_power_z=1.5,
        contact_momentum_z=1.5,
        reach_leverage_z=1.5,
        motion_velocity_z=1.5,
        spatial_compression_z=1.5,
    )
    assert res_elite["is_elite_translation_tier"] == 1.0
    assert res_elite["unified_kinematic_composite_z"] == 1.5
    assert res_elite["predicted_starter_survival_probability"] > 0.65

    # Sub-par Prospect (All -1.0 z-score)
    res_sub = compute_unified_translation_composite(
        concentric_power_z=-1.0,
        contact_momentum_z=-1.0,
        reach_leverage_z=-1.0,
        motion_velocity_z=-1.0,
        spatial_compression_z=-1.0,
    )
    assert res_sub["is_elite_translation_tier"] == 0.0
    assert res_sub["unified_kinematic_composite_z"] == -1.0
    assert res_sub["predicted_starter_survival_probability"] < 0.35


def test_down_and_distance_pressure_escalation() -> None:
    """Test Section 51 down-and-distance pass rush get-off escalation and protection attrition."""
    # Neutral down baseline (1st & 10)
    neutral = compute_down_and_distance_pressure_escalation(
        base_get_off=1.05,
        is_third_and_long=False,
    )
    assert neutral["escalated_get_off"] == 1.05
    assert neutral["get_off_delta_s"] == 0.0
    assert neutral["expected_quick_pressure_rate"] == 0.2636
    assert neutral["pressure_escalation_ratio"] == 1.0

    # 3rd & long with elite concentric power (broad jump 128 in, split 1.55s)
    elite_rusher = compute_down_and_distance_pressure_escalation(
        base_get_off=1.05,
        broad_jump_in=128.0,
        ten_yd_split=1.55,
        is_third_and_long=True,
    )
    assert elite_rusher["escalated_get_off"] < 0.85
    assert elite_rusher["expected_quick_pressure_rate"] > 0.35
    assert elite_rusher["is_high_threat_rusher"] == 1.0
    assert elite_rusher["tackle_pressure_allowed_rate"] > 0.13
    assert elite_rusher["tackle_sack_allowed_rate"] > 0.02


def test_route_stem_kinematic_smoothness() -> None:
    """Test Section 52 route stem kinematic smoothness ratio and translation metrics."""
    # Smooth separator: high cruising speed (6.0 yd/s), low jerk/accel spike (8.0 yd/s^2)
    speeds_smooth = np.array(
        [4.0, 5.0, 6.0, 6.5, 6.2, 5.8, 6.0, 6.4, 6.1, 5.9, 6.0, 6.2]
    )
    accels_smooth = np.array(
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]
    )
    res_smooth = compute_route_stem_kinematic_smoothness(speeds_smooth, accels_smooth)
    assert res_smooth["kinematic_route_smoothness_ratio"] > 0.70
    assert res_smooth["is_elite_smooth_separator"] == 1.0
    assert res_smooth["predicted_deep_comp_rate"] > 0.40
    assert res_smooth["predicted_sharp_separation_yds"] > 3.65

    # Choppy runner: low cruising speed (3.5 yd/s), erratic acceleration spike (12.0 yd/s^2)
    speeds_choppy = np.array(
        [2.0, 3.0, 3.5, 4.0, 2.5, 3.0, 3.5, 4.0, 3.0, 2.5, 3.0, 3.2]
    )
    accels_choppy = np.array(
        [3.0, 5.0, 9.0, 12.0, 10.0, 8.0, 7.0, 6.0, 4.0, 3.0, 2.0, 1.0]
    )
    res_choppy = compute_route_stem_kinematic_smoothness(speeds_choppy, accels_choppy)
    assert res_choppy["kinematic_route_smoothness_ratio"] < 0.35
    assert res_choppy["is_elite_smooth_separator"] == 0.0
    assert res_choppy["predicted_deep_comp_rate"] < 0.40
    assert res_choppy["predicted_sharp_separation_yds"] < 3.50

    # Short sequence fallback
    res_short = compute_route_stem_kinematic_smoothness(
        np.array([4.0]), np.array([2.0])
    )
    assert res_short["kinematic_route_smoothness_ratio"] == 0.5736


def test_stunt_overload_protection_hazard() -> None:
    """Test Section 53 stunt & twist multi-rusher protection overload hazard."""
    # 1-on-1 single rusher
    res_single = compute_stunt_overload_protection_hazard(pass_rushers_encountered=1)
    assert res_single["is_stunt_encountered"] == 0.0
    assert res_single["escalated_pressure_rate"] == 0.0822
    assert res_single["pressure_multiplier"] == 1.0

    # Multi-rusher stunt overload (>= 2 rushers)
    res_stunt = compute_stunt_overload_protection_hazard(pass_rushers_encountered=2)
    assert res_stunt["is_stunt_encountered"] == 1.0
    assert res_stunt["escalated_pressure_rate"] > 0.20
    assert res_stunt["escalated_sack_rate"] > 0.02
    assert res_stunt["peak_pressure_prob"] > 0.40
    assert res_stunt["expected_time_to_pressure_s"] < 2.90

    # Agile lineman with short shuttle (4.45s) mitigates breakdown
    res_agile = compute_stunt_overload_protection_hazard(
        pass_rushers_encountered=2,
        short_shuttle_s=4.45,
    )
    assert res_agile["pressure_multiplier"] < res_stunt["pressure_multiplier"]


def test_contested_catch_separation_cliff() -> None:
    """Test Section 54 contested target separation cliff and catch probability."""
    # Contested catch target (0.5 yd separation)
    res_contested = compute_contested_catch_separation_cliff(separation_yds=0.5)
    assert res_contested["is_contested_target"] == 1.0
    assert res_contested["is_open_target"] == 0.0
    assert res_contested["predicted_catch_probability"] < 0.35
    assert res_contested["predicted_epa_per_target"] < -0.30
    assert res_contested["separation_cliff_penalty_epa"] > 0.50

    # Open target (3.5 yds separation)
    res_open = compute_contested_catch_separation_cliff(separation_yds=3.5)
    assert res_open["is_contested_target"] == 0.0
    assert res_open["is_open_target"] == 1.0
    assert res_open["predicted_catch_probability"] > 0.70
    assert res_open["predicted_epa_per_target"] > 0.15


def test_formation_spatial_leverage_advantage() -> None:
    """Test Section 54 Singleback play-action leverage and protection expansion."""
    # Standard Shotgun base
    shotgun = compute_formation_spatial_leverage_advantage(formation="SHOTGUN")
    assert shotgun["is_singleback_leverage"] == 0.0
    assert shotgun["expected_pocket_duration_s"] == 2.928
    assert shotgun["expected_passing_epa_per_target"] == 0.125

    # Singleback play action with heavy prospect (225 lbs TE/WR)
    singleback = compute_formation_spatial_leverage_advantage(
        formation="SINGLEBACK",
        prospect_weight_lbs=225.0,
    )
    assert singleback["is_singleback_leverage"] == 1.0
    assert singleback["expected_pocket_duration_s"] > 3.35
    assert singleback["pressure_reduction_factor"] > 0.30
    assert singleback["target_separation_surplus_yds"] > 0.50
    assert singleback["expected_passing_epa_per_target"] > 0.30


def test_cover0_blitz_temporal_compression() -> None:
    """Test Section 55 zero-safety Cover-0 blitz temporal collapse and quick pressure."""
    # Base Cover-3 zone
    cov3 = compute_cover0_blitz_temporal_compression(is_cover_0=False)
    assert cov3["is_cover_0_man"] == 0.0
    assert cov3["expected_dropback_duration_s"] > 3.20
    assert cov3["is_emergency_release_required"] == 0.0

    # Cover-0 blitz with elite rusher
    cov0 = compute_cover0_blitz_temporal_compression(
        is_cover_0=True,
        rusher_get_off=0.80,
        is_blitzing=True,
    )
    assert cov0["is_cover_0_man"] == 1.0
    assert cov0["expected_dropback_duration_s"] < 2.60
    assert cov0["temporal_compression_s"] < -0.70
    assert cov0["quick_pressure_probability"] > 0.55
    assert cov0["unblocked_pressure_risk"] > 0.04
    assert cov0["is_emergency_release_required"] == 1.0


def test_motion_cross_vector_amplification() -> None:
    """Test Section 56 motion cross vector velocity amplification."""
    # Static crossing route
    static_cross = compute_motion_cross_vector_amplification(is_motion=False)
    assert static_cross["is_motion_cross"] == 0.0
    assert static_cross["predicted_separation_yds"] == 3.55
    assert static_cross["predicted_completion_rate"] == 0.657
    assert static_cross["expected_yac_yds"] == 5.77

    # Motion crossing route at 5.5 yd/s
    motion_cross = compute_motion_cross_vector_amplification(
        is_motion=True,
        motion_speed_mps=5.5,
    )
    assert motion_cross["is_motion_cross"] == 1.0
    assert motion_cross["predicted_separation_yds"] > 5.0
    assert motion_cross["predicted_completion_rate"] > 0.75
    assert motion_cross["expected_yac_yds"] > 7.0
    assert motion_cross["collinear_separation_boost"] > 1.80


def test_interior_dt_penetration_index() -> None:
    """Test Section 57 interior defensive tackle penetration index and TFL hazard."""
    # Penetrating 3-tech DT (high broad jump 118 in, 298 lbs)
    dt_penetrator = compute_interior_dt_penetration_index(
        broad_jump_in=118.0,
        combine_weight_lbs=298.0,
    )
    assert dt_penetrator["is_penetrating_three_tech"] == 1.0
    assert dt_penetrator["is_space_eating_nose"] == 0.0
    assert dt_penetrator["penetration_index_z"] > 0.80
    assert dt_penetrator["predicted_tfl_rate"] > 0.0090
    assert dt_penetrator["predicted_get_off_s"] < 1.05

    # Space-eating nose tackle (low broad jump 106 in, 335 lbs)
    dt_nose = compute_interior_dt_penetration_index(
        broad_jump_in=106.0,
        combine_weight_lbs=335.0,
    )
    assert dt_nose["is_penetrating_three_tech"] == 0.0
    assert dt_nose["is_space_eating_nose"] == 1.0
    assert dt_nose["penetration_index_z"] < -0.80
    assert dt_nose["predicted_tfl_rate"] < 0.0065
    assert dt_nose["predicted_get_off_s"] > 1.10


def test_tackle_pocket_protection_decay() -> None:
    """Test Section 59 tackle pocket protection decay across dropback tiers."""
    # Quick drop (<2.5s)
    quick = compute_tackle_pocket_protection_decay(
        dropback_duration_s=2.1,
        combine_weight_lbs=312.0,
    )
    assert quick["dropback_tier"] == 0.0
    assert quick["predicted_pressure_probability"] < 0.03
    assert quick["predicted_sack_probability"] < 0.001
    assert quick["extended_inertia_penalty"] == 0.0

    # Standard drop (2.5-3.5s)
    standard = compute_tackle_pocket_protection_decay(
        dropback_duration_s=3.0,
        combine_weight_lbs=318.0,
    )
    assert standard["dropback_tier"] == 1.0
    assert 0.08 < standard["predicted_pressure_probability"] < 0.12
    assert 0.004 < standard["predicted_sack_probability"] < 0.010

    # Extended drop (>3.5s) with heavy tackle (330 lbs)
    extended = compute_tackle_pocket_protection_decay(
        dropback_duration_s=4.2,
        combine_weight_lbs=330.0,
    )
    assert extended["dropback_tier"] == 2.0
    assert extended["predicted_pressure_probability"] > 0.24
    assert extended["predicted_sack_probability"] > 0.04
    assert extended["peak_pressure_probability"] > 0.45
    assert extended["extended_inertia_penalty"] > 0.03


def test_deep_ball_size_leverage_index() -> None:
    """Test Section 60 receiver deep ball size leverage and contested catch index."""
    # Heavy big-bodied receiver (6'4", 225 lbs)
    big_wr = compute_deep_ball_size_leverage_index(
        combine_weight_lbs=225.0,
        combine_height_in=76.0,
        is_deep_route=True,
    )
    assert big_wr["is_heavy_frame_deep_threat"] == 1.0
    assert big_wr["size_leverage_index_z"] > 1.0
    assert big_wr["predicted_deep_completion_rate"] > 0.43
    assert big_wr["predicted_deep_epa"] > 0.35
    assert big_wr["separation_compression_factor"] < 0.70

    # Lightweight track receiver (5'10", 175 lbs)
    light_wr = compute_deep_ball_size_leverage_index(
        combine_weight_lbs=175.0,
        combine_height_in=70.0,
        is_deep_route=True,
    )
    assert light_wr["is_heavy_frame_deep_threat"] == 0.0
    assert light_wr["size_leverage_index_z"] < -1.0
    assert light_wr["predicted_deep_completion_rate"] < 0.35
    assert light_wr["predicted_deep_epa"] < 0.22


def test_cornerback_boundary_slot_leverage() -> None:
    """Test Section 61 boundary vs slot cornerback leverage bifurcation."""
    # Heavy Boundary Cornerback (205 lbs)
    boundary_cb = compute_cornerback_boundary_slot_leverage(
        lined_up_position="CB",
        combine_weight_lbs=205.0,
        combine_height_in=73.5,
    )
    assert boundary_cb["is_boundary_corner"] == 1.0
    assert boundary_cb["is_slot_corner"] == 0.0
    assert boundary_cb["expected_cushion_yds"] == 6.11
    assert boundary_cb["expected_press_rate"] > 0.25
    assert boundary_cb["mass_leverage_benefit_score"] > 1.0
    assert boundary_cb["predicted_completion_rate_allowed"] < 0.52

    # Slot Cornerback (185 lbs)
    slot_cb = compute_cornerback_boundary_slot_leverage(
        lined_up_position="SLOT_CB",
        combine_weight_lbs=185.0,
        combine_height_in=70.0,
    )
    assert slot_cb["is_boundary_corner"] == 0.0
    assert slot_cb["is_slot_corner"] == 1.0
    assert slot_cb["expected_cushion_yds"] == 5.45
    assert slot_cb["expected_press_rate"] < 0.18
    assert slot_cb["predicted_completion_rate_allowed"] < 0.55


def test_goalline_fade_cliff_dynamics() -> None:
    """Test Section 61 goal line fade cliff and breaking route conversion superiority."""
    # Fade route inside the 10
    fade = compute_goalline_fade_cliff_dynamics(
        route_ran="GO",
        yardline_number=5.0,
        combine_weight_lbs=215.0,
    )
    assert fade["is_fade_concept"] == 1.0
    assert fade["is_breaking_concept"] == 0.0
    assert fade["predicted_completion_rate"] < 0.35
    assert fade["predicted_separation_yds"] < 1.45
    assert fade["predicted_target_epa"] < 0.0
    assert fade["breaking_efficiency_surplus"] < 0.0

    # Breaking slant/flat inside the 10
    slant = compute_goalline_fade_cliff_dynamics(
        route_ran="SLANT",
        yardline_number=5.0,
        combine_weight_lbs=195.0,
    )
    assert slant["is_fade_concept"] == 0.0
    assert slant["is_breaking_concept"] == 1.0
    assert slant["predicted_completion_rate"] > 0.65
    assert slant["predicted_separation_yds"] > 2.80
    assert slant["predicted_target_epa"] > 0.30
    assert slant["breaking_efficiency_surplus"] > 0.30


def test_slot_alignment_separation_surplus() -> None:
    """Test Section 62 slot alignment separation surplus and inertia neutrality."""
    # Heavy Receiver (220 lbs, 6'3") in the Slot
    heavy_slot = compute_slot_alignment_separation_surplus(
        lined_up_position="SLOT_WR",
        combine_weight_lbs=220.0,
        combine_height_in=75.0,
    )
    assert heavy_slot["is_slot_alignment"] == 1.0
    assert heavy_slot["is_outside_alignment"] == 0.0
    assert heavy_slot["predicted_separation_yds"] > 3.80
    assert heavy_slot["predicted_completion_rate"] > 0.65
    assert heavy_slot["outside_inertia_penalty"] == 0.0

    # Same Heavy Receiver Outside on Boundary
    heavy_outside = compute_slot_alignment_separation_surplus(
        lined_up_position="WR",
        combine_weight_lbs=220.0,
        combine_height_in=75.0,
    )
    assert heavy_outside["is_slot_alignment"] == 0.0
    assert heavy_outside["is_outside_alignment"] == 1.0
    assert heavy_outside["outside_inertia_penalty"] > 0.15
    assert heavy_outside["predicted_separation_yds"] < 3.35
    assert heavy_outside["predicted_completion_rate"] < 0.58


def test_formation_getoff_spectrum_dynamics() -> None:
    """Test Section 63 formation rusher get-off hesitation spectrum."""
    # I-Formation heavy downhill run look
    iform = compute_formation_getoff_spectrum_dynamics(
        offense_formation="I_FORM",
        baseline_get_off_s=0.9435,
    )
    assert iform["formation_hesitation_delta_s"] > 0.20
    assert iform["predicted_get_off_s"] > 1.15
    assert iform["is_run_read_hesitation"] == 1.0
    assert iform["is_pure_pass_launch"] == 0.0

    # Empty 5-wide pure pass look
    empty = compute_formation_getoff_spectrum_dynamics(
        offense_formation="EMPTY",
        baseline_get_off_s=0.9435,
    )
    assert empty["formation_hesitation_delta_s"] < -0.04
    assert empty["predicted_get_off_s"] < 0.90
    assert empty["is_pure_pass_launch"] == 1.0
    assert empty["is_run_read_hesitation"] == 0.0


def test_empty_protection_breakdown_hazard() -> None:
    """Test Section 64 empty formation pass protection breakdown hazard."""
    # Standard Shotgun protection with RB in backfield
    shotgun = compute_empty_protection_breakdown_hazard(
        is_empty_formation=False,
        is_blitzing=False,
    )
    assert shotgun["empty_vulnerability_factor"] == 1.000
    assert shotgun["predicted_quick_pressure_rate"] < 0.04
    assert shotgun["predicted_rusher_sack_rate"] < 0.02

    # Empty formation with zero backfield protection
    empty = compute_empty_protection_breakdown_hazard(
        is_empty_formation=True,
        is_blitzing=True,
    )
    assert empty["empty_vulnerability_factor"] > 1.40
    assert empty["predicted_quick_pressure_rate"] > 0.07
    assert empty["predicted_rusher_sack_rate"] > 0.03
    assert empty["peak_pressure_probability"] > 0.40


def test_tight_end_inline_vs_detached_dynamics() -> None:
    """Test Section 65 Tight End attached Y vs detached F alignment dynamics."""
    # Attached in-line TE against man coverage
    attached_man = compute_tight_end_inline_vs_detached_dynamics(
        is_attached=True,
        is_man_coverage=True,
        three_cone_seconds=6.95,
    )
    assert attached_man["is_attached"] == 1.0
    assert attached_man["predicted_separation_yds"] > 3.40
    assert attached_man["predicted_completion_rate"] > 0.65
    assert attached_man["expected_epa"] > 0.20
    assert attached_man["expected_yac_yds"] > 5.50

    # Detached flex TE against man coverage
    detached_man = compute_tight_end_inline_vs_detached_dynamics(
        is_attached=False,
        is_man_coverage=True,
        short_shuttle_seconds=4.45,
    )
    assert detached_man["is_attached"] == 0.0
    assert detached_man["predicted_separation_yds"] < 3.00
    assert detached_man["predicted_completion_rate"] < 0.60
    assert detached_man["expected_epa"] < 0.10
    assert (
        attached_man["predicted_separation_yds"]
        > detached_man["predicted_separation_yds"]
    )


def test_defensive_back_cushion_elasticity() -> None:
    """Test Section 66 defensive back down-and-distance cushion elasticity and boundary length-leverage."""
    # 3rd & Short boundary corner
    third_short = compute_defensive_back_cushion_elasticity(
        down=3,
        yards_to_go=2.0,
        is_boundary_corner=True,
        combine_height_in=74.0,  # 6'2" long corner
        combine_weight_lbs=205.0,
    )
    assert third_short["expected_cushion_yds"] < 5.50
    assert third_short["press_probability"] > 0.30
    assert third_short["length_leverage_index"] > 1.50

    # 3rd & Long boundary corner
    third_long = compute_defensive_back_cushion_elasticity(
        down=3,
        yards_to_go=9.0,
        is_boundary_corner=True,
        combine_height_in=70.0,
        combine_weight_lbs=185.0,
    )
    assert third_long["expected_cushion_yds"] > 6.50
    assert third_long["press_probability"] < 0.25
    assert third_long["expected_cushion_yds"] > third_short["expected_cushion_yds"]

    # Slot corner taller height penalty (larger cushion needed)
    slot_tall = compute_defensive_back_cushion_elasticity(
        down=1,
        is_boundary_corner=False,
        combine_height_in=74.0,
    )
    assert slot_tall["length_leverage_index"] < 0.0
    assert slot_tall["expected_cushion_yds"] > 6.50


def test_presnap_motion_vector_surplus() -> None:
    """Test Section 67 pre-snap motion vector separation surplus and cushion displacement."""
    # Receiver in pre-snap motion
    motion_target = compute_presnap_motion_vector_surplus(
        in_motion_at_snap=True,
        combine_forty_seconds=4.55,
        is_man_coverage=False,
    )
    assert motion_target["in_motion_at_snap"] == 1.0
    assert motion_target["expected_separation_yds"] > 5.00
    assert motion_target["expected_cushion_yds"] > 8.00
    assert motion_target["predicted_completion_rate"] > 0.75
    assert motion_target["expected_yac_yds"] > 6.50
    assert motion_target["motion_separation_surplus"] > 1.40

    # Static receiver
    static_target = compute_presnap_motion_vector_surplus(
        in_motion_at_snap=False,
        combine_forty_seconds=4.55,
        is_man_coverage=False,
    )
    assert static_target["in_motion_at_snap"] == 0.0
    assert static_target["expected_separation_yds"] < 4.00
    assert static_target["expected_cushion_yds"] < 6.50
    assert static_target["predicted_completion_rate"] < 0.65
    assert (
        motion_target["expected_separation_yds"]
        > static_target["expected_separation_yds"]
    )


def test_interior_ol_lever_paradox_metrics() -> None:
    """Test Section 68 Interior OL lever-arm paradox and anchor degradation."""
    # Compact Center with short arms
    compact_c = compute_interior_ol_lever_paradox_metrics(
        lined_up_position="C",
        arm_length_in=31.5,
        combine_height_in=73.5,
        wing_span_in=76.0,
    )
    assert compact_c["is_center"] == 1.0
    assert compact_c["lever_arm_penalty_index"] < 0.0
    assert compact_c["predicted_pressure_rate_allowed"] < 0.05
    assert compact_c["predicted_sack_rate_allowed"] < 0.005

    # Overextended Center with long arms
    long_c = compute_interior_ol_lever_paradox_metrics(
        lined_up_position="C",
        arm_length_in=35.0,
        combine_height_in=76.5,
        wing_span_in=84.0,
    )
    assert long_c["lever_arm_penalty_index"] > 2.0
    assert (
        long_c["predicted_sack_rate_allowed"] > compact_c["predicted_sack_rate_allowed"]
    )
    assert (
        long_c["predicted_pressure_rate_allowed"]
        > compact_c["predicted_pressure_rate_allowed"]
    )

    # Guard baseline vs Center baseline
    standard_g = compute_interior_ol_lever_paradox_metrics(
        lined_up_position="G",
        arm_length_in=33.0,
    )
    assert standard_g["is_center"] == 0.0
    assert standard_g["predicted_pressure_rate_allowed"] > 0.070


def test_overload_dual_rusher_breakdown_hazard() -> None:
    """Test Section 69 overload dual-rusher protection breakdown hazard."""
    # 0 rushers
    zero_r = compute_overload_dual_rusher_breakdown_hazard(pass_rushers_encountered=0)
    assert zero_r["predicted_pressure_rate_allowed"] == 0.0
    assert zero_r["overload_hazard_multiplier"] == 0.0

    # 1 rusher baseline
    one_r = compute_overload_dual_rusher_breakdown_hazard(pass_rushers_encountered=1)
    assert one_r["overload_hazard_multiplier"] == 1.000
    assert one_r["predicted_pressure_rate_allowed"] < 0.10
    assert one_r["predicted_sack_rate_allowed"] < 0.012

    # 2 rushers overload crisis
    two_r = compute_overload_dual_rusher_breakdown_hazard(
        pass_rushers_encountered=2,
        is_guard=True,
    )
    assert two_r["overload_hazard_multiplier"] > 2.40
    assert two_r["predicted_pressure_rate_allowed"] > 0.20
    assert two_r["predicted_sack_rate_allowed"] > 0.025
    assert two_r["peak_pressure_probability"] > 0.40
    assert (
        two_r["predicted_pressure_rate_allowed"]
        > one_r["predicted_pressure_rate_allowed"] * 2.0
    )


def test_defensive_tackle_penetration_archetype() -> None:
    """Test Section 70 DT penetration archetype and concentric lower-body burst."""
    # Explosive 3-Technique penetrator (Aaron Donald profile: explosive broad jump, sub-1.70s split)
    pen_3tech = compute_defensive_tackle_penetration_archetype(
        combine_weight_lbs=295.0,
        broad_jump_in=118.0,
        ten_yd_split_s=1.65,
    )
    assert pen_3tech["is_3tech_penetrator"] == 1.0
    assert pen_3tech["penetration_score"] > 2.0
    assert pen_3tech["predicted_get_off_s"] < 0.95
    assert pen_3tech["predicted_sack_rate"] > 0.010

    # Massive Nose Tackle anchor (Vince Wilfork / Jordan Davis profile: 335 lbs, heavy inertia)
    nose_anchor = compute_defensive_tackle_penetration_archetype(
        combine_weight_lbs=340.0,
        broad_jump_in=100.0,
        ten_yd_split_s=1.85,
    )
    assert nose_anchor["is_3tech_penetrator"] == 0.0
    assert nose_anchor["penetration_score"] < 0.0
    assert nose_anchor["predicted_get_off_s"] > 1.05
    assert nose_anchor["predicted_sack_rate"] < pen_3tech["predicted_sack_rate"]


def test_wide_receiver_press_suffocation_metrics() -> None:
    """Test Section 71 WR press coverage suffocation and separation collapse."""
    # Receiver facing Press Coverage (<=3 yds cushion)
    press_wr = compute_wide_receiver_press_suffocation_metrics(
        cushion_yds=2.0,
        arm_length_in=33.5,  # long arm pad-level disadvantage
        combine_forty_s=4.50,
    )
    assert press_wr["is_press_coverage"] == 1.0
    assert press_wr["predicted_separation_yds"] < 2.30
    assert press_wr["predicted_completion_rate"] < 0.52
    assert press_wr["press_suffocation_penalty_yds"] > 1.50
    assert press_wr["expected_yac_yds"] < 4.00

    # Receiver facing Off Coverage (>3 yds cushion)
    off_wr = compute_wide_receiver_press_suffocation_metrics(
        cushion_yds=6.5,
        arm_length_in=31.5,
        combine_forty_s=4.42,
    )
    assert off_wr["is_press_coverage"] == 0.0
    assert off_wr["predicted_separation_yds"] > 3.80
    assert off_wr["predicted_completion_rate"] > 0.60
    assert off_wr["press_suffocation_penalty_yds"] == 0.0
    assert (
        off_wr["predicted_separation_yds"] > press_wr["predicted_separation_yds"] + 1.50
    )


def test_in_vs_out_breaking_route_dynamics() -> None:
    """Test Section 72 In-Breaking vs Out-Breaking route mechanics and EPA surplus."""
    # In-Breaking route (e.g. CROSS / SLANT)
    cross_route = compute_in_vs_out_breaking_route_dynamics(
        route_ran="CROSS",
        combine_weight_lbs=205.0,
    )
    assert cross_route["is_in_breaking"] == 1.0
    assert cross_route["is_out_breaking"] == 0.0
    assert cross_route["expected_epa"] > 0.20
    assert cross_route["expected_yac_yds"] > 4.90
    assert cross_route["in_breaking_momentum_surplus_yds"] > 1.20

    # Out-Breaking route (e.g. OUT / FLAT)
    out_route = compute_in_vs_out_breaking_route_dynamics(
        route_ran="OUT",
        combine_weight_lbs=215.0,
    )
    assert out_route["is_in_breaking"] == 0.0
    assert out_route["is_out_breaking"] == 1.0
    assert out_route["predicted_separation_yds"] > 4.00
    assert out_route["expected_epa"] < cross_route["expected_epa"]
    assert cross_route["expected_epa"] > out_route["expected_epa"] * 1.5


def test_single_high_safety_range_metrics() -> None:
    """Test Section 73 Single-High safety range and 10-yard split closing velocity."""
    # Fast Single-High safety (sub-1.50s split: Kyle Hamilton / Jessie Bates profile)
    fast_safety = compute_single_high_safety_range_metrics(
        is_single_high=True,
        ten_yd_split_s=1.48,
        pass_length_yds=25.0,
    )
    assert fast_safety["is_single_high"] == 1.0
    assert fast_safety["lateral_closing_range_index"] > 1.50
    assert fast_safety["predicted_deep_comp_rate_allowed"] < 0.35
    assert fast_safety["single_high_vulnerability_score"] < 0.80

    # Slower Single-High safety (1.56s split)
    slow_safety = compute_single_high_safety_range_metrics(
        is_single_high=True,
        ten_yd_split_s=1.56,
        pass_length_yds=25.0,
    )
    assert (
        slow_safety["lateral_closing_range_index"]
        < fast_safety["lateral_closing_range_index"]
    )
    assert (
        slow_safety["predicted_deep_comp_rate_allowed"]
        > fast_safety["predicted_deep_comp_rate_allowed"]
    )


def test_red_zone_spatial_compression_metrics() -> None:
    """Test Section 74 Red Zone and Goal-to-Go spatial boundary compression and TE box-out advantage."""
    # Goal-to-Go Tight End (yardline 5, TE)
    g2g_te = compute_red_zone_spatial_compression_metrics(
        yardline_dist_to_goal=5.0,
        is_tight_end=True,
    )
    assert g2g_te["is_goal_to_go"] == 1.0
    assert g2g_te["is_red_zone"] == 1.0
    assert g2g_te["predicted_separation_yds"] > 3.30
    assert g2g_te["predicted_completion_rate"] > 0.70
    assert g2g_te["predicted_cushion_yds"] < 5.10
    assert g2g_te["space_compression_ratio"] > 0.30

    # Goal-to-Go Wide Receiver (yardline 5, WR)
    g2g_wr = compute_red_zone_spatial_compression_metrics(
        yardline_dist_to_goal=5.0,
        is_tight_end=False,
    )
    assert g2g_wr["is_goal_to_go"] == 1.0
    assert g2g_wr["predicted_separation_yds"] < 2.50
    assert g2g_wr["predicted_completion_rate"] < 0.55
    # TE retains superior separation and completion in Goal-to-Go
    assert (
        g2g_te["predicted_separation_yds"] > g2g_wr["predicted_separation_yds"] + 0.90
    )
    assert (
        g2g_te["predicted_completion_rate"] > g2g_wr["predicted_completion_rate"] + 0.15
    )

    # Open Field target (yardline 45)
    open_wr = compute_red_zone_spatial_compression_metrics(
        yardline_dist_to_goal=45.0,
        is_tight_end=False,
    )
    assert open_wr["is_goal_to_go"] == 0.0
    assert open_wr["is_red_zone"] == 0.0
    assert open_wr["predicted_separation_yds"] > 3.50
    assert open_wr["predicted_cushion_yds"] > 6.00


def test_cover_2_man_speed_rush_metrics() -> None:
    """Test Section 75 Cover-2-Man speed-rush get-off unleash and OL protection breakdown."""
    # Cover-2-Man rusher with explosive broad jump (126 in)
    c2m_elite = compute_cover_2_man_speed_rush_metrics(
        team_coverage_type="COVER_2_MAN",
        rusher_broad_jump_in=126.0,
        rusher_weight_lbs=250.0,
    )
    assert c2m_elite["is_cover_2_man"] == 1.0
    assert c2m_elite["predicted_rusher_get_off_s"] < 0.86
    assert c2m_elite["predicted_rusher_sack_rate"] > 0.025
    assert c2m_elite["predicted_ol_pressure_rate"] > 0.12
    assert c2m_elite["speed_rush_unleash_factor"] > 1.50
    assert c2m_elite["expected_dropback_duration_s"] > 3.10

    # Cover-3-Zone baseline
    c3z_base = compute_cover_2_man_speed_rush_metrics(
        team_coverage_type="COVER_3_ZONE",
    )
    assert c3z_base["is_cover_2_man"] == 0.0
    assert c3z_base["predicted_rusher_get_off_s"] > 1.00
    assert c3z_base["predicted_rusher_sack_rate"] < 0.010
    assert c3z_base["predicted_ol_pressure_rate"] < 0.08
    assert (
        c2m_elite["predicted_rusher_sack_rate"]
        > c3z_base["predicted_rusher_sack_rate"] * 2.5
    )


def test_money_down_distance_regime_metrics() -> None:
    """Test Section 76 Money-Down distance regimes and conversion route architecture."""
    # 3rd & Short (2 yards to go, SCREEN route)
    short_screen = compute_money_down_distance_regime_metrics(
        yards_to_go=2.0,
        route_ran="SCREEN",
    )
    assert short_screen["is_third_down_short"] == 1.0
    assert short_screen["is_third_down_long"] == 0.0
    assert short_screen["predicted_conversion_probability"] > 0.70
    assert short_screen["predicted_money_down_epa"] > 0.40
    assert short_screen["expected_presnap_cushion_yds"] < 5.30

    # 3rd & Long (9 yards to go, POST vs SCREEN)
    long_post = compute_money_down_distance_regime_metrics(
        yards_to_go=9.0,
        route_ran="POST",
    )
    long_screen = compute_money_down_distance_regime_metrics(
        yards_to_go=9.0,
        route_ran="SCREEN",
    )
    assert long_post["is_third_down_long"] == 1.0
    assert long_post["expected_presnap_cushion_yds"] > 6.00
    assert long_post["predicted_conversion_probability"] > 0.40
    assert long_post["predicted_money_down_epa"] > 0.60
    # POST vastly outperforms SCREEN on 3rd & Long
    assert (
        long_post["predicted_conversion_probability"]
        > long_screen["predicted_conversion_probability"] * 2.0
    )
    assert long_screen["predicted_money_down_epa"] < 0.0


def test_unblocked_rusher_timing_rupture_metrics() -> None:
    """Test Section 77 Unblocked free-rusher disruption and passing efficiency collapse."""
    # Unblocked free rusher penetration
    unb_play = compute_unblocked_rusher_timing_rupture_metrics(
        has_unblocked_rusher=True,
    )
    assert unb_play["has_unblocked_rusher"] == 1.0
    assert unb_play["predicted_completion_rate"] < 0.45
    assert unb_play["predicted_play_epa"] < -0.40
    assert unb_play["unblocked_epa_penalty"] > 0.35
    assert unb_play["timing_rupture_severity_tier"] == 3.0

    # Clean blocked protection
    blocked_play = compute_unblocked_rusher_timing_rupture_metrics(
        has_unblocked_rusher=False,
    )
    assert blocked_play["has_unblocked_rusher"] == 0.0
    assert blocked_play["predicted_completion_rate"] > 0.55
    assert blocked_play["predicted_play_epa"] > -0.05
    assert blocked_play["unblocked_epa_penalty"] == 0.0
    assert (
        blocked_play["predicted_completion_rate"]
        > unb_play["predicted_completion_rate"] + 0.14
    )


def test_trips_3x1_alignment_asymmetry_metrics() -> None:
    """Test Section 78 Trips 3x1 slot separation surplus vs outside WR compression."""
    # Frontside Slot WR in 3x1 Trips
    slot_target = compute_trips_3x1_alignment_asymmetry_metrics(
        is_slot_wr=True,
        is_trips_3x1=True,
    )
    assert slot_target["is_trips_3x1"] == 1.0
    assert slot_target["is_slot_wr"] == 1.0
    assert slot_target["predicted_separation_yds"] > 3.90
    assert slot_target["predicted_completion_rate"] > 0.64
    assert slot_target["slot_release_surplus_yds"] > 0.50

    # Outside WR in 3x1 Trips
    outside_target = compute_trips_3x1_alignment_asymmetry_metrics(
        is_slot_wr=False,
        is_trips_3x1=True,
    )
    assert outside_target["is_trips_3x1"] == 1.0
    assert outside_target["is_slot_wr"] == 0.0
    assert outside_target["predicted_separation_yds"] < 3.40
    assert outside_target["predicted_completion_rate"] < 0.58
    assert (
        slot_target["predicted_separation_yds"]
        > outside_target["predicted_separation_yds"] + 0.50
    )
    assert (
        slot_target["predicted_completion_rate"]
        > outside_target["predicted_completion_rate"] + 0.06
    )


def test_compressed_2x1_run_key_inertia_metrics() -> None:
    """Test Section 79 Compressed 2x1 run-key hesitation latency and pocket longevity."""
    # Compressed 2x1 heavy personnel alignment
    c2x1 = compute_compressed_2x1_run_key_inertia_metrics(
        receiver_alignment="2X1",
    )
    assert c2x1["is_compressed_2x1"] == 1.0
    assert c2x1["predicted_rusher_get_off_s"] > 1.10
    assert c2x1["run_key_hesitation_delay_s"] > 0.10
    assert c2x1["predicted_rusher_sack_rate"] < 0.007
    assert c2x1["expected_dropback_duration_s"] > 3.20

    # 3x2 Empty personnel alignment
    empty = compute_compressed_2x1_run_key_inertia_metrics(
        receiver_alignment="3X2",
    )
    assert empty["is_compressed_2x1"] == 0.0
    assert empty["is_empty_3x2"] == 1.0
    assert empty["predicted_rusher_get_off_s"] < 0.90
    assert (
        empty["predicted_rusher_sack_rate"] > c2x1["predicted_rusher_sack_rate"] * 2.0
    )


def test_pass_length_depth_spectrum_metrics() -> None:
    """Test Section 80 Pass length depth spectrum and deep contested leverage mechanics."""
    # Short pass (<= 5 yds)
    short = compute_pass_length_depth_spectrum_metrics(pass_length_yds=3.5)
    assert short["is_short_pass"] == 1.0
    assert short["is_deep_pass"] == 0.0
    assert short["predicted_completion_rate"] > 0.75
    assert short["predicted_target_separation_yds"] > 4.5
    assert short["expected_yac_yds"] > 5.0

    # Intermediate pass (10 yds)
    inter = compute_pass_length_depth_spectrum_metrics(pass_length_yds=10.0)
    assert inter["is_intermediate_pass"] == 1.0
    assert inter["predicted_completion_rate"] > 0.58
    assert inter["expected_epa"] > 0.20

    # Deep pass (> 15 yds) - baseline
    deep_base = compute_pass_length_depth_spectrum_metrics(pass_length_yds=22.0)
    assert deep_base["is_deep_pass"] == 1.0
    assert deep_base["predicted_completion_rate"] < 0.45
    assert deep_base["predicted_target_separation_yds"] < 2.80

    # Deep pass with heavy frame receiver leverage (weight 225 lbs, height 76 in)
    deep_heavy = compute_pass_length_depth_spectrum_metrics(
        pass_length_yds=22.0,
        receiver_weight_lbs=225.0,
        receiver_height_in=76.0,
    )
    assert deep_heavy["deep_boxout_leverage_bonus"] > 0.05
    assert (
        deep_heavy["predicted_completion_rate"] > deep_base["predicted_completion_rate"]
    )


def test_trailing_game_script_rush_acceleration_metrics() -> None:
    """Test Section 81 Second-half trailing game script and rusher pin-ears acceleration."""
    # Trailing by > 8 points in second half
    trailing = compute_trailing_game_script_rush_acceleration_metrics(
        score_differential=-10.0,
        is_second_half=True,
    )
    assert trailing["is_trailing_game_script"] == 1.0
    assert trailing["predicted_rusher_get_off_s"] < 0.98
    assert trailing["predicted_rusher_sack_rate"] > 0.012
    assert trailing["predicted_ol_pressure_rate"] > 0.090
    assert trailing["trailing_script_penalty_factor"] > 1.20

    # Protecting a lead (> 8 points) in second half
    leading = compute_trailing_game_script_rush_acceleration_metrics(
        score_differential=10.0,
        is_second_half=True,
    )
    assert leading["is_trailing_game_script"] == 0.0
    assert leading["predicted_rusher_get_off_s"] > 1.05
    assert leading["predicted_rusher_sack_rate"] < 0.008
    assert leading["predicted_ol_pressure_rate"] < 0.080
    # Rushers are significantly faster when trailing offense must pass
    assert (
        trailing["predicted_rusher_get_off_s"]
        < leading["predicted_rusher_get_off_s"] - 0.10
    )
    assert (
        trailing["predicted_rusher_sack_rate"]
        > leading["predicted_rusher_sack_rate"] * 1.5
    )


def test_dt_scramble_containment_burst_metrics() -> None:
    """Test Section 82 Interior DT scramble containment and pursuit burst sack conversion."""
    # High burst explosive DT (Broad Jump 118 in, 40 time 4.90s, Weight 295 lbs)
    explosive_dt = compute_dt_scramble_containment_burst_metrics(
        broad_jump_in=118.0,
        forty_s=4.90,
        combine_weight_lbs=295.0,
    )
    assert explosive_dt["pursuit_burst_score"] > 0.50
    assert explosive_dt["predicted_sack_conversion_rate"] > 0.54
    assert explosive_dt["predicted_ind_sack_probability"] > 0.08
    assert explosive_dt["containment_efficiency_index"] > 1.05

    # Heavy plodding DT (Broad Jump 102 in, 40 time 5.25s, Weight 330 lbs)
    heavy_dt = compute_dt_scramble_containment_burst_metrics(
        broad_jump_in=102.0,
        forty_s=5.25,
        combine_weight_lbs=330.0,
    )
    assert heavy_dt["pursuit_burst_score"] < -0.50
    assert heavy_dt["predicted_sack_conversion_rate"] < 0.52
    assert heavy_dt["predicted_ind_sack_probability"] < 0.06
    # Explosive DT achieves significantly higher sack conversion than plodding DT
    assert (
        explosive_dt["predicted_ind_sack_probability"]
        > heavy_dt["predicted_ind_sack_probability"] * 1.5
    )


def test_scramble_drill_breakdown_metrics() -> None:
    """Test Section 83 Scramble drill route breakdown and heavy-frame contested leverage."""
    # Quick rhythm throw (1.9s)
    quick = compute_scramble_drill_breakdown_metrics(dropback_duration_s=1.90)
    assert quick["is_quick_rhythm"] == 1.0
    assert quick["is_extended_scramble_drill"] == 0.0
    assert quick["predicted_completion_rate"] > 0.70
    assert quick["predicted_target_separation_yds"] > 4.00

    # Intermediate in-pocket throw (2.7s)
    inter = compute_scramble_drill_breakdown_metrics(dropback_duration_s=2.70)
    assert inter["is_intermediate_pocket"] == 1.0
    assert inter["predicted_completion_rate"] < 0.65
    assert inter["predicted_target_separation_yds"] < 3.50

    # Extended scramble drill (> 3.2s) - baseline receiver
    scramble_base = compute_scramble_drill_breakdown_metrics(dropback_duration_s=3.80)
    assert scramble_base["is_extended_scramble_drill"] == 1.0
    # Separation rebounds in scramble drill vs intermediate
    assert (
        scramble_base["predicted_target_separation_yds"]
        > inter["predicted_target_separation_yds"]
    )
    assert scramble_base["expected_epa"] > 0.20
    assert scramble_base["expected_pass_length_yds"] > 14.0

    # Extended scramble drill - heavy frame receiver (225 lbs)
    scramble_heavy = compute_scramble_drill_breakdown_metrics(
        dropback_duration_s=3.80,
        receiver_weight_lbs=225.0,
    )
    assert scramble_heavy["scramble_frame_leverage_bonus"] > 0.04
    assert (
        scramble_heavy["predicted_completion_rate"]
        > scramble_base["predicted_completion_rate"]
    )
    assert (
        scramble_heavy["predicted_target_separation_yds"]
        > scramble_base["predicted_target_separation_yds"]
    )


def test_defensive_back_blitz_timing_metrics() -> None:
    """Test Section 84 Defensive back blitz timing rupture and 10-yard split acceleration."""
    # DB blitz with elite 10-yard split burst (1.46s)
    elite_blitzer = compute_defensive_back_blitz_timing_metrics(
        ten_yd_split_s=1.46,
        is_db_blitz=True,
    )
    assert elite_blitzer["is_defensive_back_blitz"] == 1.0
    assert elite_blitzer["predicted_blitz_sack_rate"] > 0.09
    assert elite_blitzer["predicted_completion_rate_allowed"] < 0.53
    assert elite_blitzer["expected_passing_epa_allowed"] < -0.10
    assert elite_blitzer["ten_yd_split_burst_bonus"] > 0.015

    # Standard non-blitz pass coverage
    std_coverage = compute_defensive_back_blitz_timing_metrics(
        is_db_blitz=False,
    )
    assert std_coverage["is_defensive_back_blitz"] == 0.0
    assert std_coverage["predicted_blitz_sack_rate"] < 0.070
    assert std_coverage["predicted_completion_rate_allowed"] > 0.54
    # DB blitz generates significantly higher sack rate than standard coverage
    assert (
        elite_blitzer["predicted_blitz_sack_rate"]
        > std_coverage["predicted_blitz_sack_rate"] * 1.3
    )


def test_under_center_play_action_decoupling_metrics() -> None:
    """Test Section 85 Under-center play-action decoupling and second-level separation surplus."""
    # Under-center Singleback alignment
    uc = compute_under_center_play_action_decoupling_metrics(
        offense_formation="SINGLEBACK",
    )
    assert uc["is_under_center"] == 1.0
    assert uc["predicted_target_separation_yds"] > 3.90
    assert uc["expected_target_epa"] > 0.25
    assert uc["expected_dropback_duration_s"] > 3.30
    assert uc["second_level_separation_surplus_yds"] > 0.30

    # Shotgun alignment
    shotgun = compute_under_center_play_action_decoupling_metrics(
        offense_formation="SHOTGUN",
    )
    assert shotgun["is_under_center"] == 0.0
    assert shotgun["predicted_target_separation_yds"] < 3.70
    assert shotgun["expected_dropback_duration_s"] < 3.00
    assert shotgun["second_level_separation_surplus_yds"] == 0.0
    # Under-center delivers higher target separation and EPA than shotgun
    assert (
        uc["predicted_target_separation_yds"]
        > shotgun["predicted_target_separation_yds"] + 0.30
    )
    assert uc["expected_target_epa"] > shotgun["expected_target_epa"] * 2.0


def test_disguised_coverage_rotation_metrics() -> None:
    """Test Section 86 Disguised coverage rotation decision latency and sack rate surge."""
    # Disguised coverage rotation play
    disguised = compute_disguised_coverage_rotation_metrics(
        is_disguised_coverage=True,
    )
    assert disguised["is_disguised_coverage"] == 1.0
    assert disguised["expected_dropback_duration_s"] > 3.05
    assert disguised["predicted_defensive_sack_rate"] > 0.070
    assert disguised["decision_hesitation_delay_s"] > 0.07
    assert disguised["expected_target_pass_length_yds"] > 10.5

    # Static coverage play
    static = compute_disguised_coverage_rotation_metrics(
        is_disguised_coverage=False,
    )
    assert static["is_disguised_coverage"] == 0.0
    assert static["expected_dropback_duration_s"] < 3.02
    assert static["predicted_defensive_sack_rate"] < 0.065
    assert static["decision_hesitation_delay_s"] == 0.0
    # Disguised coverage increases sack rate by > 12% relative
    assert (
        disguised["predicted_defensive_sack_rate"]
        > static["predicted_defensive_sack_rate"] * 1.10
    )


def test_fourth_quarter_ol_fatigue_decay_metrics() -> None:
    """Test Section 87 Fourth-quarter offensive line fatigue decay and mass/lever vulnerability."""
    # Q1 fresh baseline OL
    q1 = compute_fourth_quarter_ol_fatigue_decay_metrics(quarter=1)
    assert q1["is_fourth_quarter"] == 0.0
    assert q1["predicted_pressure_allowed_rate"] < 0.075
    assert q1["predicted_peak_pressure_probability"] < 0.27

    # Q4 fatigued baseline OL
    q4_base = compute_fourth_quarter_ol_fatigue_decay_metrics(quarter=4)
    assert q4_base["is_fourth_quarter"] == 1.0
    assert q4_base["predicted_pressure_allowed_rate"] > 0.080
    assert q4_base["predicted_peak_pressure_probability"] > 0.28
    assert (
        q4_base["predicted_pressure_allowed_rate"]
        > q1["predicted_pressure_allowed_rate"]
    )

    # Q4 fatigued heavy lineman with long levers (335 lbs, 35.5 in arm length)
    q4_heavy = compute_fourth_quarter_ol_fatigue_decay_metrics(
        quarter=4,
        combine_weight_lbs=335.0,
        arm_length_in=35.5,
    )
    assert q4_heavy["fatigue_decay_penalty"] > 0.04
    assert (
        q4_heavy["predicted_peak_pressure_probability"]
        > q4_base["predicted_peak_pressure_probability"]
    )
    assert (
        q4_heavy["predicted_pressure_allowed_rate"]
        > q4_base["predicted_pressure_allowed_rate"]
    )


def test_pocket_lifespan_efficiency_curve_metrics() -> None:
    """Test Section 88 Pocket lifespan efficiency curve and sweet spot peak."""
    # Underneath quick game (1.8s)
    quick = compute_pocket_lifespan_efficiency_curve_metrics(dropback_duration_s=1.80)
    assert quick["predicted_completion_rate"] > 0.75
    assert quick["predicted_sack_hazard_prob"] < 0.001
    assert quick["pocket_lifespan_efficiency_tier"] == 1.0

    # Sweet spot peak (2.3s)
    sweet = compute_pocket_lifespan_efficiency_curve_metrics(dropback_duration_s=2.30)
    assert sweet["is_sweet_spot_dropback"] == 1.0
    assert sweet["predicted_play_epa"] > 0.10
    assert sweet["pocket_lifespan_efficiency_tier"] == 2.0

    # Intermediate chunk (2.8s)
    inter = compute_pocket_lifespan_efficiency_curve_metrics(dropback_duration_s=2.80)
    assert inter["predicted_play_epa"] > 0.05
    assert inter["pocket_lifespan_efficiency_tier"] == 3.0

    # Breakdown cliff (3.8s)
    cliff = compute_pocket_lifespan_efficiency_curve_metrics(dropback_duration_s=3.80)
    assert cliff["predicted_play_epa"] < 0.0
    assert cliff["predicted_sack_hazard_prob"] > 0.10
    assert cliff["pocket_lifespan_efficiency_tier"] == 5.0

    # Catastrophic collapse (4.5s)
    collapse = compute_pocket_lifespan_efficiency_curve_metrics(
        dropback_duration_s=4.50
    )
    assert collapse["predicted_play_epa"] < -0.40
    assert collapse["predicted_sack_hazard_prob"] > 0.25
    assert collapse["predicted_completion_rate"] < 0.25
    assert collapse["pocket_lifespan_efficiency_tier"] == 6.0


def test_deep_perimeter_cb_stature_leverage_metrics() -> None:
    """Test Section 89 Deep perimeter CB stature and mass leverage suppression."""
    # Tall, physical boundary corner (74 in, 202 lbs) on deep vertical target
    tall_cb = compute_deep_perimeter_cb_stature_leverage_metrics(
        combine_height_in=74.0,
        combine_weight_lbs=202.0,
        is_deep_target=True,
    )
    assert tall_cb["is_deep_vertical_target"] == 1.0
    assert tall_cb["stature_mass_suppression_bonus"] > 0.05
    assert tall_cb["predicted_completion_rate_allowed"] < 0.38
    assert tall_cb["catch_point_boxout_resistance_index"] > 1.10

    # Undersized track-speed corner (69 in, 178 lbs) on deep vertical target
    undersized_cb = compute_deep_perimeter_cb_stature_leverage_metrics(
        combine_height_in=69.0,
        combine_weight_lbs=178.0,
        is_deep_target=True,
    )
    assert undersized_cb["stature_mass_suppression_bonus"] < -0.05
    assert undersized_cb["predicted_completion_rate_allowed"] > 0.45
    # Tall physical corner achieves significantly lower completion allowed
    assert (
        tall_cb["predicted_completion_rate_allowed"]
        < undersized_cb["predicted_completion_rate_allowed"] - 0.08
    )


def test_cover0_allout_blitz_cb_leverage_metrics() -> None:
    """Test Section 90 Cover 0 all-out blitz CB concentric power leverage."""
    # Explosive CB (128 in broad jump, 39 in vertical) in Cover 0
    elite_cb = compute_cover0_allout_blitz_cb_leverage_metrics(
        broad_jump_in=128.0,
        vertical_in=39.0,
        is_cover_0=True,
    )
    assert elite_cb["is_cover_0_man"] == 1.0
    assert elite_cb["predicted_dropback_duration_s"] < 2.70
    assert elite_cb["concentric_burst_leverage_bonus"] > 0.08
    assert elite_cb["predicted_epa_allowed"] < -0.15
    assert elite_cb["predicted_completion_rate_allowed"] < 0.46

    # Lower-burst CB (116 in broad jump, 33 in vertical) in Cover 0
    low_burst_cb = compute_cover0_allout_blitz_cb_leverage_metrics(
        broad_jump_in=116.0,
        vertical_in=33.0,
        is_cover_0=True,
    )
    assert low_burst_cb["concentric_burst_leverage_bonus"] < -0.08
    assert low_burst_cb["predicted_epa_allowed"] > 0.0
    assert (
        elite_cb["predicted_epa_allowed"] < low_burst_cb["predicted_epa_allowed"] - 0.15
    )


def test_simulated_pressure_edge_coverage_metrics() -> None:
    """Test Section 91 Simulated pressure / fire zone edge dropping dynamics."""
    # Fast 10-yd split edge (1.55s) dropping into zone coverage
    fast_edge = compute_simulated_pressure_edge_coverage_metrics(
        ten_yd_split_s=1.55,
        is_edge_dropping=True,
    )
    assert fast_edge["is_edge_dropping_in_coverage"] == 1.0
    assert fast_edge["predicted_dropback_duration_s"] > 3.10
    assert fast_edge["split_burst_coverage_penalty"] < -0.10
    assert fast_edge["predicted_epa_allowed"] < -0.10

    # Slow 10-yd split edge (1.72s) dropping into zone coverage
    slow_edge = compute_simulated_pressure_edge_coverage_metrics(
        ten_yd_split_s=1.72,
        is_edge_dropping=True,
    )
    assert slow_edge["split_burst_coverage_penalty"] > 0.20
    assert slow_edge["predicted_epa_allowed"] > 0.20
    assert slow_edge["predicted_completion_rate_allowed"] > 0.60
    assert (
        fast_edge["predicted_epa_allowed"] < slow_edge["predicted_epa_allowed"] - 0.25
    )


def test_presnap_motion_coverage_rupture_metrics() -> None:
    """Test Section 92 Pre-snap motion man coverage rupture and separation surplus."""
    # Motion target in Man coverage
    man_motion = compute_presnap_motion_coverage_rupture_metrics(
        is_motion=True,
        is_man_coverage=True,
    )
    assert man_motion["is_in_motion"] == 1.0
    assert man_motion["is_man_coverage"] == 1.0
    assert man_motion["predicted_separation_yds"] > 3.70
    assert man_motion["predicted_completion_rate"] > 0.70
    assert man_motion["man_beater_separation_surplus"] > 1.30
    assert man_motion["predicted_target_epa"] > 0.18

    # Static target in Man coverage
    man_static = compute_presnap_motion_coverage_rupture_metrics(
        is_motion=False,
        is_man_coverage=True,
    )
    assert man_static["man_beater_separation_surplus"] == 0.0
    assert man_static["predicted_separation_yds"] < 2.60
    assert man_static["predicted_completion_rate"] < 0.56
    assert (
        man_motion["predicted_separation_yds"]
        > man_static["predicted_separation_yds"] + 1.25
    )
    assert (
        man_motion["predicted_completion_rate"]
        > man_static["predicted_completion_rate"] + 0.15
    )


def test_heavy_formation_te_efficiency_metrics() -> None:
    """Test Section 93 Heavy formation TE separation and EPA efficiency."""
    # Heavy run personnel (Singleback, Pistol)
    heavy = compute_heavy_formation_te_efficiency_metrics(
        is_heavy_formation=True,
        is_i_formation=False,
    )
    assert heavy["is_heavy_formation"] == 1.0
    assert heavy["predicted_separation_yds"] > 4.40
    assert heavy["predicted_completion_rate"] > 0.75
    assert heavy["epa_surplus_over_shotgun"] > 0.09

    # I-Formation heavy power
    iform = compute_heavy_formation_te_efficiency_metrics(
        is_heavy_formation=True,
        is_i_formation=True,
    )
    assert iform["is_i_formation"] == 1.0
    assert iform["predicted_completion_rate"] > 0.80
    assert iform["predicted_target_epa"] > 0.40
    assert iform["epa_surplus_over_shotgun"] > 0.25

    # Shotgun spread baseline
    shotgun = compute_heavy_formation_te_efficiency_metrics(
        is_heavy_formation=False,
        is_i_formation=False,
    )
    assert shotgun["is_heavy_formation"] == 0.0
    assert shotgun["predicted_separation_yds"] < 4.10
    assert shotgun["predicted_completion_rate"] < 0.74
    assert (
        heavy["predicted_separation_yds"] > shotgun["predicted_separation_yds"] + 0.40
    )
    assert iform["predicted_target_epa"] > shotgun["predicted_target_epa"] + 0.20


def test_two_high_run_defense_concession_metrics() -> None:
    """Test Section 94 Two-high safety light box run concession and TFL rates."""
    two_high = compute_two_high_run_defense_concession_metrics(
        is_two_high_shell=True,
        is_cover_0=False,
    )
    assert two_high["is_two_high_shell"] == 1.0
    assert two_high["predicted_rushing_yards_per_carry"] > 4.70
    assert two_high["light_box_yardage_surplus"] > 0.50
    assert two_high["predicted_tfl_rate"] < 0.09
    assert two_high["predicted_run_success_rate"] > 0.40

    single_high = compute_two_high_run_defense_concession_metrics(
        is_two_high_shell=False,
        is_cover_0=False,
    )
    assert single_high["predicted_rushing_yards_per_carry"] < 4.30
    assert single_high["predicted_tfl_rate"] > 0.11
    assert (
        two_high["predicted_rushing_yards_per_carry"]
        > single_high["predicted_rushing_yards_per_carry"] + 0.45
    )

    cover_0 = compute_two_high_run_defense_concession_metrics(
        is_two_high_shell=False,
        is_cover_0=True,
    )
    assert cover_0["is_cover_0"] == 1.0
    assert cover_0["predicted_rushing_yards_per_carry"] < 2.05
    assert cover_0["predicted_tfl_rate"] > 0.14


def test_deep_pass_coverage_shell_risk_metrics() -> None:
    """Test Section 95 Deep pass hole shot efficiency and two-high turnover risk."""
    deep_two_high = compute_deep_pass_coverage_shell_risk_metrics(
        is_two_high_shell=True,
        is_cover_0=False,
    )
    assert deep_two_high["is_two_high_shell"] == 1.0
    assert deep_two_high["predicted_deep_completion_rate"] > 0.43
    assert deep_two_high["predicted_deep_pass_epa"] > 0.35
    assert deep_two_high["predicted_deep_interception_rate"] > 0.055
    assert deep_two_high["interception_risk_multiplier"] > 1.20

    deep_single_high = compute_deep_pass_coverage_shell_risk_metrics(
        is_two_high_shell=False,
        is_cover_0=False,
    )
    assert deep_single_high["predicted_deep_completion_rate"] < 0.41
    assert deep_single_high["predicted_deep_interception_rate"] < 0.05
    assert deep_single_high["interception_risk_multiplier"] == 1.0
    assert (
        deep_two_high["predicted_deep_pass_epa"]
        > deep_single_high["predicted_deep_pass_epa"] + 0.10
    )


def test_interior_offensive_line_leverage_metrics() -> None:
    """Test Section 96 Interior OL pad-level leverage and wingspan vulnerability."""
    compact_guard = compute_interior_offensive_line_leverage_metrics(
        height_in=74.5,
        wingspan_in=78.0,
        arm_length_in=32.5,
    )
    assert compact_guard["leverage_advantage_flag"] == 1.0
    assert compact_guard["predicted_peak_pressure_prob"] < 0.245
    assert compact_guard["predicted_pressure_rate"] < 0.070

    tall_long_guard = compute_interior_offensive_line_leverage_metrics(
        height_in=77.5,
        wingspan_in=82.0,
        arm_length_in=34.5,
    )
    assert tall_long_guard["leverage_advantage_flag"] == 0.0
    assert tall_long_guard["predicted_peak_pressure_prob"] > 0.260
    assert tall_long_guard["predicted_pressure_rate"] > 0.075
    assert (
        tall_long_guard["predicted_peak_pressure_prob"]
        > compact_guard["predicted_peak_pressure_prob"] + 0.015
    )


def test_interior_defensive_line_burst_penetration_metrics() -> None:
    """Test Section 97 Interior DL lower-body concentric drive and A-gap penetration."""
    explosive_dt = compute_interior_defensive_line_burst_penetration_metrics(
        broad_jump_in=114.0,
        vertical_jump_in=33.0,
        ten_yard_split_s=1.70,
        weight_lbs=308.0,
    )
    assert explosive_dt["elite_penetration_flag"] == 1.0
    assert explosive_dt["predicted_get_off_s"] < 1.05
    assert explosive_dt["predicted_pressure_rate"] > 0.040
    assert explosive_dt["predicted_quick_pressure_rate"] > 0.30

    stiff_dt = compute_interior_defensive_line_burst_penetration_metrics(
        broad_jump_in=102.0,
        vertical_jump_in=27.0,
        ten_yard_split_s=1.82,
        weight_lbs=315.0,
    )
    assert stiff_dt["elite_penetration_flag"] == 0.0
    assert stiff_dt["predicted_get_off_s"] > 1.12
    assert stiff_dt["predicted_pressure_rate"] < 0.030
    assert stiff_dt["predicted_quick_pressure_rate"] < 0.15
    assert (
        explosive_dt["predicted_quick_pressure_rate"]
        > stiff_dt["predicted_quick_pressure_rate"] + 0.15
    )


def test_slot_vs_boundary_cb_anthropometric_divergence_metrics() -> None:
    """Test Section 98 Slot vs Boundary CB anthropometric divergence."""
    heavy_boundary_cb = compute_slot_vs_boundary_cb_anthropometric_divergence_metrics(
        is_slot=False,
        weight_lbs=198.0,
        height_in=73.5,
    )
    assert heavy_boundary_cb["is_slot"] == 0.0
    assert heavy_boundary_cb["anthropometric_fit_rating"] == 1.0
    assert heavy_boundary_cb["predicted_completion_rate_allowed"] < 0.52

    light_boundary_cb = compute_slot_vs_boundary_cb_anthropometric_divergence_metrics(
        is_slot=False,
        weight_lbs=185.0,
        height_in=70.0,
    )
    assert light_boundary_cb["anthropometric_fit_rating"] == 0.0
    assert light_boundary_cb["predicted_completion_rate_allowed"] > 0.54
    assert (
        heavy_boundary_cb["predicted_completion_rate_allowed"]
        < light_boundary_cb["predicted_completion_rate_allowed"]
    )

    heavy_slot_cb = compute_slot_vs_boundary_cb_anthropometric_divergence_metrics(
        is_slot=True,
        weight_lbs=202.0,
        height_in=73.0,
    )
    assert heavy_slot_cb["is_slot"] == 1.0
    assert heavy_slot_cb["anthropometric_fit_rating"] == 0.0
    assert heavy_slot_cb["predicted_completion_rate_allowed"] > 0.535

    agile_slot_cb = compute_slot_vs_boundary_cb_anthropometric_divergence_metrics(
        is_slot=True,
        weight_lbs=190.0,
        height_in=70.5,
    )
    assert agile_slot_cb["anthropometric_fit_rating"] == 1.0
    assert agile_slot_cb["predicted_completion_rate_allowed"] < 0.520
    assert (
        agile_slot_cb["predicted_completion_rate_allowed"]
        < heavy_slot_cb["predicted_completion_rate_allowed"]
    )


def test_fourth_quarter_pass_rush_acceleration_metrics() -> None:
    """Test Section 99 Fourth-quarter pass rush pin-ears acceleration and mass fatigue."""
    q1_rusher = compute_fourth_quarter_pass_rush_acceleration_metrics(
        quarter=1,
        weight_lbs=255.0,
        ten_yard_split_s=1.60,
    )
    assert q1_rusher["quarter"] == 1.0
    assert q1_rusher["predicted_get_off_s"] > 0.970

    q4_speed_rusher = compute_fourth_quarter_pass_rush_acceleration_metrics(
        quarter=4,
        weight_lbs=248.0,
        ten_yard_split_s=1.56,
    )
    assert q4_speed_rusher["quarter"] == 4.0
    assert q4_speed_rusher["late_game_motor_advantage_flag"] == 1.0
    assert q4_speed_rusher["predicted_get_off_s"] < 0.920
    assert q4_speed_rusher["predicted_pressure_rate"] > 0.090
    assert q4_speed_rusher["pin_ears_acceleration_s"] > 0.060

    q4_heavy_rusher = compute_fourth_quarter_pass_rush_acceleration_metrics(
        quarter=4,
        weight_lbs=272.0,
        ten_yard_split_s=1.68,
    )
    assert q4_heavy_rusher["late_game_motor_advantage_flag"] == 0.0
    assert q4_heavy_rusher["predicted_get_off_s"] > 0.980
    assert q4_heavy_rusher["predicted_pressure_rate"] < 0.070
    assert (
        q4_speed_rusher["predicted_pressure_rate"]
        > q4_heavy_rusher["predicted_pressure_rate"] + 0.025
    )


def test_receiver_alignment_structural_density_metrics() -> None:
    """Test Section 100 Receiver alignment structural density and 2x1 spacing efficiency."""
    set_2x1 = compute_receiver_alignment_structural_density_metrics(alignment="2x1")
    assert set_2x1["predicted_dropback_duration_s"] > 3.20
    assert set_2x1["predicted_passing_epa_per_play"] > 0.020
    assert set_2x1["predicted_target_separation_yds"] > 4.00
    assert set_2x1["predicted_target_epa"] > 0.20
    assert set_2x1["spacing_efficiency_index"] == 1.00

    set_2x2 = compute_receiver_alignment_structural_density_metrics(alignment="2x2")
    assert set_2x2["predicted_dropback_duration_s"] < 3.00
    assert set_2x2["predicted_sack_rate"] < 0.060
    assert set_2x2["predicted_target_separation_yds"] < 3.70

    set_3x1 = compute_receiver_alignment_structural_density_metrics(alignment="3x1")
    assert set_3x1["predicted_sack_rate"] > 0.070
    assert set_3x1["predicted_passing_epa_per_play"] < -0.040

    set_3x2 = compute_receiver_alignment_structural_density_metrics(alignment="3x2")
    assert set_3x2["predicted_dropback_duration_s"] < 2.90
    assert set_3x2["predicted_interception_rate"] > 0.022
    assert set_3x2["predicted_target_epa"] < 0.120
    assert (
        set_2x1["predicted_target_separation_yds"]
        > set_3x1["predicted_target_separation_yds"] + 0.35
    )
    assert (
        set_2x1["predicted_passing_epa_per_play"]
        > set_3x2["predicted_passing_epa_per_play"] + 0.070
    )


def test_goal_line_formation_penalty_metrics() -> None:
    """Test Section 101 Goal-line formation mechanics and shotgun penalty."""
    sb_rush = compute_goal_line_formation_penalty_metrics(
        formation="SINGLEBACK", is_pass=False
    )
    assert sb_rush["predicted_epa_per_play"] > 0.070
    assert sb_rush["predicted_success_rate"] > 0.550
    assert sb_rush["predicted_positive_gain_rate"] > 0.600
    assert sb_rush["shotgun_penalty_delta"] == 0.0
    assert sb_rush["is_shotgun_flag"] == 0.0

    shotgun_rush = compute_goal_line_formation_penalty_metrics(
        formation="SHOTGUN", is_pass=False
    )
    assert shotgun_rush["predicted_epa_per_play"] < -0.100
    assert shotgun_rush["predicted_success_rate"] < 0.470
    assert shotgun_rush["predicted_positive_gain_rate"] < 0.430
    assert shotgun_rush["shotgun_penalty_delta"] < -0.200
    assert shotgun_rush["is_shotgun_flag"] == 1.0

    shotgun_pass = compute_goal_line_formation_penalty_metrics(
        formation="SHOTGUN", is_pass=True
    )
    assert (
        shotgun_pass["predicted_epa_per_play"] < shotgun_rush["predicted_epa_per_play"]
    )

    empty_play = compute_goal_line_formation_penalty_metrics(
        formation="EMPTY", is_pass=False
    )
    assert empty_play["predicted_epa_per_play"] < -0.300
    assert empty_play["predicted_success_rate"] < 0.430


def test_safety_spin_down_coverage_risk_metrics() -> None:
    """Test Section 102 Safety box spin-down and deep island concession."""
    deep_safety = compute_safety_spin_down_coverage_risk_metrics(is_spun_down=False)
    assert deep_safety["is_spun_down_flag"] == 0.0
    assert deep_safety["predicted_completion_rate_allowed"] < 0.580
    assert deep_safety["predicted_passing_epa_allowed"] < 0.0
    assert deep_safety["completion_concession_delta"] == 0.0
    assert deep_safety["predicted_sack_rate"] > 0.060

    spun_down_safety = compute_safety_spin_down_coverage_risk_metrics(is_spun_down=True)
    assert spun_down_safety["is_spun_down_flag"] == 1.0
    assert spun_down_safety["predicted_completion_rate_allowed"] > 0.640
    assert spun_down_safety["predicted_passing_epa_allowed"] > 0.010
    assert spun_down_safety["completion_concession_delta"] > 0.070
    assert (
        spun_down_safety["predicted_completion_rate_allowed"]
        > deep_safety["predicted_completion_rate_allowed"] + 0.070
    )
    assert (
        spun_down_safety["predicted_yards_allowed"]
        > deep_safety["predicted_yards_allowed"] + 0.70
    )


def test_slot_blitz_unblocked_pursuit_metrics() -> None:
    """Test Section 103 Slot cornerback nickel blitz latency and unblocked pursuit angle advantage."""
    slot_unb = compute_slot_blitz_unblocked_pursuit_metrics(
        rusher_position="SLOT_CB", is_unblocked=True
    )
    assert slot_unb["is_slot_cb_flag"] == 1.0
    assert slot_unb["is_unblocked_flag"] == 1.0
    assert slot_unb["predicted_time_to_pressure_s"] < 2.20
    assert slot_unb["predicted_dropback_duration_s"] < 3.00
    assert slot_unb["nickel_blitz_latency_advantage_s"] > 0.50
    assert slot_unb["predicted_completion_rate_allowed"] < 0.450

    edge_unb = compute_slot_blitz_unblocked_pursuit_metrics(
        rusher_position="EDGE", is_unblocked=True
    )
    assert edge_unb["is_slot_cb_flag"] == 0.0
    assert edge_unb["predicted_time_to_pressure_s"] > 2.60
    assert edge_unb["predicted_dropback_duration_s"] > 3.50
    assert edge_unb["nickel_blitz_latency_advantage_s"] == 0.0
    assert (
        edge_unb["predicted_time_to_pressure_s"]
        > slot_unb["predicted_time_to_pressure_s"] + 0.50
    )

    blocked = compute_slot_blitz_unblocked_pursuit_metrics(
        rusher_position="SLOT_CB", is_unblocked=False
    )
    assert blocked["is_unblocked_flag"] == 0.0
    assert blocked["predicted_time_to_pressure_s"] > 3.00


def test_cover6_split_field_turnover_trap_metrics() -> None:
    """Test Section 104 Cover 6 hybrid split-field asymmetry and intermediate turnover trap."""
    c6_post = compute_cover6_split_field_turnover_trap_metrics(
        route_type="POST", coverage_type="COVER_6_ZONE"
    )
    assert c6_post["is_cover_6_flag"] == 1.0
    assert c6_post["is_intermediate_breaking_route_flag"] == 1.0
    assert c6_post["predicted_target_interception_rate"] > 0.110
    assert c6_post["turnover_trap_risk_multiplier"] > 4.0

    c6_corner = compute_cover6_split_field_turnover_trap_metrics(
        route_type="CORNER", coverage_type="COVER_6_ZONE"
    )
    assert c6_corner["predicted_target_interception_rate"] > 0.100

    c3_post = compute_cover6_split_field_turnover_trap_metrics(
        route_type="POST", coverage_type="COVER_3_ZONE"
    )
    assert c3_post["is_cover_6_flag"] == 0.0
    assert c3_post["predicted_target_interception_rate"] < 0.060
    assert (
        c6_post["predicted_target_interception_rate"]
        > c3_post["predicted_target_interception_rate"] + 0.050
    )

    c6_flat = compute_cover6_split_field_turnover_trap_metrics(
        route_type="FLAT", coverage_type="COVER_6_ZONE"
    )
    assert c6_flat["predicted_target_interception_rate"] == 0.0
    assert c6_flat["predicted_target_completion_rate"] > 0.800


def test_wide_receiver_boundary_inertia_metrics() -> None:
    """Test Section 105 Perimeter wide receiver inertia tax and slot decoupling."""
    light_outside = compute_wide_receiver_boundary_inertia_metrics(
        alignment="OUTSIDE", weight_lbs=190.0
    )
    assert light_outside["is_slot_alignment_flag"] == 0.0
    assert light_outside["predicted_separation_yds"] > 3.60
    assert light_outside["predicted_yac_yds"] > 4.00
    assert light_outside["boundary_inertia_penalty_yds"] == 0.0

    heavy_outside = compute_wide_receiver_boundary_inertia_metrics(
        alignment="OUTSIDE", weight_lbs=215.0
    )
    assert heavy_outside["predicted_separation_yds"] < 3.20
    assert heavy_outside["predicted_yac_yds"] < 3.70
    assert heavy_outside["boundary_inertia_penalty_yds"] > 0.50
    assert (
        light_outside["predicted_separation_yds"]
        > heavy_outside["predicted_separation_yds"] + 0.45
    )

    heavy_slot = compute_wide_receiver_boundary_inertia_metrics(
        alignment="SLOT", weight_lbs=215.0
    )
    assert heavy_slot["is_slot_alignment_flag"] == 1.0
    assert heavy_slot["boundary_inertia_penalty_yds"] == 0.0
    assert heavy_slot["predicted_separation_yds"] > 3.80
    assert heavy_slot["predicted_yac_yds"] > 4.60


def test_tight_end_inline_momentum_cliff_metrics() -> None:
    """Test Section 106 Tight end in-line play-action momentum and detached YAC cliff."""
    inline_te = compute_tight_end_inline_momentum_cliff_metrics(
        is_inline=True, broad_jump_in=122.0, weight_lbs=254.0
    )
    assert inline_te["is_inline_flag"] == 1.0
    assert inline_te["predicted_yac_yds"] > 6.00
    assert inline_te["predicted_target_epa"] > 0.230
    assert inline_te["detached_yac_cliff_delta_yds"] == 0.0
    assert inline_te["concentric_burst_yac_bonus_yds"] > 0.15

    detached_te = compute_tight_end_inline_momentum_cliff_metrics(
        is_inline=False, broad_jump_in=122.0, weight_lbs=254.0
    )
    assert detached_te["is_inline_flag"] == 0.0
    assert detached_te["predicted_yac_yds"] < 4.50
    assert detached_te["predicted_target_epa"] < 0.100
    assert detached_te["detached_yac_cliff_delta_yds"] < -1.90
    assert inline_te["predicted_yac_yds"] > detached_te["predicted_yac_yds"] + 1.80
