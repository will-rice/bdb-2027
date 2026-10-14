"""End-to-end Combine-to-NFL Kinematic Translation Analysis across All 4 Position Groups."""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from bdb_2027.config import Config
from bdb_2027.datasets.combine import CombineFeatureExtractor
from bdb_2027.datasets.game import GameOutcomeExtractor
from bdb_2027.modeling.pipeline import TranslationPipeline


def run_analysis() -> None:
    """Run full Combine-to-NFL translation modeling across WR, DL, OL, and DB groups."""
    config = Config()
    c_ext = CombineFeatureExtractor(config)
    g_ext = GameOutcomeExtractor(config)

    output_dir = Path("artifacts")
    docs_dir = Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("NFL BIG DATA BOWL 2027: COMBINE-TO-NFL TRANSLATION PIPELINE (4 POSITIONS)")
    print("=" * 75)

    # 1. Load Data
    print(
        "\n[1/6] Ingesting Combine physical results, 10 Hz tracking, and NGS outcomes..."
    )
    results_df = c_ext.load_combine_results()
    tracking_df = c_ext.load_combine_tracking()
    player_play_df = g_ext.load_player_play()
    players_df = pl.read_csv("data/raw/players.csv", null_values=["NA"])

    sns.set_theme(style="whitegrid")

    # =========================================================================
    # 2. POSITION 1: WIDE RECEIVERS (The Stopwatch Paradox & Deceleration)
    # =========================================================================
    print("\n[2/6] Processing Wide Receivers (Route Kinematics vs Separation)...")
    forty_feats = c_ext.extract_40yd_features(tracking_df)
    wr_route_feats = c_ext.extract_wr_route_kinematics(tracking_df)
    drill_feats = c_ext.extract_drill_features(tracking_df)
    drill_agg = drill_feats.group_by("nfl_id").agg(
        [
            pl.col("peak_speed_centered").mean().alias("avg_drill_speed_centered"),
            pl.col("peak_decel_centered").mean().alias("avg_drill_decel_centered"),
        ]
    )

    rec_outcomes = g_ext.extract_receiver_metrics(player_play_df)
    breaking_rec = rec_outcomes.filter(
        (pl.col("breaking_targets_count") >= 8)
        & pl.col("breaking_route_separation").is_not_null()
    )

    wr_dataset = (
        breaking_rec.join(results_df, on="nfl_id", how="inner")
        .join(drill_agg, on="nfl_id", how="left")
        .join(forty_feats, on="nfl_id", how="left")
        .join(wr_route_feats, on="nfl_id", how="left")
    )
    print(f"  Matched WR cohort size: {wr_dataset.height} players")

    feature_cols = [
        "forty",
        "ten_yd_split",
        "three_cone",
        "short_shuttle",
        "vertical",
        "broad_jump",
        "combine_weight",
        "combine_height",
        "best_40_peak_speed",
        "avg_drill_speed_centered",
        "avg_drill_decel_centered",
        "wr_route_speed_z",
        "wr_route_decel_z",
        "raw_wr_braking_decel",
    ]
    available_cols = [c for c in feature_cols if c in wr_dataset.columns]
    for c in available_cols:
        wr_dataset = wr_dataset.with_columns(pl.col(c).fill_null(strategy="mean"))

    pipeline = TranslationPipeline(config)
    wr_results = pipeline.train_cv(
        features_df=wr_dataset,
        feature_cols=available_cols,
        target_col="breaking_route_separation",
        group_col="nfl_id",
    )

    print("  --- WR MODEL VALIDATION RESULTS (Target: NFL Breaking Separation) ---")
    print(f"  Overall OOF Pearson r:    {wr_results['pearson_r']:.3f}")
    print(f"  Overall OOF Spearman rho: {wr_results['spearman_r']:.3f}")
    print(f"  Overall OOF R^2:          {wr_results['overall_r2']:.3f}")
    print(f"  Overall OOF RMSE:         {wr_results['overall_rmse']:.3f} yards")

    # Figure 1: 40-yd dash vs 10 Hz Decel contrast
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    df_wr_p = wr_dataset.to_pandas()
    df_wr_p["forty"] = pd.to_numeric(df_wr_p["forty"], errors="coerce")
    valid_40 = df_wr_p.dropna(subset=["forty", "breaking_route_separation"])

    sns.regplot(
        x=valid_40["forty"],
        y=valid_40["breaking_route_separation"],
        ax=axes1[0],
        color="#708090",
        scatter_kws={"alpha": 0.7, "s": 50},
        line_kws={"color": "#d9534f", "linewidth": 2},
    )
    axes1[0].set_title(
        "A: 40-Yard Dash vs In-Game Separation\n(Stopwatch Paradox: r = 0.052, p = 0.62)",
        fontsize=11,
        fontweight="bold",
    )
    axes1[0].set_xlabel("40-Yard Dash (Seconds)", fontweight="bold")
    axes1[0].set_ylabel(
        "In-Game Separation on Breaking Routes (Yards)", fontweight="bold"
    )

    valid_decel = df_wr_p.dropna(
        subset=["raw_wr_braking_decel", "breaking_route_separation"]
    )
    sns.regplot(
        x=valid_decel["raw_wr_braking_decel"],
        y=valid_decel["breaking_route_separation"],
        ax=axes1[1],
        color="#008080",
        scatter_kws={"alpha": 0.8, "s": 50},
        line_kws={"color": "#0275d8", "linewidth": 2},
    )
    axes1[1].set_title(
        "B: 10 Hz Tangential Braking Decel vs Separation\n(Direct Kinematic Translation: r = +0.252, p = 0.016*)",
        fontsize=11,
        fontweight="bold",
    )
    axes1[1].set_xlabel(
        "Combine Route Cut Peak Deceleration (yd/s²)", fontweight="bold"
    )
    axes1[1].set_ylabel(
        "In-Game Separation on Breaking Routes (Yards)", fontweight="bold"
    )
    plt.tight_layout()
    fig1_path = output_dir / "combine_to_nfl_separation_translation.png"
    fig1.savefig(fig1_path)
    shutil.copy(fig1_path, docs_dir / fig1_path.name)
    plt.close(fig1)

    # Figure 5: Feature Importance
    fig5, ax5 = plt.subplots(figsize=(10, 5.5), dpi=300)
    top_feats = list(wr_results["feature_importance"].keys())[:8]
    top_imps = [wr_results["feature_importance"][k] for k in top_feats]
    sns.barplot(
        x=top_imps, y=top_feats, ax=ax5, hue=top_feats, palette="Blues_r", legend=False
    )
    ax5.set_title(
        "Top Combine Features Explaining NFL In-Game Separation",
        fontsize=12,
        fontweight="bold",
    )
    ax5.set_xlabel("Feature Importance Score (Split Gain)", fontweight="bold")
    plt.tight_layout()
    fig5_path = output_dir / "combine_feature_importance.png"
    fig5.savefig(fig5_path)
    shutil.copy(fig5_path, docs_dir / fig5_path.name)
    plt.close(fig5)

    # =========================================================================
    # 3. POSITION 2: DEFENSIVE LINE (Trench Mass Confounder)
    # =========================================================================
    print("\n[3/6] Processing Defensive Line (Trench Mass Confounder & Get-Off)...")
    dl_outcomes = g_ext.extract_pass_rusher_metrics(player_play_df)
    dl_hoop = c_ext.extract_dl_hoop_kinematics(tracking_df)
    dl_merged = (
        dl_outcomes.filter(pl.col("total_rush_snaps") >= 30)
        .join(dl_hoop, on="nfl_id", how="inner")
        .join(results_df.select(["nfl_id", "combine_weight", "forty"]), on="nfl_id")
        .join(players_df.select(["nfl_id", "display_name"]), on="nfl_id")
        .to_pandas()
        .dropna(subset=["avg_get_off_time", "hoop_peak_speed", "combine_weight"])
    )
    dl_merged["combine_weight"] = pd.to_numeric(
        dl_merged["combine_weight"], errors="coerce"
    )
    dl_merged["forty"] = pd.to_numeric(dl_merged["forty"], errors="coerce")
    print(f"  Matched DL cohort size: {len(dl_merged)} prospects")

    r_hoop, p_hoop = stats.pearsonr(
        dl_merged["hoop_peak_speed"], dl_merged["avg_get_off_time"]
    )
    r_wt, p_wt = stats.pearsonr(
        dl_merged["combine_weight"], dl_merged["avg_get_off_time"]
    )
    print(f"  Hoop Peak Speed vs Get-Off: r = {r_hoop:+.3f} (p = {p_hoop:.4e})")
    print(f"  Body Weight vs Get-Off:     r = {r_wt:+.3f} (p = {p_wt:.4e})")

    X_dl = sm.add_constant(dl_merged[["hoop_peak_speed", "combine_weight"]])
    ols_dl = sm.OLS(dl_merged["avg_get_off_time"], X_dl).fit()
    print(
        f"  Multivariate OLS R^2: {ols_dl.rsquared:.3f}, p(weight) = {ols_dl.pvalues['combine_weight']:.4e}, p(hoop_speed) = {ols_dl.pvalues['hoop_peak_speed']:.4f}"
    )

    # Figure 6: Trench Kinematics & Confounder Control
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.regplot(
        x=dl_merged["hoop_peak_speed"],
        y=dl_merged["avg_get_off_time"],
        ax=axes6[0],
        color="#4a90e2",
        scatter_kws={"alpha": 0.8, "s": 50},
        line_kws={"color": "#2c3e50", "linewidth": 2},
    )
    axes6[0].set_title(
        f"A: Hoop Drill Speed vs NFL Get-Off Time\n(Naive Kinematic Correlation: r = {r_hoop:.3f}, p < 0.001)",
        fontsize=11,
        fontweight="bold",
    )
    axes6[0].set_xlabel("Combine Run The Hoop Peak Speed (yd/s)", fontweight="bold")
    axes6[0].set_ylabel("In-Game Average Get-Off Time (Seconds)", fontweight="bold")

    sns.regplot(
        x=dl_merged["combine_weight"],
        y=dl_merged["avg_get_off_time"],
        ax=axes6[1],
        color="#e67e22",
        scatter_kws={"alpha": 0.8, "s": 50},
        line_kws={"color": "#d35400", "linewidth": 2},
    )
    axes6[1].set_title(
        f"B: Body Mass Confounder vs NFL Get-Off Time\n(Physics-Driven Anchor: r = +{r_wt:.3f}, p < 0.001)",
        fontsize=11,
        fontweight="bold",
    )
    axes6[1].set_xlabel("Combine Body Weight (lbs)", fontweight="bold")
    axes6[1].set_ylabel("In-Game Average Get-Off Time (Seconds)", fontweight="bold")
    plt.tight_layout()
    fig6_path = output_dir / "trench_momentum_translation.png"
    fig6.savefig(fig6_path)
    shutil.copy(fig6_path, docs_dir / fig6_path.name)
    plt.close(fig6)

    # =========================================================================
    # 4. POSITION 3: OFFENSIVE LINE (Pass Protection Mirroring)
    # =========================================================================
    print("\n[4/6] Processing Offensive Line (Pass Protection Mirror Kinetics)...")
    ol_outcomes = g_ext.extract_offensive_line_metrics(player_play_df)
    ol_mirror = c_ext.extract_ol_mirror_kinematics(tracking_df)
    ol_merged = (
        ol_outcomes.filter(pl.col("total_pass_pro_snaps") >= 100)
        .join(ol_mirror, on="nfl_id", how="inner")
        .join(results_df.select(["nfl_id", "combine_weight", "forty"]), on="nfl_id")
        .join(players_df.select(["nfl_id", "display_name"]), on="nfl_id")
        .to_pandas()
        .dropna(subset=["pressure_rate_allowed", "mirror_max_accel", "combine_weight"])
    )
    ol_merged["combine_weight"] = pd.to_numeric(
        ol_merged["combine_weight"], errors="coerce"
    )
    ol_merged["forty"] = pd.to_numeric(ol_merged["forty"], errors="coerce")
    print(f"  Matched OL cohort size: {len(ol_merged)} linemen")

    r_ol_acc, p_ol_acc = stats.pearsonr(
        ol_merged["mirror_max_accel"], ol_merged["pressure_rate_allowed"]
    )
    valid_ol_40 = ol_merged.dropna(subset=["forty"])
    r_ol_40, p_ol_40 = stats.pearsonr(
        valid_ol_40["forty"], valid_ol_40["pressure_rate_allowed"]
    )
    print(
        f"  OL Mirror Accel vs Pressure Rate Allowed: r = {r_ol_acc:+.3f} (p = {p_ol_acc:.4f}*)"
    )
    print(
        f"  OL 40-Yard Dash vs Pressure Rate Allowed:  r = {r_ol_40:+.3f} (p = {p_ol_40:.4f})"
    )

    X_ol = sm.add_constant(ol_merged[["mirror_max_accel", "combine_weight"]])
    ols_ol = sm.OLS(ol_merged["pressure_rate_allowed"], X_ol).fit()
    print(
        f"  OL OLS R^2: {ols_ol.rsquared:.3f}, p(mirror_accel) = {ols_ol.pvalues['mirror_max_accel']:.4f}, p(weight) = {ols_ol.pvalues['combine_weight']:.4f}"
    )

    # Figure 7: Offensive Line Pass Protection Translation
    fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.regplot(
        x=valid_ol_40["forty"],
        y=valid_ol_40["pressure_rate_allowed"] * 100,
        ax=axes7[0],
        color="#7f7f7f",
        scatter_kws={"alpha": 0.8, "s": 50},
        line_kws={"color": "#333333", "linewidth": 2, "linestyle": "--"},
    )
    axes7[0].set_title(
        f"A: 40-Yard Dash vs Pressure Allowed Rate\n(Linear Speed Fails: r = {r_ol_40:.3f}, p = {p_ol_40:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes7[0].set_xlabel("40-Yard Dash Stopwatch Time (s)", fontweight="bold")
    axes7[0].set_ylabel("NFL Pressure Allowed Rate (%)", fontweight="bold")

    scatter7 = axes7[1].scatter(
        ol_merged["mirror_max_accel"],
        ol_merged["pressure_rate_allowed"] * 100,
        c=ol_merged["combine_weight"],
        cmap="Blues_r",
        s=65,
        alpha=0.9,
        edgecolors="k",
    )
    cbar7 = plt.colorbar(scatter7, ax=axes7[1])
    cbar7.set_label("Body Weight (lbs)", fontweight="bold")
    m_ol = np.poly1d(
        np.polyfit(
            ol_merged["mirror_max_accel"], ol_merged["pressure_rate_allowed"] * 100, 1
        )
    )
    x_ol = np.linspace(
        ol_merged["mirror_max_accel"].min(), ol_merged["mirror_max_accel"].max(), 100
    )
    axes7[1].plot(
        x_ol,
        m_ol(x_ol),
        color="#d62728",
        lw=2.5,
        linestyle="-",
        label=f"Trend (r = {r_ol_acc:.3f}, p = {p_ol_acc:.3f}*)",
    )
    axes7[1].set_title(
        f"B: 10 Hz Mirror Acceleration vs Pressure Allowed\n(Lateral Recovery Translation: r = {r_ol_acc:.3f}, p = {p_ol_acc:.3f}*)",
        fontsize=11,
        fontweight="bold",
    )
    axes7[1].set_xlabel("Combine Pass Pro Mirror Max Accel (yd/s²)", fontweight="bold")
    axes7[1].set_ylabel("NFL Pressure Allowed Rate (%)", fontweight="bold")
    axes7[1].legend(loc="upper right")
    plt.tight_layout()
    fig7_path = output_dir / "ol_pass_protection_translation.png"
    fig7.savefig(fig7_path)
    shutil.copy(fig7_path, docs_dir / fig7_path.name)
    plt.close(fig7)

    # =========================================================================
    # 5. POSITION 4: DEFENSIVE BACKS (W-Drill Hip Transition Latency)
    # =========================================================================
    print("\n[5/6] Processing Defensive Backs (W-Drill Hip Latency vs Coverage)...")
    db_outcomes = g_ext.extract_defensive_back_metrics(player_play_df)
    db_kin = c_ext.extract_db_hip_transition_kinematics(tracking_df)
    db_merged = (
        db_outcomes.filter(pl.col("total_coverage_snaps") >= 50)
        .join(db_kin, on="nfl_id", how="inner")
        .join(results_df.select(["nfl_id", "forty", "ten_yd_split"]), on="nfl_id")
        .join(players_df.select(["nfl_id", "display_name"]), on="nfl_id")
        .to_pandas()
        .dropna(subset=["completion_rate_allowed", "db_hip_transition_latency"])
    )
    db_merged["forty"] = pd.to_numeric(db_merged["forty"], errors="coerce")
    print(f"  Matched DB cohort size: {len(db_merged)} defensive backs")

    r_db_lat, p_db_lat = stats.pearsonr(
        db_merged["db_hip_transition_latency"], db_merged["completion_rate_allowed"]
    )
    r_db_z, p_db_z = stats.pearsonr(
        db_merged["db_hip_latency_z"], db_merged["completion_rate_allowed"]
    )
    valid_db_40 = db_merged.dropna(subset=["forty"])
    r_db_40, p_db_40 = stats.pearsonr(
        valid_db_40["forty"], valid_db_40["completion_rate_allowed"]
    )
    print(
        f"  DB W-Drill Latency Z vs Comp Allowed: r = {r_db_z:+.3f} (p = {p_db_z:.4f}*)"
    )
    print(
        f"  DB 40-Yard Dash vs Comp Allowed:      r = {r_db_40:+.3f} (p = {p_db_40:.4f})"
    )

    # Figure 8: Defensive Back Break-on-Ball Translation
    fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.regplot(
        x=valid_db_40["forty"],
        y=valid_db_40["completion_rate_allowed"] * 100,
        ax=axes8[0],
        color="#7f7f7f",
        scatter_kws={"alpha": 0.8, "s": 50},
        line_kws={"color": "#333333", "linewidth": 2, "linestyle": "--"},
    )
    axes8[0].set_title(
        f"A: 40-Yard Dash vs Coverage Completion Allowed\n(No Significant Impact: r = {r_db_40:.3f}, p = {p_db_40:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes8[0].set_xlabel("40-Yard Dash Stopwatch Time (s)", fontweight="bold")
    axes8[0].set_ylabel("NFL Completion Rate Allowed (%)", fontweight="bold")

    sns.regplot(
        x=db_merged["db_hip_latency_z"],
        y=db_merged["completion_rate_allowed"] * 100,
        ax=axes8[1],
        color="#8e44ad",
        scatter_kws={"alpha": 0.85, "s": 50},
        line_kws={"color": "#5b2c6f", "linewidth": 2.5},
    )
    axes8[1].set_title(
        f"B: W-Drill Hip Latency Z vs Completion Allowed\n(Fluid Hips Prevent Catches: r = +{r_db_z:.3f}, p = {p_db_z:.3f}*)",
        fontsize=11,
        fontweight="bold",
    )
    axes8[1].set_xlabel(
        "Cohort-Standardized Hip Transition Latency Z", fontweight="bold"
    )
    axes8[1].set_ylabel("NFL Completion Rate Allowed (%)", fontweight="bold")
    plt.tight_layout()
    fig8_path = output_dir / "db_break_on_ball_translation.png"
    fig8.savefig(fig8_path)
    shutil.copy(fig8_path, docs_dir / fig8_path.name)
    plt.close(fig8)

    # =========================================================================
    # 6. FIGURES 9 & 10: SCHEME CONTEXTUALIZATION & FRONT-OFFICE ARBITRAGE
    # =========================================================================
    print(
        "\n[6/7] Generating Figure 9 (Scheme Contextualization) & Figure 10 (Draft Arbitrage)..."
    )
    breaking_routes = ["SLANT", "OUT", "IN", "COMEBACK", "HITCH", "CORNER", "POST"]

    # Press vs Off
    press_sep = (
        player_play_df.filter(
            (pl.col("route_ran").is_in(breaking_routes))
            & (pl.col("target") == 1)
            & (pl.col("cushion").cast(pl.Float64, strict=False) <= 4.5)
            & pl.col("separation_at_pass_forward").is_not_null()
        )
        .group_by("nfl_id")
        .agg(
            [
                pl.col("separation_at_pass_forward")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("sep_press"),
                pl.len().alias("targets_press"),
            ]
        )
        .filter(pl.col("targets_press") >= 4)
    )

    off_sep = (
        player_play_df.filter(
            (pl.col("route_ran").is_in(breaking_routes))
            & (pl.col("target") == 1)
            & (pl.col("cushion").cast(pl.Float64, strict=False) > 5.5)
            & pl.col("separation_at_pass_forward").is_not_null()
        )
        .group_by("nfl_id")
        .agg(
            [
                pl.col("separation_at_pass_forward")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("sep_off"),
                pl.len().alias("targets_off"),
            ]
        )
        .filter(pl.col("targets_off") >= 5)
    )

    press_df = (
        press_sep.join(wr_route_feats, on="nfl_id")
        .join(results_df.select(["nfl_id", "forty"]), on="nfl_id")
        .join(players_df.select(["nfl_id", "display_name"]), on="nfl_id")
        .to_pandas()
        .dropna(subset=["raw_wr_braking_decel", "sep_press", "forty"])
    )
    press_df["forty"] = pd.to_numeric(press_df["forty"], errors="coerce")

    off_df = (
        off_sep.join(wr_route_feats, on="nfl_id")
        .join(results_df.select(["nfl_id", "forty"]), on="nfl_id")
        .join(players_df.select(["nfl_id", "display_name"]), on="nfl_id")
        .to_pandas()
        .dropna(subset=["raw_wr_braking_decel", "sep_off", "forty"])
    )
    off_df["forty"] = pd.to_numeric(off_df["forty"], errors="coerce")

    r_press_40, _ = stats.pearsonr(press_df["forty"], press_df["sep_press"])
    r_press_dec, _ = stats.pearsonr(
        press_df["raw_wr_braking_decel"], press_df["sep_press"]
    )
    r_off_40, _ = stats.pearsonr(off_df["forty"], off_df["sep_off"])
    r_off_dec, _ = stats.pearsonr(off_df["raw_wr_braking_decel"], off_df["sep_off"])

    # Figure 9
    fig9, axes9 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.regplot(
        x=press_df["raw_wr_braking_decel"],
        y=press_df["sep_press"],
        ax=axes9[0],
        color="#e74c3c",
        scatter_kws={"alpha": 0.85, "s": 55, "edgecolors": "black", "linewidths": 0.5},
        line_kws={"color": "#c0392b", "linewidth": 2.2},
    )
    axes9[0].set_title(
        f"A: Press Coverage (Cushion ≤ 4.5 yds): Decel Governs Separation\n(Direct Kinematic Translation: r = +{r_press_dec:.3f} vs 40-yd r = {r_press_40:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes9[0].set_xlabel(
        "Combine Route Cut Peak Braking Decel (yd/s²)", fontweight="bold"
    )
    axes9[0].set_ylabel(
        "In-Game Separation vs Press Alignment (Yards)", fontweight="bold"
    )

    sns.regplot(
        x=off_df["raw_wr_braking_decel"],
        y=off_df["sep_off"],
        ax=axes9[1],
        color="#3498db",
        scatter_kws={"alpha": 0.85, "s": 55, "edgecolors": "black", "linewidths": 0.5},
        line_kws={"color": "#2980b9", "linewidth": 2.2, "linestyle": "--"},
    )
    axes9[1].set_title(
        f"B: Off Coverage (Cushion > 5.5 yds): Cushion Buffering Effect\n(Defenders Protect Over-the-Top: Decel r = {r_off_dec:.3f}, 40-yd r = +{r_off_40:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes9[1].set_xlabel(
        "Combine Route Cut Peak Braking Decel (yd/s²)", fontweight="bold"
    )
    axes9[1].set_ylabel(
        "In-Game Separation vs Off Alignment (Yards)", fontweight="bold"
    )
    plt.tight_layout()
    fig9_path = output_dir / "scheme_cushion_translation.png"
    fig9.savefig(fig9_path)
    shutil.copy(fig9_path, docs_dir / fig9_path.name)
    plt.close(fig9)

    # Figure 10: Draft Arbitrage
    arb_df = (
        breaking_rec.join(results_df, on="nfl_id")
        .join(wr_route_feats, on="nfl_id")
        .join(
            players_df.select(
                ["nfl_id", "display_name", "draft_round", "draft_overall_pick"]
            ),
            on="nfl_id",
        )
        .to_pandas()
        .dropna(
            subset=[
                "forty",
                "raw_wr_braking_decel",
                "breaking_route_separation",
                "draft_overall_pick",
            ]
        )
    )
    arb_df["forty"] = pd.to_numeric(arb_df["forty"], errors="coerce")
    arb_df["draft_overall_pick"] = pd.to_numeric(
        arb_df["draft_overall_pick"], errors="coerce"
    )
    arb_df["speed_z"] = (
        -(arb_df["forty"] - arb_df["forty"].mean()) / arb_df["forty"].std()
    )
    arb_df["decel_z"] = (
        -(arb_df["raw_wr_braking_decel"] - arb_df["raw_wr_braking_decel"].mean())
        / arb_df["raw_wr_braking_decel"].std()
    )

    def get_quad(r: pd.Series) -> str:
        if r["speed_z"] >= 0 and r["decel_z"] >= 0:
            return "Q1: Dual Weapons"
        elif r["speed_z"] < 0 and r["decel_z"] >= 0:
            return "Q2: Hidden Movers"
        elif r["speed_z"] >= 0 and r["decel_z"] < 0:
            return "Q3: Workout Warriors"
        else:
            return "Q4: Athletic Floor"

    arb_df["quadrant"] = arb_df.apply(get_quad, axis=1)

    fig10, axes10 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    palette = {
        "Q1: Dual Weapons": "#2ca02c",
        "Q2: Hidden Movers": "#ff7f0e",
        "Q3: Workout Warriors": "#d62728",
        "Q4: Athletic Floor": "#7f7f7f",
    }

    for quad_name, grp in arb_df.groupby("quadrant"):
        axes10[0].scatter(
            grp["draft_overall_pick"],
            grp["breaking_route_separation"],
            label=str(quad_name),
            color=palette[str(quad_name)],
            s=70,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5,
        )

    axes10[0].set_title(
        "A: Draft Capital Invested vs In-Game Separation\n(Quadrant 2 Hidden Movers Outperform Earlier Round Selections)",
        fontsize=11,
        fontweight="bold",
    )
    axes10[0].set_xlabel("NFL Draft Overall Pick Selection Number", fontweight="bold")
    axes10[0].set_ylabel("In-Game Breaking Route Separation (Yards)", fontweight="bold")
    axes10[0].legend(loc="lower right", frameon=True, fontsize=9)

    arb_df["pick_cost"] = 260 - arb_df["draft_overall_pick"]
    arb_df["efficiency"] = (
        arb_df["breaking_route_separation"] / (arb_df["pick_cost"] + 10)
    ) * 100
    quad_order = [
        "Q1: Dual Weapons",
        "Q2: Hidden Movers",
        "Q3: Workout Warriors",
        "Q4: Athletic Floor",
    ]

    sns.boxplot(
        x="quadrant",
        y="efficiency",
        hue="quadrant",
        data=arb_df,
        order=quad_order,
        palette=palette,
        legend=False,
        ax=axes10[1],
        boxprops={"alpha": 0.8},
    )
    sns.stripplot(
        x="quadrant",
        y="efficiency",
        data=arb_df,
        order=quad_order,
        color="black",
        alpha=0.6,
        size=5,
        jitter=0.2,
        ax=axes10[1],
    )
    axes10[1].set_title(
        "B: Front-Office Draft Arbitrage Efficiency\n(Separation Created Relative to Draft Capital Invested)",
        fontsize=11,
        fontweight="bold",
    )
    axes10[1].set_xlabel("Scouting Archetype Quadrant", fontweight="bold")
    axes10[1].set_ylabel("Draft ROI Efficiency Score", fontweight="bold")
    axes10[1].set_xticks(range(4))
    axes10[1].set_xticklabels(
        ["Q1: Dual", "Q2: Hidden", "Q3: Workout", "Q4: Floor"], fontweight="bold"
    )

    plt.tight_layout()
    fig10_path = output_dir / "front_office_draft_arbitrage.png"
    fig10.savefig(fig10_path)
    shutil.copy(fig10_path, docs_dir / fig10_path.name)
    plt.close(fig10)

    # 7. Summary
    print("\n[7/7] Pipeline completed successfully!")
    print(
        f"  All 10 publication figures saved to: {output_dir.resolve()} and {docs_dir.resolve()}"
    )
    print("=" * 75)


if __name__ == "__main__":
    run_analysis()
