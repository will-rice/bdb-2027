"""NFL Big Data Bowl 2027 Configuration."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    """File and directory paths for BDB 2027 data."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    logs_dir: Path = Path("logs")

    # Raw file paths
    players_file: Path = Path("data/raw/players.csv")
    combine_results_file: Path = Path("data/raw/combine_results.csv")
    combine_tracking_file: Path = Path("data/raw/combine_tracking.csv")
    player_career_file: Path = Path("data/raw/player_career_successes.csv")
    player_play_file: Path = Path("data/raw/player_play.csv")
    games_file: Path = Path("data/raw/games.csv")
    game_tracking_2023_file: Path = Path("data/raw/game_tracking_2023.csv")
    game_tracking_2024_file: Path = Path("data/raw/game_tracking_2024.csv")
    game_tracking_2025_file: Path = Path("data/raw/game_tracking_2025.csv")


class FeatureConfig(BaseModel):
    """Tracking and feature engineering parameters."""

    tracking_hz: float = 10.0
    field_length: float = 120.0
    field_width: float = 53.33

    # Standardization & Normalization
    center_by_draft_year: bool = True
    min_tracking_frames: int = 10

    # Drill feature thresholds
    forty_yard_distance: float = 40.0
    ten_yard_distance: float = 10.0
    twenty_yard_distance: float = 20.0


class Config(BaseModel):
    """Main project configuration class for NFL Big Data Bowl 2027."""

    # Competition Info
    competition_slug: str = "nfl-big-data-bowl-2027"
    project_name: str = "bdb-2027"

    # Paths
    paths: DataPaths = Field(default_factory=DataPaths)
    features: FeatureConfig = Field(default_factory=FeatureConfig)

    # Reproducibility
    seed: int = 42
    base_model: str = "lightgbm"

    # Validation Strategy
    cv_strategy: Literal["group_kfold", "kfold", "stratified"] = "group_kfold"
    n_splits: int = 5
    group_col: str = "nfl_id"  # Group by player to prevent data leakage

    # Position Focus (optional filter for sub-group modeling)
    target_positions: list[str] = [
        "WR",
        "CB",
        "DE",
        "DT",
        "T",
        "G",
        "C",
        "TE",
        "RB",
        "LB",
        "S",
    ]

    # Training Hyperparameters
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 15
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-2

    # Tree models
    lgb_n_estimators: int = 500
    lgb_learning_rate: float = 0.03
    lgb_max_depth: int = 5
