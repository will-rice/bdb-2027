"""The helper functions defined in the submission notebook, run against small synthetic inputs."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "frame_first.ipynb"


@pytest.fixture(scope="module")
def helpers() -> dict:
    """Execute the notebook's ``# helpers`` cell and return its namespace."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    matches = [s for s in sources if s.startswith("# helpers")]
    assert len(matches) == 1
    namespace: dict = {}
    exec(matches[0], namespace)
    return namespace


def test_split_half_reliability_is_high_for_a_stable_trait(helpers: dict) -> None:
    """Players who differ by far more than their rep-to-rep noise should agree across random halves."""
    rng = np.random.default_rng(0)
    reps = pd.DataFrame({"nfl_id": np.repeat(np.arange(40), 60)})
    reps["value"] = reps["nfl_id"] * 1.0 + rng.normal(0, 0.5, len(reps))
    assert helpers["split_half_reliability"](reps, "value", splits=20, seed=0) > 0.95


def test_split_half_reliability_is_near_zero_for_noise(helpers: dict) -> None:
    """With no player effect at all, the two halves should be unrelated."""
    rng = np.random.default_rng(1)
    reps = pd.DataFrame(
        {"nfl_id": np.repeat(np.arange(60), 40), "value": rng.normal(0, 1, 2400)}
    )
    assert (
        abs(helpers["split_half_reliability"](reps, "value", splits=50, seed=0)) < 0.15
    )


def test_leave_one_class_out_never_sees_the_held_out_targets(helpers: dict) -> None:
    """Replacing one draft class's outcomes must not change the predictions made for that class."""
    rng = np.random.default_rng(2)
    players = pd.DataFrame(
        {
            "draft_year": np.repeat([2023, 2024, 2025], 20),
            "weight": rng.normal(200, 15, 60),
        }
    )
    players["sep"] = 3.0 - 0.01 * players["weight"] + rng.normal(0, 0.05, 60)
    before = helpers["leave_one_class_out"](players, "weight", "sep")
    held_out = (players["draft_year"] == 2025).to_numpy()
    players.loc[held_out, "sep"] = 99.0
    after = helpers["leave_one_class_out"](players, "weight", "sep")
    assert np.allclose(before[held_out], after[held_out])
    assert not np.allclose(before[~held_out], after[~held_out])


def test_leave_one_class_out_recovers_a_relation_shared_across_classes(
    helpers: dict,
) -> None:
    """A relation that holds in every class should be predicted well for each held-out class."""
    rng = np.random.default_rng(3)
    players = pd.DataFrame(
        {
            "draft_year": np.repeat([2023, 2024, 2025], 20),
            "weight": rng.normal(200, 15, 60),
        }
    )
    players["sep"] = 3.0 - 0.01 * players["weight"] + rng.normal(0, 0.02, 60)
    predicted = helpers["leave_one_class_out"](players, "weight", "sep")
    assert np.corrcoef(predicted, players["sep"])[0, 1] > 0.95


def test_context_residual_removes_route_mix(helpers: dict) -> None:
    """Two equal receivers who run different routes should look equal once the route is held fixed."""
    rng = np.random.default_rng(4)
    route_mean = {"GO": 2.0, "SCREEN": 6.0}
    rows = []
    for player, routes, skill in [
        (1, ["GO"], 0.0),
        (2, ["SCREEN"], 0.0),
        (3, ["GO", "SCREEN"], 1.0),
        (4, ["GO", "SCREEN"], -1.0),
    ]:
        for route in routes:
            for _ in range(200):
                rows.append(
                    {
                        "nfl_id": player,
                        "route_ran": route,
                        "sep": route_mean[route] + skill + rng.normal(0, 0.3),
                    }
                )
    reps = pd.DataFrame(rows)
    reps["adjusted"] = helpers["context_residual"](reps, "sep", "C(route_ran)")
    raw = reps.groupby("nfl_id")["sep"].mean()
    adjusted = reps.groupby("nfl_id")["adjusted"].mean()
    assert raw[2] - raw[1] > 3.5
    assert abs(adjusted[2] - adjusted[1]) < 0.1
    assert adjusted[3] > 0.8
    assert adjusted[4] < -0.8


def drill_rep(
    nfl_id: int,
    event_id: str,
    drill: str,
    brake: float,
    start_second: int,
    ball: bool = False,
) -> list[dict]:
    """Frames for one route-drill rep: speed ramps to 8 yd/s in a second, then falls at ``brake`` yd/s^2."""
    speeds = np.concatenate(
        [np.linspace(0.0, 8.0, 10), np.arange(8.0 - brake / 10, 0.0, -brake / 10)]
    )
    rows = []
    for i, speed in enumerate(speeds):
        frame = {
            "nfl_id": nfl_id,
            "event_id": event_id,
            "draft_year": 2025,
            "drill_name": drill,
            "entity_type": "PLAYER",
            "time": f"2025-03-02T00:{start_second:02d}:{i * 0.1:06.3f}",
            "s": float(speed),
        }
        rows.append(frame)
        if ball:
            rows.append({**frame, "entity_type": "BALL", "s": 20.0 if i < 12 else 0.0})
    return rows


def test_drill_reps_ignore_ball_rows_and_keep_attempts_apart(helpers: dict) -> None:
    """A ball stopping at the catch, or a second attempt, must not register as the receiver braking."""
    pl = helpers["pl"]
    rows = (
        drill_rep(1, "a1", "CURL_ROUTE_RIGHT", brake=4.0, start_second=0, ball=True)
        + drill_rep(1, "a2", "CURL_ROUTE_RIGHT", brake=4.0, start_second=1)
        + drill_rep(2, "b1", "CURL_ROUTE_RIGHT", brake=2.0, start_second=2)
    )
    reps = helpers["drill_reps"](pl.DataFrame(rows))
    assert reps["event_id"].to_list() == ["a1", "a2", "b1"]
    assert np.allclose(reps["brake"], [4.0, 4.0, 2.0], atol=0.2)
    assert np.allclose(reps["speed"], 8.0)
    assert reps["brake_z"][0] > reps["brake_z"][2]


def test_drill_reps_are_standardized_within_drill(helpers: dict) -> None:
    """The hardest braker in a gentle drill should score the same as the hardest braker in a violent one."""
    pl = helpers["pl"]
    rows = []
    for i, (drill, brake) in enumerate(
        [
            ("COMEBACK", 6.0),
            ("COMEBACK", 5.0),
            ("COMEBACK", 4.0),
            ("GO", 3.0),
            ("GO", 2.5),
            ("GO", 2.0),
        ]
    ):
        rows += drill_rep(i, f"e{i}", drill, brake=brake, start_second=i)
    reps = helpers["drill_reps"](pl.DataFrame(rows))
    assert reps["event_id"].to_list() == [f"e{i}" for i in range(6)]
    assert reps["brake"][0] - reps["brake"][3] > 2.5
    assert np.isclose(reps["brake_z"][0], reps["brake_z"][3], atol=0.05)


def test_drill_reps_drop_clips_too_short_to_differentiate(helpers: dict) -> None:
    """A clip of a few frames has no usable speed derivative."""
    pl = helpers["pl"]
    rows = (
        drill_rep(1, "long1", "GO", 4.0, 0)
        + drill_rep(2, "long2", "GO", 2.0, 1)
        + drill_rep(3, "short", "GO", 4.0, 2)[:5]
    )
    reps = helpers["drill_reps"](pl.DataFrame(rows))
    assert sorted(reps["event_id"].to_list()) == ["long1", "long2"]


def game_rep(play_id: int, with_throw: bool) -> list[dict]:
    """Frames for one game rep: fast pre-snap motion, a ramp to 6 yd/s over two seconds, then faster after the throw."""
    rows = []
    for i in range(-5, 31):
        seconds = i / 10
        speed = (
            9.0 if seconds < 0 else min(3.0 * seconds, 6.0) if seconds <= 2.0 else 9.5
        )
        event = (
            "ball_snap"
            if i == 0
            else "pass_forward"
            if i == 20 and with_throw
            else None
        )
        rows.append(
            {
                "game_id": 1,
                "play_id": play_id,
                "nfl_id": 7,
                "time": datetime(2025, 9, 7, 13, 0, 10) + timedelta(seconds=seconds),
                "s": speed,
                "event": event,
            }
        )
    return rows


def test_play_speed_is_the_peak_between_snap_and_throw(helpers: dict) -> None:
    """Pre-snap motion and running after the throw must not count toward play speed."""
    pl = helpers["pl"]
    reps = helpers["play_speed"](
        pl.DataFrame(game_rep(1, with_throw=True), schema_overrides={"event": pl.Utf8})
    )
    assert reps.height == 1
    assert np.isclose(reps["peak_speed"][0], 6.0)
    assert np.isclose(reps["time_to_throw"][0], 2.0)


def test_play_speed_drops_reps_with_no_throw(helpers: dict) -> None:
    """A rep with no forward pass has no window to measure."""
    pl = helpers["pl"]
    frames = pl.DataFrame(
        game_rep(1, with_throw=True) + game_rep(2, with_throw=False),
        schema_overrides={"event": pl.Utf8},
    )
    assert helpers["play_speed"](frames)["play_id"].to_list() == [1]


def test_play_speed_rows_come_back_in_a_fixed_order(helpers: dict) -> None:
    """Reliability is estimated by random splits of the rows, so their order must not vary between runs."""
    pl = helpers["pl"]
    frames = pl.DataFrame(
        [row for play in (5, 3, 9, 1, 7) for row in game_rep(play, with_throw=True)],
        schema_overrides={"event": pl.Utf8},
    )
    assert helpers["play_speed"](frames)["play_id"].to_list() == [1, 3, 5, 7, 9]
