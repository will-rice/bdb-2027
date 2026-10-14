"""Datasets and feature extractors for BDB 2027."""

from typing import NamedTuple

from bdb_2027.datasets.combine import CombineFeatureExtractor
from bdb_2027.datasets.game import GameOutcomeExtractor


class Batch(NamedTuple):
    """Batch of data."""


__all__ = ["Batch", "CombineFeatureExtractor", "GameOutcomeExtractor"]
