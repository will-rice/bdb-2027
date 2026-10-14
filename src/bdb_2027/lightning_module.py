"""Lightning Module."""

import torch
from lightning.pytorch import LightningModule

from bdb_2027.config import Config
from bdb_2027.datasets import Batch


class BaseLightningModule(LightningModule):
    """Base Lightning Module."""

    def __init__(
        self, config: Config, push_to_hub: bool = False, sync_dist: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.push_to_hub = push_to_hub
        self.sync_dist = sync_dist

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        """Train step."""
        raise NotImplementedError("Training step not implemented.")

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Validate step."""
        raise NotImplementedError("Validation step not implemented.")

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list]:
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return [optimizer], []
