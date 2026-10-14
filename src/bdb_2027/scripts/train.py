"""Train script."""

from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from git import Repo
from lightning import seed_everything

from bdb_2027.config import Config


def main() -> None:
    """Train script."""
    parser = ArgumentParser(description="Train script.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--project", default="bdb-2027", type=str)
    parser.add_argument("--num_devices", default=1, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    config = Config()

    seed_everything(config.seed, workers=True)

    git_repo = Repo()
    git_hash = git_repo.head.object.hexsha[:7]
    model_name = config.base_model.split("/")[-1]
    experiment_path = args.log_root / f"{model_name}-{git_hash}"
    experiment_path.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    main()
