"""Download and unpack NFL Big Data Bowl 2027 competition dataset."""

import argparse
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from bdb_2027.config import Config


def download_competition_data(dest_dir: Path | None = None) -> None:
    """Download and extract all competition files using the Kaggle API."""
    load_dotenv()
    config = Config()
    target_dir = dest_dir or config.paths.raw_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = config.competition_slug
    print(f"Connecting to Kaggle API to download '{slug}' into {target_dir}...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading competition files to {target_dir}...")
    api.competition_download_files(config.competition_slug, path=str(target_dir))

    zip_path = target_dir / f"{config.competition_slug}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete. Cleaning up zip archive...")
        zip_path.unlink()

    # Flatten nested folder if archive extracted into a subdirectory
    nested_dir = target_dir / config.competition_slug
    if nested_dir.exists() and nested_dir.is_dir():
        import shutil

        for item in nested_dir.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        nested_dir.rmdir()

    print(f"Successfully downloaded and prepared files in {target_dir}:")
    for f in sorted(target_dir.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.2f} MB")


def main() -> None:
    """CLI entry point to download competition data."""
    parser = argparse.ArgumentParser(
        description="Download NFL Big Data Bowl 2027 dataset"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: data/raw)",
    )
    args = parser.parse_args()
    download_competition_data(args.dest)


if __name__ == "__main__":
    main()
