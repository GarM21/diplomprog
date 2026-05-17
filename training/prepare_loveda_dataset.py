from pathlib import Path
import argparse
import random
import shutil

import numpy as np
from PIL import Image


DATASET_SLUG = "alienxc137/loveda-satellite-images-semantic-segmentation"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_GREEN_CLASSES = {6, 7}


def download_dataset(raw_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError(
            "Kaggle package is not installed. Run: pip install -r requirements.txt"
        ) from error

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(raw_dir), unzip=True)


def collect_loveda_samples(raw_dir: Path, split_name: str) -> list[tuple[Path, Path, str]]:
    split_dirs = [
        path
        for path in raw_dir.rglob(split_name)
        if path.is_dir() and path.name.lower() == split_name.lower()
    ]
    samples: list[tuple[Path, Path, str]] = []

    for split_dir in split_dirs:
        for images_dir in split_dir.rglob("images_png"):
            masks_dir = images_dir.parent / "masks_png"
            if not masks_dir.exists():
                masks_dir = images_dir.parent / "labels_png"
            if not masks_dir.exists():
                continue

            domain = images_dir.parent.name.lower()
            for image_path in _iter_images(images_dir):
                mask_path = masks_dir / image_path.name
                if mask_path.exists():
                    samples.append((image_path, mask_path, domain))

    if not samples:
        raise FileNotFoundError(
            f"No LoveDA image-mask pairs found for split '{split_name}' in {raw_dir}."
        )

    return sorted(samples, key=lambda item: str(item[0]))


def prepare_loveda(
    raw_dir: Path,
    output_dir: Path,
    val_test_ratio: float,
    seed: int,
    green_classes: set[int],
    overwrite: bool,
) -> None:
    train_samples = collect_loveda_samples(raw_dir, "Train")
    val_samples = collect_loveda_samples(raw_dir, "Val")
    random.Random(seed).shuffle(val_samples)

    test_size = int(len(val_samples) * val_test_ratio)
    test_samples = val_samples[:test_size]
    validation_samples = val_samples[test_size:]

    split_map = {
        "train": train_samples,
        "val": validation_samples,
        "test": test_samples,
    }

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite to recreate it."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split, samples in split_map.items():
        images_dir = output_dir / split / "images"
        masks_dir = output_dir / split / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path, domain in samples:
            output_name = f"{domain}_{image_path.stem}.png"
            _save_image(image_path, images_dir / output_name)
            _save_binary_mask(mask_path, masks_dir / output_name, green_classes)

        print(f"{split}: {len(samples)} pairs")


def _save_image(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path) as image:
        image.convert("RGB").save(target_path)


def _save_binary_mask(
    source_path: Path,
    target_path: Path,
    green_classes: set[int],
) -> None:
    with Image.open(source_path) as mask:
        mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)

    binary_mask = np.isin(mask_array, list(green_classes)).astype(np.uint8) * 255
    Image.fromarray(binary_mask, mode="L").save(target_path)


def _iter_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _parse_classes(value: str) -> set[int]:
    classes = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not classes:
        raise argparse.ArgumentTypeError("At least one class id is required.")
    return classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert LoveDA masks to binary vegetation masks."
    )
    parser.add_argument("--raw-dir", default="dataset_raw/loveda")
    parser.add_argument("--output-dir", default="dataset/loveda_binary")
    parser.add_argument("--val-test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--green-classes",
        type=_parse_classes,
        default=DEFAULT_GREEN_CLASSES,
        help="Comma-separated LoveDA class ids treated as green areas. Default: 6,7.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate --output-dir if it already exists.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already downloaded files in --raw-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not 0 <= args.val_test_ratio < 1:
        raise ValueError("val-test-ratio must be in range [0, 1).")

    if not args.skip_download:
        download_dataset(raw_dir)

    prepare_loveda(
        raw_dir=raw_dir,
        output_dir=output_dir,
        val_test_ratio=args.val_test_ratio,
        seed=args.seed,
        green_classes=args.green_classes,
        overwrite=args.overwrite,
    )
    print(f"Prepared LoveDA binary dataset: {output_dir}")


if __name__ == "__main__":
    main()
