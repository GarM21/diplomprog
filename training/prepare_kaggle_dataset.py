from pathlib import Path
import argparse
import random
import shutil

from PIL import Image


DATASET_SLUG = "asim3000/satellite-imagery-urban-tree-segmentation-india"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def find_pair_dirs(raw_dir: Path) -> tuple[Path, Path]:
    image_dirs = [path for path in raw_dir.rglob("images") if path.is_dir()]
    mask_dirs = [path for path in raw_dir.rglob("masks") if path.is_dir()]

    for images_dir in image_dirs:
        for masks_dir in mask_dirs:
            image_stems = _file_stems(images_dir)
            mask_stems = _file_stems(masks_dir)
            if image_stems and image_stems.issubset(mask_stems):
                return images_dir, masks_dir

    raise FileNotFoundError(
        "Could not find matching images/ and masks/ directories in downloaded dataset."
    )


def prepare_split(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    overwrite: bool,
) -> None:
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    mask_by_stem = {
        path.stem: path
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    samples = [(image_path, mask_by_stem.get(image_path.stem)) for image_path in image_paths]
    missing = [image_path.name for image_path, mask_path in samples if mask_path is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing masks for images: {preview}")

    paired_samples = [(image_path, mask_path) for image_path, mask_path in samples if mask_path]
    random.Random(seed).shuffle(paired_samples)

    total = len(paired_samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    split_map = {
        "train": paired_samples[:train_end],
        "val": paired_samples[train_end:val_end],
        "test": paired_samples[val_end:],
    }

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite to recreate it."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split, split_samples in split_map.items():
        split_images_dir = output_dir / split / "images"
        split_masks_dir = output_dir / split / "masks"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_masks_dir.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in split_samples:
            output_name = f"{image_path.stem}.png"
            _save_image_as_png(image_path, split_images_dir / output_name)
            _save_mask_as_png(mask_path, split_masks_dir / output_name)

        print(f"{split}: {len(split_samples)} pairs")


def _save_image_as_png(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path) as image:
        image.convert("RGB").save(target_path)


def _save_mask_as_png(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path) as mask:
        mask.convert("L").save(target_path)


def _file_stems(directory: Path) -> set[str]:
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the Kaggle urban tree segmentation dataset."
    )
    parser.add_argument("--raw-dir", default="dataset_raw/urban_tree_india")
    parser.add_argument("--output-dir", default="dataset/urban_tree_india")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
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

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train-ratio and val-ratio must be positive.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1.")

    if not args.skip_download:
        download_dataset(raw_dir)

    images_dir, masks_dir = find_pair_dirs(raw_dir)
    prepare_split(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Prepared dataset: {output_dir}")


if __name__ == "__main__":
    main()
