from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLASS_MAP = {
    "FR": "fire",
    "FU": "fire",
    "NR": "no_fire",
    "NU": "no_fire",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_split(source_root: Path, output_root: Path, split_name: str) -> int:
    split_dir = source_root / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    copied = 0
    for source_code, target_class in CLASS_MAP.items():
        source_class_dir = split_dir / source_code
        if not source_class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {source_class_dir}")

        target_dir = output_root / split_name.lower() / target_class
        ensure_dir(target_dir)

        for image_path in sorted(source_class_dir.glob("*.jpg")):
            target_name = f"{source_code.lower()}__{image_path.name}"
            shutil.copy2(image_path, target_dir / target_name)
            copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize MultiFire20K into the project's canonical fire/no_fire split layout.")
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing extracted MultiFire20K split folders")
    parser.add_argument("--output-root", type=Path, required=True, help="Canonical output root to create")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train"],
        choices=["Train", "Val", "Test"],
        help="MultiFire20K splits to copy into canonical layout",
    )
    args = parser.parse_args()

    total = 0
    for split_name in args.splits:
        copied = copy_split(args.input_root, args.output_root, split_name)
        print(f"Copied {copied} files from {split_name}")
        total += copied

    print(f"Prepared canonical dataset at: {args.output_root}")
    print(f"Total copied files: {total}")


if __name__ == "__main__":
    main()
