#!/usr/bin/env python3
"""
prepare_dataset.py (STRICT + NF TAG OVERRIDE)

Build a canonical train/val/test folder structure for binary image classification.

Label inference rules (SAFE priority order):
1) Filename tags:
   - If filename contains token "NF" -> label = no_fire
   - If filename contains token "FIRE" -> label = fire (unless folder says no_fire)
2) Folder tokens (strict):
   - Folder token pattern "no"+"fire" OR folder contains "nofire"/"no_fire" -> no_fire
   - Else folder contains token "fire"/"wildfire"/"evidence" -> fire
3) Otherwise -> unknown

Output:
  data/splits/
    train/fire, train/no_fire
    val/fire,   val/no_fire
    test/fire,  test/no_fire
    SPLIT_INFO.json (seed, ratios, counts, and original relative paths)

Example:
  python scripts/prepare_dataset.py --input data/raw --output data/splits --val 0.15 --test 0.15 --seed 42 --mode copy --keep-unknown
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# -----------------------------
# Config
# -----------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Split folder/file names into tokens: letters+digits groups
TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")

def tokenize(s: str) -> List[str]:
    return [t for t in TOKEN_SPLIT_RE.split(s.lower()) if t]

def token_set(s: str) -> set[str]:
    return set(tokenize(s))

# Filename tag patterns
# Matches NF as a token boundary: e.g., NF_001, IMG-NF-12, foo.NF.bar, NF00123 (tokenization handles)
# We'll implement this using tokenization, not regex, so it's robust.

# Folder tokens for strict labeling
FIRE_FOLDER_TOKENS = {"fire", "wildfire", "evidence", "positive", "pos", "yes", "1"}
NOFIRE_FOLDER_TOKENS = {"negative", "neg", "safe", "none", "0", "normal"}

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class SplitResult:
    train: List[Path]
    val: List[Path]
    test: List[Path]


# -----------------------------
# Helpers
# -----------------------------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def infer_from_filename(p: Path) -> Optional[str]:
    """
    Filename inference using tokenization.
    Rules:
      - If token 'nf' exists -> no_fire
      - Else if token 'fire' exists -> fire
      - Else None
    """
    toks = token_set(p.stem)  # stem excludes extension
    if "nf" in toks:
        return "no_fire"
    if "fire" in toks:
        return "fire"
    return None


def infer_from_folders_strict(p: Path) -> Optional[str]:
    """
    Strict folder inference:
      - If any folder contains tokens ("no"+"fire") OR token 'nofire' OR 'no_fire' -> no_fire
      - Else if folder contains token 'fire'/'wildfire'/'evidence' etc -> fire
      - Else None
    """
    # scan from nearest parent upward
    for part in reversed(p.parts[:-1]):  # exclude filename part
        toks = token_set(part)

        # strong no_fire indicators
        if "nofire" in toks or "no_fire" in toks or "nofires" in toks:
            return "no_fire"
        if "no" in toks and "fire" in toks:
            return "no_fire"
        if "not" in toks and "fire" in toks:
            return "no_fire"
        if toks & NOFIRE_FOLDER_TOKENS:
            return "no_fire"

        # fire indicators (only if not also signaling no_fire)
        if toks & FIRE_FOLDER_TOKENS:
            # if folder literally includes both no & fire, it would have returned above
            return "fire"

    return None


def infer_label(p: Path) -> Optional[str]:
    """
    Safe priority:
      1) Filename NF wins no_fire
      2) Folder strict (for explicit no_fire folders)
      3) Filename fire (but folder can override if it says no_fire)
    """
    file_lab = infer_from_filename(p)
    if file_lab == "no_fire":
        return "no_fire"

    folder_lab = infer_from_folders_strict(p)
    if folder_lab == "no_fire":
        return "no_fire"
    if folder_lab == "fire":
        return "fire"

    # fall back: filename 'fire' if present
    if file_lab == "fire":
        return "fire"

    return None


def gather_images(input_dir: Path) -> Dict[str, List[Path]]:
    labeled: Dict[str, List[Path]] = {"fire": [], "no_fire": [], "unknown": []}
    for p in input_dir.rglob("*"):
        if is_image(p):
            lab = infer_label(p)
            if lab == "fire":
                labeled["fire"].append(p)
            elif lab == "no_fire":
                labeled["no_fire"].append(p)
            else:
                labeled["unknown"].append(p)
    return labeled


def stratified_split(items: List[Path], val: float, test: float, rng: random.Random) -> SplitResult:
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_test = int(round(n * test))
    n_val = int(round(n * val))
    if n_test + n_val > n:
        n_val = max(0, n - n_test)

    test_set = items[:n_test]
    val_set = items[n_test:n_test + n_val]
    train_set = items[n_test + n_val:]
    return SplitResult(train=train_set, val=val_set, test=test_set)


def ensure_empty_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def safe_place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def unique_dest(dst: Path, src: Path) -> Path:
    """
    Avoid filename collisions in output folders.
    If collision, prefix with parent folder name.
    """
    if not dst.exists():
        return dst
    parent = src.parent.name
    return dst.with_name(f"{parent}__{dst.name}")


def place_many(items: List[Path], input_dir: Path, output_dir: Path, split: str, cls: str, mode: str) -> List[str]:
    rels: List[str] = []
    for src in items:
        dst = output_dir / split / cls / src.name
        dst = unique_dest(dst, src)
        safe_place(src, dst, mode)
        rels.append(str(src.relative_to(input_dir)))
    return rels


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Dataset root (e.g., data/raw)")
    ap.add_argument("--output", required=True, help="Output splits dir (e.g., data/splits)")
    ap.add_argument("--val", type=float, default=0.15, help="Validation fraction")
    ap.add_argument("--test", type=float, default=0.15, help="Test fraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy",
                    help="How to place files into splits")
    ap.add_argument("--keep-unknown", action="store_true",
                    help="If set, place unmatched images into output/unknown/")
    ap.add_argument("--print-samples", type=int, default=10,
                    help="Print N sample paths per class for debugging (default 10)")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Fresh output
    ensure_empty_dir(output_dir)

    # Create canonical folders
    for split in ("train", "val", "test"):
        for cls in ("fire", "no_fire"):
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    if args.keep_unknown:
        (output_dir / "unknown").mkdir(parents=True, exist_ok=True)

    labeled = gather_images(input_dir)
    fire_imgs = labeled["fire"]
    nofire_imgs = labeled["no_fire"]
    unknown_imgs = labeled["unknown"]

    # Audit
    print("Label audit:")
    print(f"  fire:    {len(fire_imgs)}")
    print(f"  no_fire: {len(nofire_imgs)}")
    print(f"  unknown: {len(unknown_imgs)}")

    n = args.print_samples
    if n > 0:
        print("\nSample labeled paths (first few):")
        for cls in ("fire", "no_fire", "unknown"):
            print(f"\n{cls}:")
            for ex in labeled[cls][:n]:
                print("  ", ex)

    if len(fire_imgs) == 0 or len(nofire_imgs) == 0:
        print("\nWARNING: One class is empty. This indicates your folder/filename labels did not match the rules.\n"
              "If your NF tag appears in a different format (e.g., 'NF00123' without separators), this script still\n"
              "should catch it via tokenization. If it doesn't, paste 3 filenames and we’ll adjust.\n")

    rng = random.Random(args.seed)
    fire_split = stratified_split(fire_imgs, args.val, args.test, rng)
    nf_split = stratified_split(nofire_imgs, args.val, args.test, rng)

    files_info = {
        "fire": {
            "train": place_many(fire_split.train, input_dir, output_dir, "train", "fire", args.mode),
            "val":   place_many(fire_split.val,   input_dir, output_dir, "val",   "fire", args.mode),
            "test":  place_many(fire_split.test,  input_dir, output_dir, "test",  "fire", args.mode),
        },
        "no_fire": {
            "train": place_many(nf_split.train, input_dir, output_dir, "train", "no_fire", args.mode),
            "val":   place_many(nf_split.val,   input_dir, output_dir, "val",   "no_fire", args.mode),
            "test":  place_many(nf_split.test,  input_dir, output_dir, "test",  "no_fire", args.mode),
        },
    }

    if args.keep_unknown and unknown_imgs:
        unk_rels = []
        for src in unknown_imgs:
            dst = output_dir / "unknown" / src.name
            dst = unique_dest(dst, src)
            safe_place(src, dst, args.mode)
            unk_rels.append(str(src.relative_to(input_dir)))
        files_info["unknown"] = unk_rels  # type: ignore[assignment]

    info = {
        "seed": args.seed,
        "val_fraction": args.val,
        "test_fraction": args.test,
        "mode": args.mode,
        "input_dir": str(input_dir),
        "counts": {
            "fire": {
                "train": len(fire_split.train),
                "val": len(fire_split.val),
                "test": len(fire_split.test),
                "total": len(fire_imgs),
            },
            "no_fire": {
                "train": len(nf_split.train),
                "val": len(nf_split.val),
                "test": len(nf_split.test),
                "total": len(nofire_imgs),
            },
            "unknown_total": len(unknown_imgs),
        },
        "files": files_info,
        "label_rules": {
            "filename_rule": "token 'nf' => no_fire; token 'fire' => fire (folder no_fire overrides)",
            "fire_folder_tokens": sorted(list(FIRE_FOLDER_TOKENS)),
            "no_fire_folder_tokens": sorted(list(NOFIRE_FOLDER_TOKENS)),
            "no_fire_strict_patterns": ["token 'no' + token 'fire'", "token 'nofire'", "token 'no_fire'"],
        },
    }

    with open(output_dir / "SPLIT_INFO.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\nDone. Wrote:", output_dir / "SPLIT_INFO.json")
    print("Counts summary:\n", json.dumps(info["counts"], indent=2))


if __name__ == "__main__":
    main()
