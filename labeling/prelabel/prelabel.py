#!/usr/bin/env python3
"""Run a labeler function on every .jpg in <base_dir>/images and store the result
in <base_dir>/labels/<name>.json.

The labeler is a function with the signature:

    def label(image_path: Path, label: dict) -> dict

It receives the path of one image and the current label (an empty dict if no
label file exists yet) and returns the updated label.

Usage:
    python prelabel.py BASE_DIR --labeler path/to/script.py
    python prelabel.py BASE_DIR --labeler path/to/script.py:my_function
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

DEFAULT_FUNCTION = "label"


def load_labeler(spec: str) -> Callable[[Path, dict], dict]:
    """Load a function from a script given as 'path/to/script.py[:function_name]'."""
    path_str, _, func_name = spec.partition(":")
    func_name = func_name or DEFAULT_FUNCTION

    script_path = Path(path_str)
    if not script_path.is_file():
        sys.exit(f"Error: labeler script '{script_path}' does not exist.")

    module_spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    func = getattr(module, func_name, None)
    if not callable(func):
        sys.exit(f"Error: '{script_path}' has no function '{func_name}'.")
    return func


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_dir", type=Path, help="Directory containing the 'images' subfolder")
    parser.add_argument(
        "--labeler",
        required=True,
        help=f"Labeler as 'script.py[:function]' (default function: '{DEFAULT_FUNCTION}')",
    )
    args = parser.parse_args()

    images_dir = args.base_dir / "images"
    labels_dir = args.base_dir / "labels"

    if not images_dir.is_dir():
        sys.exit(f"Error: '{images_dir}' does not exist.")

    labeler = load_labeler(args.labeler)
    labels_dir.mkdir(exist_ok=True)

    done, failed = 0, 0
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() != ".jpg":
            continue

        label_path = labels_dir / f"{image_path.stem}.json"
        try:
            # Start from the existing label so fields from other labelers are kept
            if label_path.exists():
                with open(label_path, encoding="utf-8") as f:
                    label = json.load(f)
            else:
                label = {}

            label = labeler(image_path, label)
            if not isinstance(label, dict):
                raise TypeError(f"labeler must return a dict, got {type(label).__name__}")

            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(label, f, indent=2)
            done += 1
        except Exception as e:
            # Keep going so one bad image does not abort a long run
            print(f"Failed on {image_path.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"Labeled {done} images, {failed} failed.")


if __name__ == "__main__":
    main()