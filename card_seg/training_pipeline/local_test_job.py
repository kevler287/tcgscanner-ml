"""
Local smoke test for the training handler — runs Extract → Train → Evaluate → Load
without touching RunPod. Useful to verify the full chain works end-to-end before
deploying to RunPod Serverless.

Usage:
    EPOCHS=1 python test_handler_local.py --dataset-version v1 --model-version v1-test
"""
import argparse
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Override epochs BEFORE importing anything that reads CONFIG at import time.
os.environ.setdefault("TEST_EPOCHS", "1")
os.environ.setdefault("WORKSPACE_DIR", str(Path(tempfile.gettempdir()) / "card_seg_local_test"))

from dataclasses import replace
from card_seg.config import CONFIG

# Patch the frozen yolo_seg config with a 1-epoch override for this test run.
CONFIG.yolo_seg = replace(CONFIG.yolo_seg, epochs=int(os.environ["TEST_EPOCHS"]))

from card_seg.training_pipeline.training_job import handler


def main():
    parser = argparse.ArgumentParser(description="Local handler smoke test")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()

    job = {
        "input": {
            "dataset_version": args.dataset_version,
            "model_version": args.model_version,
        }
    }

    result = handler(job)
    print("\n=== HANDLER RESULT ===")
    print(result)

    if result.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()