import logging
import traceback
from pathlib import Path

import runpod

from card_seg.training_pipeline.tasks.extract import extract
from card_seg.training_pipeline.tasks.train import train
from card_seg.training_pipeline.tasks.evaluate import evaluate
from card_seg.training_pipeline.tasks.load import load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def count_dataset_images(data_yaml: str) -> int:
    dataset_dir = Path(data_yaml).parent
    return sum(
        1
        for split in ("train", "val", "test")
        for _ in (dataset_dir / "images" / split).glob("*.jpg")
    )


def handler(job):
    job_input = job.get("input", {})

    dataset_version = job_input.get("dataset_version")
    model_version = job_input.get("model_version")

    if not dataset_version or not model_version:
        return {
            "status": "failed",
            "step": "validation",
            "error": "Both 'dataset_version' and 'model_version' are required.",
        }

    try:
        logger.info("=== Step 1/4: Extract ===")
        data_yaml = extract(dataset_version)
        dataset_size = count_dataset_images(data_yaml)
    except Exception as e:
        return _error_response("extract", e)

    try:
        logger.info("=== Step 2/4: Train ===")
        train_result = train(data_yaml, run_name=model_version)
    except Exception as e:
        return _error_response("train", e)

    try:
        logger.info("=== Step 3/4: Evaluate ===")
        eval_metrics = evaluate(train_result["best_pt"], data_yaml)
    except Exception as e:
        return _error_response("evaluate", e)

    try:
        logger.info("=== Step 4/4: Load ===")
        weight_paths = load(
            model_version=model_version,
            dataset_version=dataset_version,
            dataset_size=dataset_size,
            best_pt=train_result["best_pt"],
            last_pt=train_result["last_pt"],
            epoch_metrics=train_result["epoch_metrics"],
            eval_metrics=eval_metrics,
        )
    except Exception as e:
        return _error_response("load", e)

    logger.info("=== Done ===")
    return {
        "status": "success",
        "model_version": model_version,
        "dataset_version": dataset_version,
        "dataset_size": dataset_size,
        "eval_metrics": eval_metrics,
        "weight_paths": weight_paths,
    }


def _error_response(step: str, exc: Exception) -> dict:
    logger.error("Step '%s' failed: %s", step, exc)
    return {
        "status": "failed",
        "step": step,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


runpod.serverless.start({"handler": handler})