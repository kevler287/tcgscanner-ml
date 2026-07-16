"""
Generic setup script for BigQuery datasets & tables.

Creates a dataset and, for every JSON schema file found in a given
directory, creates a table in that dataset. The table name is derived
from the filename (without .json).

Example directory structure for --schema-dir:
    schemas/
        model_runs.json
        training_epochs.json

Each *.json file contains a BigQuery schema in the standard format, e.g.:
    [
        {"name": "run_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "created_at", "type": "TIMESTAMP", "mode": "NULLABLE"}
    ]

Usage:
    python setup_bigquery.py --dataset model_results --schema-dir ./schemas
"""

import argparse
import logging
from dotenv import load_dotenv
from pathlib import Path

from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


def create_dataset(client: bigquery.Client, dataset_name: str, location: str) -> str:
    """Creates the dataset if it doesn't already exist. Returns the dataset_id."""
    dataset_id = f"{client.project}.{dataset_name}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = location

    try:
        client.create_dataset(dataset, exists_ok=False)
        logger.info("Created dataset: %s", dataset_id)
    except Exception as e:
        logger.error("Failed to create dataset %s: %s", dataset_id, e)

    return dataset_id


def create_table(client: bigquery.Client, dataset_id: str, table_name: str, schema_path: Path) -> None:
    """Creates a table based on a JSON schema file."""
    table_id = f"{dataset_id}.{table_name}"

    try:
        schema = client.schema_from_json(str(schema_path))
    except Exception as e:
        logger.error("Failed to load schema from %s: %s", schema_path, e)
        return

    table = bigquery.Table(table_id, schema=schema)

    try:
        client.create_table(table, exists_ok=False)
        logger.info("Created table: %s", table_id)
    except Exception as e:
        logger.error("Failed to create table %s: %s", table_id, e)


def discover_schemas(schema_dir: Path) -> list[Path]:
    """Finds all *.json schema files in schema_dir (non-recursive)."""
    if not schema_dir.is_dir():
        raise NotADirectoryError(f"Schema directory not found: {schema_dir}")

    schema_files = sorted(schema_dir.glob("*.json"))
    if not schema_files:
        raise FileNotFoundError(f"No .json schema files found in: {schema_dir}")

    return schema_files


def setup(dataset_name: str, schema_dir: Path, location: str) -> None:
    client = bigquery.Client()

    dataset_id = create_dataset(client, dataset_name, location)

    schema_files = discover_schemas(schema_dir)
    for schema_path in schema_files:
        table_name = schema_path.stem
        create_table(client, dataset_id, table_name, schema_path)

    logger.info("BigQuery setup complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Creates a BigQuery dataset and tables from JSON schema files."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name of the BigQuery dataset to create.",
    )
    parser.add_argument(
        "--schema-dir",
        required=True,
        type=Path,
        help="Path to the directory containing JSON schema files (one file per table, "
             "filename without .json = table name).",
    )
    parser.add_argument(
        "--location",
        default="EU",
        help="BigQuery dataset location (default: EU).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(
        dataset_name=args.dataset,
        schema_dir=args.schema_dir,
        location=args.location,
    )