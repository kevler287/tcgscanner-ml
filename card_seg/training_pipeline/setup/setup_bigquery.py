import logging
from google.cloud import bigquery
import dotenv
from card_seg.config import CONFIG

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_RUNS_SCHEMA = [
    bigquery.SchemaField("model_prefix",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("model_version",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("timestamp",        "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("class_names",      "STRING",    mode="REPEATED"),
    bigquery.SchemaField("dataset_version",      "STRING",   mode="NULLABLE"),
    bigquery.SchemaField("dataset_size",     "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("samples_per_card", "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("train_split",      "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("val_split",        "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("test_split",       "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("empty_split",      "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("pretrained_model", "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("epochs_planned",   "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("batch_size",       "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("learning_rate",    "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("optimizer",        "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("amp",              "BOOL",      mode="NULLABLE"),
    bigquery.SchemaField("test_precision",   "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("test_recall",      "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("test_map50",       "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("test_map50_95",    "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("min_iou",          "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("max_iou",          "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("avg_iou",          "FLOAT",     mode="NULLABLE"),
]

TRAINING_EPOCHS_SCHEMA = [
    bigquery.SchemaField("model_prefix",     "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("model_version",  "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("epoch",          "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("train_box_loss", "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("train_seg_loss", "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("train_cls_loss", "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("train_dfl_loss", "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_box_loss",   "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_seg_loss",   "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_cls_loss",   "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_dfl_loss",   "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_precision",  "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_recall",     "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_map50",      "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("val_map50_95",   "FLOAT",   mode="NULLABLE"),
]


def create_dataset(client: bigquery.Client):
    dataset_id = f"{client.project}.{CONFIG.model_results_dataset.name}"
    dataset    = bigquery.Dataset(dataset_id)
    dataset.location = "EU"

    try:
        client.create_dataset(dataset, exists_ok=False)
        logger.info("Created dataset: %s", dataset_id)
    except Exception as e:
        logger.error("Failed to create dataset: %s", e)


def create_table(client: bigquery.Client, table_name: str, schema: list):
    table_id = f"{client.project}.{CONFIG.model_results_dataset.name}.{table_name}"
    table    = bigquery.Table(table_id, schema=schema)

    try:
        client.create_table(table, exists_ok=False)
        logger.info("Created table: %s", table_id)
    except Exception as e:
        logger.error("Failed to create table %s: %s", table_name, e)


def setup():
    client = bigquery.Client()

    create_dataset(client)
    create_table(client, CONFIG.model_results_dataset.model_runs_table,       MODEL_RUNS_SCHEMA)
    create_table(client, CONFIG.model_results_dataset.training_epoch_table,  TRAINING_EPOCHS_SCHEMA)

    logger.info("BigQuery setup complete.")


if __name__ == "__main__":
    setup()