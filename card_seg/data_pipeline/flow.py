from prefect import flow, task

@task(name="Extract")
def extract():
    """Download cards and backgrounds from GCS to local cache."""
    from card_seg.data_pipeline.tasks.extract import main as run_extract
    run_extract()


@task(name="Transform")
def transform():
    """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
    from card_seg.data_pipeline.tasks.transform import main as run_transform
    run_transform()


@task(name="Load")
def load(dataset_version: str):
    """Upload best.pt to GCS and write model results to BigQuery."""
    from card_seg.data_pipeline.tasks.load import load as run_load
    run_load(dataset_version)


@flow(name="Card Segmentation Data Pipeline")
def data_pipeline(dataset_version: str):
    extract()
    transform()
    load(dataset_version)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    data_pipeline(dataset_version=args.version)