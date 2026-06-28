# TCG Scanner ML

Machine learning pipeline for [TCG Scanner](https://github.com/kevler287/tcgscanner) — handles synthetic training data generation, model training, evaluation, and artifact management via a fully orchestrated **Prefect** flow backed by **Google Cloud**.

> **Note:** Currently supports Yu-Gi-Oh! card segmentation. Additional models (e.g. rarity detection) and other TCGs will be added in future releases.

**Tech Stack:** Python · PyTorch · YOLOv11 · OpenCV · Prefect · Google Cloud Storage · BigQuery · CUDA

## Architecture

```
GCS (raw cards + backgrounds)
        ↓  Extract
    local cache
        ↓  Transform
    synthetic YOLO dataset (perspective warp, glare, blur, augmentation)
        ↓  Train
    YOLO segmentation model
        ↓  Evaluate
    IoU + mAP metrics on test set
        ↓  Load
    best.pt → GCS · training results → BigQuery
```

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA
- Google Cloud account with GCS and BigQuery enabled
- Prefect Cloud account

## Setup

```bash
pip install -e .
```

Configure credentials:
create .env file and fill in GOOGLE_APPLICATION_CREDENTIALS, LOCAL_CARDS_DIR, LOCAL_BG_DIR

Set up BigQuery tables:
```bash
python data_platform/setup/setup_bigquery.py
```

Upload raw data to GCS (one-time):
```bash
python data_platform/setup/upload_raw.py
```

## Usage

```bash
python data_platform/flows/training_flow.py \
  --model-version v1 \
  --model-type segmentation \
  --epochs 50
```

## Model Versioning

Releases are tagged per model type:

```
seg/v1    ← Yu-Gi-Oh card segmentation v1
seg/v2
foo/v1    ← other model
```

## Related

- [tcgscanner](https://github.com/kevler287/tcgscanner) — OCR service and client