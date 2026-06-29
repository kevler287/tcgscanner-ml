# TCG Scanner ML

Machine learning pipeline for [TCG Scanner](https://github.com/kevler287/tcgscanner) — handles synthetic training data generation, model training, evaluation, and artifact management via a fully orchestrated **Prefect** flow backed by **Google Cloud**.

> **Note:** Currently supports Yu-Gi-Oh! card segmentation. Additional models (e.g. rarity detection) and other TCGs will be added in future releases.

**Tech Stack:** Python · PyTorch · YOLOv11 · OpenCV · Prefect · Google Cloud Storage · BigQuery · CUDA

---

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (training pipeline)
- Google Cloud account with GCS and BigQuery enabled
- Prefect Cloud account

---

## Setup

```bash
pip install -e .
```

Configure credentials — create a `.env` file:

```
GOOGLE_APPLICATION_CREDENTIALS=...
LOCAL_CARDS_DIR=...
LOCAL_BG_DIR=...
```

One-time setup:

```bash
# Upload raw data to GCS
python card_seg/data_pipeline/setup/upload_raw_data.py
```

---

## Usage

**Data Pipeline** — generate and upload a synthetic dataset:

```bash
python card_seg/data_pipeline/flow.py --dataset-version v1
```

---

## Architecture

### Data Pipeline

```
GCS (raw cards + backgrounds)
        ↓  Extract
    local cache
        ↓  Transform
    synthetic YOLO dataset (perspective warp, glare, blur, augmentation)
        ↓  Load
    dataset.zip → GCS (datasets/card_seg/v1.zip)
```

### Training Pipeline

> **Note:** Work in progress

```
GCS (dataset.zip)
        ↓  Extract
    local cache
        ↓  Train  [RunPod GPU]
    YOLO segmentation model
        ↓  Evaluate
    mAP50 · mAP50-95 · IoU metrics
        ↓  Load
    best.pt + last.pt → GCS (models/card_seg/v1-1/)
    training metrics   → BigQuery (ml_tracking.training_runs)
```

## Versioning

Dataset and model versions are managed independently and linked via BigQuery:

```
datasets/card_seg/v1.zip      ← dataset version
models/card_seg/v1-1/         ← trained on dataset v1, run 1
models/card_seg/v1-2/         ← trained on dataset v1, run 2 (different hyperparams)
models/card_seg/v2-1/         ← trained on dataset v2
```

---

## Related

- [tcgscanner](https://github.com/kevler287/tcgscanner) — OCR service and client