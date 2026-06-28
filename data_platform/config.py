from dataclasses import dataclass, field


@dataclass(frozen=True)
class TCGBucket:
    name:               str = "tcg-image-data"
    ygo_prefix:         str = "raw/ygo_cards/"
    background_prefix:  str = "raw/backgrounds/"
    seg_models_prefix:  str = "models/segmentation/"


@dataclass(frozen=True)
class ModelResultDataset:
    name:                 str = "model_results"
    model_runs_table:     str = "model_runs"
    training_epoch_table: str = "training_epochs"


@dataclass(frozen=True)
class YoloSegConfig:
    pretrained_model: str   = "yolo11n-seg.pt"
    epochs:           int   = 300
    imgsz:            int   = 640
    batch:            int   = 16
    device:           int   = 0
    amp:              bool  = False
    optimizer:        str   = "Adam"
    lr0:              float = 0.001

@dataclass(frozen=True)
class TransformConfig:
    output_dir: str = "data_platform/etl/output"
    background_size:  tuple = (600, 800)
    max_angle_deg:    int   = 10
    samples_per_card: int   = 3
    val_split:        float = 0.05
    test_split:       float = 0.05
    empty_split:      float = 0.0


@dataclass(frozen=True)
class ETLConfig:
    random_seed:          int              = 42
    yolo_seg:             YoloSegConfig    = field(default_factory=YoloSegConfig)
    transform:            TransformConfig  = field(default_factory=TransformConfig)
    bucket:               TCGBucket        = field(default_factory=TCGBucket)
    model_results_dataset: ModelResultDataset = field(default_factory=ModelResultDataset)


CONFIG = ETLConfig()