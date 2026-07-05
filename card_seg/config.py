from dataclasses import dataclass, field
import json
import os
from config.project_config import ProjectConfig

_overrides = json.loads(os.getenv("YOLO_CONFIG", "{}"))

@dataclass
class TrainingFlowConfig:
    pretrained_model: str   = _overrides.get("pretrained_model", "yolo11n-seg.pt")
    epochs:           int   = _overrides.get("epochs", 300)
    imgsz:            int   = _overrides.get("imgsz", 640)
    batch:            int   = _overrides.get("batch", 16)
    device:           int   = _overrides.get("device", 0)
    amp:              bool  = _overrides.get("amp", False)
    optimizer:        str   = _overrides.get("optimizer", "Adam")
    lr0:              float = _overrides.get("lr0", 0.001)
    fraction:         float = _overrides.get("fraction", 1.0)


@dataclass
class DataFlowConfig:
    output_dir:       str   = "card_seg/data_pipeline/output"
    background_size:  tuple = (600, 800)
    max_angle_deg:    int   = 10
    samples_per_card: int   = 3
    val_split:        float = 0.05
    test_split:       float = 0.05
    empty_split:      float = 0.0


@dataclass
class CardSegConfig(ProjectConfig):
    model_prefix: str = "card_seg"
    pf_ygo_cards: str = ""
    pf_bg: str = ""
    yolo_seg:    TrainingFlowConfig   = field(default_factory=TrainingFlowConfig)
    transform:   DataFlowConfig = field(default_factory=DataFlowConfig)

    def __post_init__(self):
        self.bucket.pf_datasets = self.bucket.pf_datasets + self.model_prefix + "/"
        self.bucket.pf_models   = self.bucket.pf_models   + self.model_prefix + "/"
        self.pf_ygo_cards = self.bucket.pf_raw + "ygo_cards/"
        self.pf_bg = self.bucket.pf_raw + "backgrounds/"

CONFIG = CardSegConfig()