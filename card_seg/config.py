from dataclasses import dataclass, field
from config.project_config import ProjectConfig

@dataclass
class YoloSegConfig:
    pretrained_model: str   = "yolo11n-seg.pt"
    epochs:           int   = 300
    imgsz:            int   = 640
    batch:            int   = 16
    device:           int   = 0
    amp:              bool  = False
    optimizer:        str   = "Adam"
    lr0:              float = 0.001


@dataclass
class TransformConfig:
    output_dir:       str   = "card_seg/data_pipeline/output"
    background_size:  tuple = (600, 800)
    max_angle_deg:    int   = 10
    samples_per_card: int   = 3
    val_split:        float = 0.05
    test_split:       float = 0.05
    empty_split:      float = 0.0


@dataclass
class CardSegConfig(ProjectConfig):
    pf_ygo_cards: str = ""
    pf_bg: str = ""
    yolo_seg:    YoloSegConfig   = field(default_factory=YoloSegConfig)
    transform:   TransformConfig = field(default_factory=TransformConfig)

    def __post_init__(self):
        self.bucket.pf_datasets = self.bucket.pf_datasets + "card_seg/"
        self.bucket.pf_models   = self.bucket.pf_models   + "card_seg/"
        self.pf_ygo_cards = self.bucket.pf_raw + "ygo_cards/"
        self.pf_bg = self.bucket.pf_raw + "backgrounds/"

CONFIG = CardSegConfig()