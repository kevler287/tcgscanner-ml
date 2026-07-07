from dataclasses import dataclass, field
from config.project_config import ProjectConfig

@dataclass
class DefaultTrainConfig:
    epochs:           int   = 300
    imgsz:            int   = 640
    batch:            int   = 16
    device:           int   = 0
    amp:              bool  = True
    optimizer:        str   = "Adam"
    lr0:              float = 0.001

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
    yolo_seg:    DefaultTrainConfig   = field(default_factory=DefaultTrainConfig)
    transform:   DataFlowConfig = field(default_factory=DataFlowConfig)

    def __post_init__(self):
        self.bucket.pf_datasets = self.bucket.pf_datasets + self.model_prefix + "/"
        self.bucket.pf_models   = self.bucket.pf_models   + self.model_prefix + "/"
        self.pf_ygo_cards = self.bucket.pf_raw + "ygo_cards/"
        self.pf_bg = self.bucket.pf_raw + "backgrounds/"

CONFIG = CardSegConfig()