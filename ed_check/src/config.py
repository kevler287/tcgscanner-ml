from dataclasses import dataclass, field
from common.config import ProjectConfig

@dataclass
class DefaultTrainConfig:
    imgsz: tuple = (64, 192)
    epochs:           int   = 3
    batch:            int   = 16
    num_workers: int = 0
    lr0:              float = 0.001

@dataclass
class TransformConfig:
    samples_per_type: int = 100
    crop_margin: float = 0.2
    edition_0 = [[0.05, 0.7], [0.3, 0.76]]
    edition_1 = [[0.177, 0.9375], [0.38, 0.99]]
    val_split:        float = 0.1


@dataclass
class EditionCheckerConfig(ProjectConfig):
    model_prefix: str = "ed_check"
    pf_ed_types: str = ""
    pf_test_data: str = ""
    train_cfg:    DefaultTrainConfig   = field(default_factory=DefaultTrainConfig)
    transform_cfg:   TransformConfig = field(default_factory=TransformConfig)

    def __post_init__(self):
        self.bucket.pf_datasets = self.bucket.pf_datasets + self.model_prefix + "/"
        self.pf_test_data = self.bucket.pf_datasets + "test_data/"
        self.bucket.pf_models   = self.bucket.pf_models   + self.model_prefix + "/"
        self.pf_ed_types = self.bucket.pf_raw + "ygo_ed_types/"
        self.bucket.pf_logs = self.bucket.pf_logs + self.model_prefix + "/"

CONFIG = EditionCheckerConfig()