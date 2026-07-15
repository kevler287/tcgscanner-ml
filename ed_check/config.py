from dataclasses import dataclass, field
from config.project_config import ProjectConfig

@dataclass
class DefaultTrainConfig:
    epochs:           int   = 300
    batch:            int   = 16
    device:           int   = 0
    amp:              bool  = True
    optimizer:        str   = "Adam"
    lr0:              float = 0.001

@dataclass
class TransformConfig:
    output_dir:       str   = "ed_check/data_pipeline/v1"
    samples_per_type: int = 100
    crop_margin: float = 0.2
    edition_0 = [[0.05, 0.7], [0.3, 0.76]]
    edition_1 = [[0.177, 0.9375], [0.38, 0.99]]


@dataclass
class EditionCheckerConfig(ProjectConfig):
    model_prefix: str = "ed_check"
    pf_ed_types: str = ""
    train_cfg:    DefaultTrainConfig   = field(default_factory=DefaultTrainConfig)
    transform_cfg:   TransformConfig = field(default_factory=TransformConfig)

    def __post_init__(self):
        self.bucket.pf_datasets = self.bucket.pf_datasets + self.model_prefix + "/"
        self.bucket.pf_models   = self.bucket.pf_models   + self.model_prefix + "/"
        self.pf_ed_types = self.bucket.pf_raw + "ygo_ed_types/"

CONFIG = EditionCheckerConfig()