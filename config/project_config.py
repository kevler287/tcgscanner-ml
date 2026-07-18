from abc import ABC
from dataclasses import dataclass, field

@dataclass
class TCGBucket:
    name:                str = "tcg-image-data"
    pf_raw:              str = "raw/"
    pf_datasets:         str = "datasets/"
    pf_models:           str = "models/"

@dataclass
class ProjectConfig(ABC):
    random_seed: int                          = 42
    bucket:                TCGBucket          = field(default_factory=TCGBucket)
    bq_model_runs:     str = "model_runs"
    bq_training_epochs: str = "training_epochs"