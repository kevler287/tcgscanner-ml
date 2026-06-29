from abc import ABC
from dataclasses import dataclass, field

@dataclass
class TCGBucket:
    name:                str = "tcg-image-data"
    pf_raw:              str = "raw/"
    pf_datasets:         str = "datasets/"
    pf_models:           str = "models/"

@dataclass
class ModelResultDataset:
    name:                 str = "model_results"
    model_runs_table:     str = "model_runs"
    training_epoch_table: str = "training_epochs"

@dataclass
class ProjectConfig(ABC):
    random_seed: int                          = 42
    bucket:                TCGBucket          = field(default_factory=TCGBucket)
    model_results_dataset: ModelResultDataset = field(default_factory=ModelResultDataset)