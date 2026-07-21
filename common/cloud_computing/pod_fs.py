from pathlib import Path

class PodFileSystem:
    def __init__(self):
        self.root: Path = Path("/app")
        self.data_dir: Path = Path("/app/dataset")
        self.run_dir: Path = Path("/app/runs")
        self.log_path: Path = Path("/app/training.log")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)