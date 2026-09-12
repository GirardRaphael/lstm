"""Central configuration for the whole pipeline.

Everything that a reader (or a teacher) might reasonably ask "why that value?"
about lives here, in one place, with the reasoning attached.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

# Project layout -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
VAULT_DIR = PROJECT_ROOT / "obsidian_vault" / "Traffic_LSTM_Brain"

DEFAULT_DATASET = RAW_DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"


@dataclass
class TrainingConfig:
    """Hyper-parameters and data contract for one training run."""

    # --- Data contract ---
    data_path: Path = DEFAULT_DATASET
    datetime_column: str = "date_time"
    target_column: str = "traffic_volume"

    # --- Extra input features ---
    # Empty tuple + False reproduces the univariate baseline: past traffic only.
    # See `features.py`. The target always stays column 0 of the matrix.
    exogenous_columns: Sequence[str] = ()
    calendar_features: bool = False

    # --- Sequence construction ---
    # 24 = one full day. The traffic cycle is daily (morning peak, evening
    # peak, night trough), so a 24-hour window lets the network see a complete
    # cycle before it has to predict.
    sequence_length: int = 24

    # How many hours ahead to predict. (1,) reproduces the classic
    # "next hour" task; (1, 3, 6) trains one model that outputs all three.
    horizons: Sequence[int] = (1,)

    # --- Architecture ---
    lstm_units: Sequence[int] = (64, 32)
    dense_units: int = 16
    dropout: float = 0.2

    # --- Optimisation ---
    learning_rate: float = 1e-3
    epochs: int = 50          # upper bound; EarlyStopping usually stops sooner
    batch_size: int = 32
    validation_split: float = 0.2
    patience: int = 5
    seed: int = 42

    # Sliding windows are built over the rows that exist. When the series has
    # gaps (the Metro dataset has 2,588 of them, including a 308-day outage),
    # some windows silently span a jump in time: "the last 24 hours" is then
    # really 24 readings taken across a much longer period.
    #
    # Leaving this False reproduces the documented baseline run. Setting it
    # True drops every window that crosses a gap - more correct, at the cost of
    # ~29% of the training windows on this dataset. Retrain after changing it.
    drop_gapped_windows: bool = False

    # --- Split ---
    # Chronological, never shuffled: we always train on the past and test on
    # the future, which is the only split that matches how the model would be
    # used in production.
    train_ratio: float = 0.8

    # --- Traffic level thresholds (vehicles per hour) ---
    level_low: int = 2000
    level_high: int = 4000

    run_name: str = "baseline_univariate"

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.horizons = tuple(int(h) for h in self.horizons)
        self.lstm_units = tuple(int(u) for u in self.lstm_units)
        self.exogenous_columns = tuple(self.exogenous_columns)
        if min(self.horizons) < 1:
            raise ValueError("horizons must be >= 1")
        if not 0.5 <= self.train_ratio < 1.0:
            raise ValueError("train_ratio must be in [0.5, 1.0)")

    # -- convenience ---------------------------------------------------------
    @property
    def n_outputs(self) -> int:
        return len(self.horizons)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / f"{self.run_name}.keras"

    @property
    def artifact_path(self) -> Path:
        return MODEL_DIR / f"{self.run_name}_artifacts.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_path"] = str(self.data_path)
        d["horizons"] = list(self.horizons)
        d["lstm_units"] = list(self.lstm_units)
        d["exogenous_columns"] = list(self.exogenous_columns)
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# Ready-made presets for the comparison runs.
MULTI_HORIZON = TrainingConfig(horizons=(1, 3, 6), run_name="multi_horizon")

GAP_GUARDED = TrainingConfig(drop_gapped_windows=True, run_name="gap_guarded")

MULTIVARIATE = TrainingConfig(
    exogenous_columns=("temp", "rain_1h", "snow_1h", "clouds_all"),
    calendar_features=True,
    run_name="multivariate",
)
