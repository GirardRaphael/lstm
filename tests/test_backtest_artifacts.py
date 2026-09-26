import json
from pathlib import Path
import pandas as pd
import pytest


@pytest.mark.parametrize("dataset",["motorway","bikes"])
def test_recorded_fold_calibration_separation_and_counts(dataset):
    root=Path(__file__).resolve().parents[1]/f"reports/backtests/{dataset}_20260925"
    artifact=json.loads((root/"results.json").read_text())
    assert len(artifact["runs"])==9
    assert {r["seed"] for r in artifact["runs"]}=={7,42,123}
    fold_ranges=[]
    for run in artifact["runs"]:
        frame=pd.read_csv(root/f"fold{run['fold']}_seed{run['seed']}.csv",parse_dates=["timestamp"])
        calibration=frame[frame.partition=="calibration"]
        evaluation=frame[frame.partition=="evaluation"]
        assert calibration.timestamp.max()<evaluation.timestamp.min()
        assert run["selected"]==min(run["validation"],key=lambda name:run["validation"][name]["mae"])
        assert {r["metrics"]["n"] for r in run["evaluation"].values()}=={len(evaluation)}
        assert {r["calibration"]["n"] for r in run["evaluation"].values()}=={len(calibration)}
        if run["seed"]==7:
            fold_ranges.append((evaluation.timestamp.min(),evaluation.timestamp.max()))
    assert all(left[1]<right[0] for left,right in zip(fold_ranges,fold_ranges[1:]))
