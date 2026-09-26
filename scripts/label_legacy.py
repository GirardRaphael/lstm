"""Mark committed v1 educational artifacts; leave original metric values intact."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICE = ("> **Archived v1 exploratory evidence.** Preprocessing includes validation contamination; "
          "some runs bridge gaps or use future-informed weather fills. Test results informed historical choices. "
          "These scores do not validate a deployed product. See the current `reports/REPORT.md`.\n\n")


def main():
    report = ROOT / "reports/REPORT.md"
    archive = ROOT / "reports/LEGACY_REPORT.md"
    if not archive.exists():
        archive.write_text(NOTICE + report.read_text(encoding="utf-8"), encoding="utf-8")
    for path in (ROOT / "obsidian_vault/Traffic_LSTM_Brain").glob("*.md"):
        content = path.read_text(encoding="utf-8")
        if content.startswith("# Presentation script: hourly forecasting, evidence first"):
            continue
        if "Archived v1 exploratory evidence" not in content:
            if content.startswith("---\n"):
                end = content.find("\n---", 4) + 4
                content = content[:end] + "\n\n" + NOTICE + content[end:].lstrip()
            else:
                content = NOTICE + content
            path.write_text(content, encoding="utf-8")
    for name in ("model_comparison.json", "window_sweep.json"):
        path = ROOT / "reports" / name
        data = json.loads(path.read_text(encoding="utf-8"))
        aborted = "window" in name and all(row.get("lstm_epochs", 0) <= 2 for row in data["rows"])
        data["evidence_status"] = "aborted_two_epoch_run_not_for_selection" if aborted else "v1_exploratory"
        data["pipeline_version"] = "v1"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for path in (ROOT / "reports/web").glob("*.html"):
        content = path.read_text(encoding="utf-8")
        if "Archived v1 exploratory evidence" not in content:
            banner = '<aside style="padding:20px;background:#fff0cc;color:#222">' \
                '<strong>Archived v1 exploratory evidence.</strong> These historical scores have preprocessing and '
            banner += 'model-selection limitations. They do not establish production readiness. See reports/REPORT.md.</aside>'
            anchor = '<div class="wrap">'
            content = content.replace(anchor, anchor + banner, 1)
        content = content.replace("sensors → hourly history → model → forecast → controller → adaptive lights",
                                  "stored hourly observations → forecast → offline evaluation")
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
