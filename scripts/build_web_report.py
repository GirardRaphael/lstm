"""Inline every figure into the web report so the page is self-contained.

`reports/web/results.src.html` keeps `{{IMG:name.png}}` tokens instead of file
references, because a published page cannot fetch anything from disk or from
another host. This swaps each token for a base64 data URI and writes
`reports/web/results.html`.

    python scripts/build_web_report.py
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "web" / "results.src.html"
OUT = ROOT / "reports" / "web" / "results.html"
FIGURES = ROOT / "reports" / "figures"

# Alt text is written by hand: a screen reader should get the finding, not the
# filename. Anything not listed here falls back to its filename and is flagged.
ALT = {
    "02_daily_profile.png":
        "Line chart of average traffic volume by hour of day with a shaded "
        "standard-deviation band, showing a morning peak, an evening peak and "
        "an overnight trough.",
    "04_actual_vs_predicted.png":
        "Time-series chart comparing actual and predicted traffic over eight "
        "days of the test set; the dashed prediction line follows the daily "
        "cycle closely.",
    "11_horizons.png":
        "Line chart of mean absolute error against forecast horizon, rising "
        "from 245 at +1 hour to 472 at +6 hours.",
    "06_heatmap_lstm_1.png":
        "Heatmap of the hidden state of all 64 first-layer LSTM units across "
        "the 24 hours of the input window, in diverging red and blue.",
    "08_input_sensitivity.png":
        "Bar chart of how much each of the 24 input hours moves the "
        "prediction; the two most recent hours dominate, with a second "
        "cluster around 11 to 14 hours back.",
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit("Missing template: {}".format(SRC))

    html = SRC.read_text(encoding="utf-8")
    missing, unlabelled = [], []

    def swap(match: re.Match) -> str:
        name = match.group(1)
        path = FIGURES / name
        if not path.exists():
            missing.append(name)
            return ""
        if name not in ALT:
            unlabelled.append(name)
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return '<img src="data:image/png;base64,{}" alt="{}" loading="lazy">'.format(
            payload, ALT.get(name, name))

    html = re.sub(r"\{\{IMG:([^}]+)\}\}", swap, html)

    if missing:
        raise SystemExit(
            "Figures not found in {} - run the training first:\n  {}".format(
                FIGURES, "\n  ".join(missing)))
    if "{{IMG:" in html:
        raise SystemExit("A token was left unsubstituted.")
    for name in unlabelled:
        print("  warning: no alt text written for {}".format(name))

    OUT.write_text(html, encoding="utf-8")
    print("Wrote {} ({:.2f} MB, {} images inlined)".format(
        OUT, OUT.stat().st_size / 1_048_576, html.count("data:image/png;base64,")))


if __name__ == "__main__":
    main()
