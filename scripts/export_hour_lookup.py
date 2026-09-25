"""Write presentation/public/hour_lookup.json from baseline_univariate.keras."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))

from infer import write_lookup  # noqa: E402


def main() -> None:
    path = write_lookup()
    print("wrote", path, "bytes", path.stat().st_size)


if __name__ == "__main__":
    main()
