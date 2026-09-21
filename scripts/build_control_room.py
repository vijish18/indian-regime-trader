"""Inline the dashboard payload into the control-room page.

    python scripts/build_control_room.py

``state/control_room.html`` is the page as written: template, styles and
render code, with ``__SEED__`` standing in for the data. This writes
``state/control_room.built.html`` with the current payload substituted in,
which is the file published as the artifact.

The seed is only the *starting* state. Once published, the page subscribes to
its data channel and merges whatever ``scripts/refresh_live_book.py`` pushes
there, so a stale seed shows briefly and is then replaced. Rebuilding is
needed when the page's own markup or render code changes -- not on every
price refresh, which the channel handles.

Distinct from ``scripts/build_dashboard.py``, which generates the older
standalone ``state/dashboard.html`` from backtest reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDER = "__SEED__"


def build(template_path: Path, data_path: Path, out_path: Path) -> int:
    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"{template_path} has no {PLACEHOLDER} placeholder to fill")

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    # Reject NaN/Infinity rather than emit them. Python writes them as bare
    # NaN tokens, which json.loads happily reads back and every browser's
    # JSON.parse rejects -- and the page's parse is wrapped in a try/catch
    # that falls back to {}, so a single NaN silently blanks the whole
    # dashboard instead of breaking one tile. This has happened.
    seed = json.dumps(
        raw, separators=(",", ":"), allow_nan=False, default=str
    )
    # The payload sits inside a <script> element, so the only sequence that
    # can escape it is a literal closing tag.
    seed = seed.replace("</", "<\\/")

    out_path.write_text(template.replace(PLACEHOLDER, seed), encoding="utf-8")
    print(
        f"{out_path.name}  {out_path.stat().st_size:,} bytes  "
        f"(seed {len(seed):,} bytes, generated_at {raw.get('generated_at')})"
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path, default=REPO_ROOT / "state" / "control_room.html"
    )
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "state" / "dashboard_data.json"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "state" / "control_room.built.html"
    )
    args = parser.parse_args(argv[1:])
    for path in (args.template, args.data):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    return build(args.template, args.data, args.out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
