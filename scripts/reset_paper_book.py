"""Archive the paper account and start a fresh one.

    python scripts/reset_paper_book.py

For scrapping a run: the book's positions, closed trades and cash balance are
written to ``state/paper_book_archive/`` under a timestamp and replaced with a
new book at full budget holding nothing.

**The archive is the point.** A scrapped run is still evidence -- it is how
you find out that a rule change mattered -- so nothing is deleted, only moved
aside. ``--list`` shows what has been kept.

The new book holds no positions and all cash rather than being seeded from a
selection close. The next refresh while the market is open buys the current
top ten at live prices, which is a real set of entries; seeding from a close
that may be days old invents entry prices the account never paid.

Nothing here touches the dashboard payload. The next refresh rewrites the
live book and the realised section from the new account, and the archived run
stays readable on disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_settings  # noqa: E402
from execution.paper_book import PaperBook  # noqa: E402

IST = dt.timedelta(hours=5, minutes=30)


def summarize(book: PaperBook) -> str:
    realized = book.realized()
    return (
        f"{len(book.positions)} open, {realized['trades']} closed, "
        f"realised {realized['net_pnl']:+,.0f}, cash {book.cash:,.0f} "
        f"of {book.budget:,.0f}"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, default=REPO_ROOT / "state" / "paper_book.json")
    parser.add_argument(
        "--archive-dir", type=Path, default=REPO_ROOT / "state" / "paper_book_archive"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="starting capital for the new book (default: paper_book.budget_inr)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list archived runs and exit, changing nothing"
    )
    args = parser.parse_args(argv[1:])

    if args.list:
        archives = sorted(args.archive_dir.glob("*.json"))
        if not archives:
            print(f"no archived runs in {args.archive_dir}")
            return 0
        for path in archives:
            try:
                book = PaperBook.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError) as exc:
                print(f"{path.name}  unreadable: {exc}")
                continue
            print(f"{path.name}  {summarize(book)}")
        return 0

    budget = args.budget
    if budget is None:
        budget = load_settings().paper_book.budget_inr

    existing = PaperBook.load(args.book)
    if existing is not None:
        args.archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = (dt.datetime.now(dt.UTC) + IST).strftime("%Y-%m-%d_%H%M")
        archived = args.archive_dir / f"paper_book_{stamp}.json"
        # copy2 rather than rename: if writing the new book fails, the old one
        # is still where every other tool expects to find it.
        shutil.copy2(args.book, archived)
        print(f"archived {summarize(existing)}")
        print(f"  -> {archived}")
    else:
        print(f"no existing book at {args.book}")

    opened = (dt.datetime.now(dt.UTC) + IST).date()
    fresh = PaperBook(
        budget=budget,
        cash=budget,
        opened_at=opened.isoformat(),
        updated_at=opened.isoformat(),
    )
    fresh.save(args.book)
    print(
        f"new book: {budget:,.0f} cash, no positions. The next refresh while the "
        "market is open buys the current top ten at live prices."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
