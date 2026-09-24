"""Read corporate-action evidence from NSE PR archives without rewriting prices."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import re
import urllib.request
import zipfile
from decimal import Decimal
from pathlib import Path

URL = "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{day:%d%m%y}.zip"


def parse_date(value: str) -> str | None:
    if not value.strip():
        return None
    for format_ in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(value.strip(), format_).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unknown PR date {value!r}")


def terms(purpose: str) -> dict[str, str]:
    """Only recognise complete, unambiguous BC terms; keep the rest for review."""
    text = " ".join(purpose.upper().split())
    text = re.sub(r"^AGM/", "", text)
    bonus = re.fullmatch(r"BONUS\s+(\d+)\s*:\s*(\d+)", text)
    split = re.fullmatch(
        r"FV\s*SPLT FRM R[SE]\.?\s*(\d+(?:\.\d+)?) TO R[SE]\.?\s*(\d+(?:\.\d+)?)", text
    )
    dividend = re.fullmatch(
        r"(?:INT\s*DIV|INTERIM DIV|DIV|SPL\s*DIV|SPECIAL DIV)[\s-]*"
        r"R[SE]\.?\s*(\d+(?:\.\d+)?)(?:/-)?(?:\s*PER SH(?:ARE)?)?",
        text,
    )
    if bonus and all(Decimal(v) > 0 for v in bonus.groups()):
        return {"action_type": "bonus", "ratio_new": bonus[1], "ratio_old": bonus[2]}
    if split and all(Decimal(v) > 0 for v in split.groups()):
        return {"action_type": "split", "ratio_new": split[1], "ratio_old": split[2]}
    if dividend and Decimal(dividend[1]) >= 0:
        return {"action_type": "dividend", "cash_amount": dividend[1]}
    if text == "DEMERGER":
        return {"action_type": "demerger"}
    if text in {"MERGER", "SCHEME OF ARRANGEMENT", "SCHEME OF AMALGAMATION"}:
        return {"action_type": "merger"}
    return {"action_type": "unresolved"}


def all_terms(purpose: str) -> list[dict[str, str]]:
    """Handle multiple entitlements in one purpose; don't lose special dividends."""
    text = re.sub(r"^AGM/", "", " ".join(purpose.upper().split()))
    compact = re.fullmatch(r"DIV-\s*(\d+(?:\.\d+)?) SPLDV-\s*(\d+(?:\.\d+)?)", text)
    if compact:
        return [
            {
                "action_type": "dividend",
                "cash_amount": str(Decimal(compact[1]) + Decimal(compact[2])),
            }
        ]
    parts = re.split(r"/(?=(?:BONUS|(?:SPL|INT)?\s*DIV))", text)
    parsed = [terms(part) for part in parts]
    dividends = [p for p in parsed if p["action_type"] == "dividend"]
    others = [p for p in parsed if p["action_type"] != "dividend"]
    if dividends:
        others.append(
            {
                "action_type": "dividend",
                "cash_amount": str(sum(Decimal(p["cash_amount"]) for p in dividends)),
            }
        )
    return others


def archive_actions(blob: bytes) -> list[dict[str, str]]:
    """Retain original purpose and dates; descriptions may lack event terms."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [
            n
            for n in archive.namelist()
            if re.fullmatch(r"bc\d+\.csv", Path(n).name, re.IGNORECASE)
        ]
        if len(names) != 1:
            raise ValueError(f"Expected one BC corporate-action file, found {names}")
        text = archive.read(names[0]).decode("utf-8-sig").replace("\r\r\n", "\n")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if not {"SYMBOL", "EX_DT", "PURPOSE"} <= set(reader.fieldnames or []):
            raise ValueError("Unexpected PR corporate-action columns")
        rows = []
        for row in reader:
            if not row.get("SYMBOL", "").strip():
                continue
            cleaned = {k: (v or "").strip() for k, v in row.items() if k is not None}
            cleaned["archive_member"] = names[0]
            rows.append(cleaned)
        return rows


def load_archive(day: dt.date, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"PR{day:%d%m%y}.zip"
    url = URL.format(day=day)
    if path.exists():
        blob = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            blob = response.read()
        archive_actions(blob)  # validate before caching; never cache an HTML error
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(blob)
        temporary.replace(path)
    return {
        "archive_date": day.isoformat(),
        "source_url": url,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "rows": archive_actions(blob),
    }
