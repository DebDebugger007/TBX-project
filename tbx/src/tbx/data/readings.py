"""Parse the NLM ClinicalReadings text files.

The two sets use different formats:

Montgomery::

    Patient's Sex: M
    Patient's Age: 087Y
    <free-text reading>

Shenzhen::

    male 35yrs
    <free-text reading>

Some Shenzhen files give age in months or days (infants), and a few have stray
whitespace or odd encodings, so parsing is lenient and returns None for anything
it cannot read rather than guessing.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_MC_SEX = re.compile(r"Patient'?s\s+Sex\s*:\s*([MF])", re.IGNORECASE)
_MC_AGE = re.compile(r"Patient'?s\s+Age\s*:\s*0*(\d+)\s*([YMD])?", re.IGNORECASE)
# Shenzhen headers have typos ("femal 32yrs", "female24yrs", "male 42"), so the
# unit is optional (defaults to years) and sex only needs the word's start.
_SZ_AGE = re.compile(r"(\d+)\s*(yrs?|years?|y|months?|mos?|days?|d)?", re.IGNORECASE)
_SZ_SEX = re.compile(r"^\s*(fem|male|f\b|m\b)", re.IGNORECASE)

_UNIT_TO_YEARS = {"y": 1.0, "m": 1.0 / 12.0, "d": 1.0 / 365.25}


def _to_years(value: str, unit: Optional[str]) -> float:
    u = (unit or "y").lower()
    if u.startswith("mo") or u == "m":
        factor = _UNIT_TO_YEARS["m"]
    elif u.startswith("d"):
        factor = _UNIT_TO_YEARS["d"]
    else:
        factor = _UNIT_TO_YEARS["y"]
    return round(int(value) * factor, 2)


def _read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_reading(path: str | Path, dataset: str) -> dict:
    """Return {"age": float|None (years), "sex": "M"|"F"|None, "finding": str}."""
    text = _read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    age: Optional[float] = None
    sex: Optional[str] = None
    finding_lines: list[str] = []

    if dataset == "montgomery":
        for ln in lines:
            ms = _MC_SEX.search(ln)
            ma = _MC_AGE.search(ln)
            if ms:
                sex = ms.group(1).upper()
            elif ma:
                age = _to_years(ma.group(1), ma.group(2))
            else:
                finding_lines.append(ln)
    elif dataset == "shenzhen":
        if lines:
            header = lines[0]
            ms = _SZ_SEX.search(header)
            ma = _SZ_AGE.search(header)
            if ms:
                sex = "F" if ms.group(1).lower().startswith("f") else "M"
            if ma:
                age = _to_years(ma.group(1), ma.group(2))
            # if the first line held neither field, treat it as part of the reading
            finding_lines = lines[1:] if (ms or ma) else lines
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    return {"age": age, "sex": sex, "finding": " ".join(finding_lines)}
