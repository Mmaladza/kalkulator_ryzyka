"""Prosty cache parquet dla szeregów cenowych, trzymany na dysku.

Uzasadnienie: wielokrotne uderzanie w yfinance / Stooq podczas iteracji
na UI jest powolne i czasem rate-limitowane. Kluczem jest (source, ticker,
start, end), a pliki trzymamy w ~/.cache/kalkulator_ryzyka/. To cache
poziomu demo, nie produkcji — brak TTL, brak blokad, brak scalania
częściowych zakresów.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd


def _cache_dir() -> Path:
    base = Path.home() / ".cache" / "kalkulator_ryzyka"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _key(source: str, ticker: str, start: date, end: date) -> str:
    raw = f"{source}|{ticker}|{start.isoformat()}|{end.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _path(source: str, ticker: str, start: date, end: date) -> Path:
    return _cache_dir() / f"{_key(source, ticker, start, end)}.parquet"


def get(source: str, ticker: str, start: date, end: date) -> pd.Series | None:
    p = _path(source, ticker, start, end)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        # Zawsze zapisane jako pojedyncza kolumna 'close':
        return df["close"]
    except Exception:
        # Uszkodzony wpis cache — usuwamy i pobierzemy ponownie.
        p.unlink(missing_ok=True)
        return None


def put(source: str, ticker: str, start: date, end: date, series: pd.Series) -> None:
    p = _path(source, ticker, start, end)
    df = series.rename("close").to_frame()
    df.to_parquet(p)
