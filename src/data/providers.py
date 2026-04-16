"""Dostawcy danych cenowych — yfinance i Stooq.

Obaj dostawcy dzielą ten sam interfejs:

    fetch(ticker, start, end) -> pd.Series dziennych cen zamknięcia

dzięki czemu warstwa portfelowa może transparentnie sklejać tickery
z różnych źródeł. Cienka nakładka `fetch_prices` przyjmuje listę par
(ticker, source) i zwraca jeden wyrównany DataFrame.

Dla Stooq odpytujemy bezpośrednio publiczny endpoint CSV
(`https://stooq.com/q/d/l/`) zamiast korzystać z pandas-datareader, który
jest porzucony i nie działa już na Pythonie 3.12 (importuje wycofane
`distutils`). Stooq od 2024 r. wymaga `apikey` w zapytaniu — klucz
pobiera się ze strony `https://stooq.com/q/d/?s=<ticker>&get_apikey`
(po wpisaniu captchy) i podaje do aplikacji przez zmienną środowiskową
`STOOQ_API_KEY`. Jeśli klucza nie ma, dostawca rzuca czytelny błąd.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date
from io import StringIO
from typing import Iterable

import pandas as pd
import requests

from src.data import cache
from src.models import DataSource, Position


# User-Agent jest tu istotny — Stooq odrzuca część requestów bez nagłówka.
_STOOQ_HEADERS = {"User-Agent": "kalkulator-ryzyka/0.1 (+https://example.org)"}
_STOOQ_URL = "https://stooq.com/q/d/l/"
_STOOQ_APIKEY_ENV = "STOOQ_API_KEY"


class PriceProvider(ABC):
    source: DataSource

    @abstractmethod
    def fetch(self, ticker: str, start: date, end: date) -> pd.Series:
        """Zwraca szereg dziennych cen zamknięcia zaindeksowany po dacie."""

    def fetch_cached(self, ticker: str, start: date, end: date) -> pd.Series:
        hit = cache.get(self.source.value, ticker, start, end)
        if hit is not None:
            return hit
        series = self.fetch(ticker, start, end)
        if series.empty:
            raise ValueError(
                f"Dostawca {self.source.value} nie zwrócił danych dla {ticker} "
                f"w przedziale {start} – {end}"
            )
        cache.put(self.source.value, ticker, start, end, series)
        return series


class YFinanceProvider(PriceProvider):
    source = DataSource.YFINANCE

    def fetch(self, ticker: str, start: date, end: date) -> pd.Series:
        import yfinance as yf

        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)
        # Nowsze wersje yfinance zwracają MultiIndex na kolumnach — spłaszczamy.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].astype(float)
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close.name = ticker
        return close


class StooqProvider(PriceProvider):
    """Pobieranie cen ze Stooq przez bezpośredni endpoint CSV.

    Wymagane parametry zapytania:
        s  — symbol (np. "wig20", "kgh", "^spx")
        d1 — data początkowa w formacie YYYYMMDD
        d2 — data końcowa w formacie YYYYMMDD
        i  — interwał ("d" = dzienny)

    Stooq zwraca CSV z kolumnami: Date,Open,High,Low,Close,Volume.
    Gdy symbol nie istnieje albo brakuje danych w zakresie, odpowiedź to
    pojedyncza linia "No data".
    """

    source = DataSource.STOOQ

    def fetch(self, ticker: str, start: date, end: date) -> pd.Series:
        api_key = os.environ.get(_STOOQ_APIKEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                "Stooq wymaga klucza API. Ustaw zmienną środowiskową "
                f"{_STOOQ_APIKEY_ENV} (zob. README → sekcja 'Konfiguracja "
                "Stooq'). Klucz pobierzesz ze strony "
                "https://stooq.com/q/d/?s=<ticker>&get_apikey "
                "po wpisaniu captchy. Alternatywnie użyj źródła yfinance — "
                "polskie tickery są dostępne z sufiksem '.WA' (np. CDR.WA, "
                "PKO.WA, ALE.WA)."
            )

        params = {
            "s": ticker.lower(),
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "i": "d",
            "apikey": api_key,
        }
        resp = requests.get(
            _STOOQ_URL,
            params=params,
            headers=_STOOQ_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()

        text = resp.text.strip()
        if not text or text.lower().startswith("no data"):
            return pd.Series(dtype=float)
        # Stooq przy nieważnym kluczu zwraca instrukcję z 'apikey' w treści
        # zamiast statusu HTTP błędu — wykrywamy to po nagłówku.
        first_line = text.splitlines()[0].lower()
        if "apikey" in first_line or "captcha" in first_line:
            raise RuntimeError(
                f"Stooq odrzucił klucz {_STOOQ_APIKEY_ENV} (możliwe, że jest "
                "nieważny lub wygasł). Pobierz nowy klucz na "
                "https://stooq.com/q/d/?s=<ticker>&get_apikey."
            )

        df = pd.read_csv(StringIO(text))
        # Brak kolumny "Close" oznacza nietypową odpowiedź (np. komunikat
        # o limicie) — traktujemy jak brak danych zamiast wybuchać.
        if "Close" not in df.columns or "Date" not in df.columns:
            return pd.Series(dtype=float)

        close = df.set_index("Date")["Close"].astype(float).sort_index()
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close.name = ticker
        return close


def _provider(source: DataSource) -> PriceProvider:
    if source == DataSource.YFINANCE:
        return YFinanceProvider()
    if source == DataSource.STOOQ:
        return StooqProvider()
    raise ValueError(f"Nieznane źródło: {source}")


def fetch_prices(
    positions: Iterable[Position],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Pobiera i wyrównuje ceny zamknięcia dla zbioru pozycji.

    Wynikowy DataFrame ma jedną kolumnę na ticker (w podanej kolejności),
    indeksowany iloczynem dni sesyjnych wspólnych dla wszystkich. Wiersze,
    w których któryś ticker jest pusty, są odrzucane — to upraszcza liczenie
    zwrotów kosztem ignorowania świąt obowiązujących tylko na jednym rynku.
    """
    series_list: list[pd.Series] = []
    for pos in positions:
        provider = _provider(pos.source)
        s = provider.fetch_cached(pos.ticker, start, end)
        series_list.append(s.rename(pos.ticker))

    df = pd.concat(series_list, axis=1, join="inner")
    df = df.dropna(how="any").sort_index()
    return df
