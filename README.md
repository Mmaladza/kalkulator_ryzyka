# Kalkulator ryzyka rynkowego — VaR / CVaR


[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Aplikacja desktopowa (PyQt6) do obliczania **Value at Risk** oraz
**Conditional Value at Risk** (Expected Shortfall) dla portfela akcji.
Dane cenowe pobierane są z **Yahoo Finance** oraz **Stooq**.

Projekt zbudowany jako demonstracja przepływu pracy analityka ryzyka
rynkowego: pobranie danych → obliczenie zwrotów → estymacja VaR kilkoma
metodami → backtesting.

## Funkcje

### Metody obliczania ryzyka
- **Symulacja historyczna** — nieparametryczny kwantyl empiryczny.
- **Parametryczna (wariancja–kowariancja)**
  - Rozkład normalny — zamknięta forma.
  - Rozkład t-Studenta — stopnie swobody dopasowane metodą największej
    wiarygodności, lepiej oddaje grube ogony.
- **Monte Carlo** — parametryczne MC na szeregu zwrotów portfela;
  konfigurowalna liczba symulacji oraz długość horyzontu.

### Backtesting
- **Test Kupca Proportion-of-Failures (POF)** — test ilorazu wiarygodności
  porównujący zrealizowaną częstość przekroczeń z częstością implikowaną
  przez model; rozkład χ²(1) przy hipotezie zerowej. Regulatorzy stosują
  ten test przy walidacji modeli wewnętrznych (Bazylea, 1996).

### Obsługa portfela
- Wiele tickerów z wagami zdefiniowanymi przez użytkownika
  (automatyczna normalizacja).
- Źródło danych wybierane osobno dla każdego tickera (yfinance lub Stooq).
- Skalowanie horyzontu regułą pierwiastka z czasu.

### Interfejs
- Okno główne PyQt6 z edytowalną tabelą portfela, formularzem parametrów,
  osadzonymi wykresami matplotlib (histogram zwrotów z liniami VaR/CVaR,
  krzywa kapitału) i panelem wyników.
- Cięższe operacje (pobieranie danych, Monte Carlo) liczą się w wątku
  roboczym `QThreadPool`, więc GUI nie zamarza.
- Eksport wyników do CSV.
- Cache parquet na dysku, żeby nie bombardować API przy powtórnych
  obliczeniach.

## Struktura projektu

```
src/
    models.py              # dataclasses: Position, PortfolioSpec, RiskParams,
                           # RiskResult, BacktestResult
    data/
        providers.py       # YFinanceProvider, StooqProvider
        cache.py           # cache parquet po kluczu (źródło, ticker, daty)
    risk/
        returns.py         # zwroty log/simple, agregacja portfelowa
        historical.py      # historyczny VaR/CVaR, skalowanie sqrt(T)
        parametric.py      # VaR/CVaR dla rozkładu normalnego i t-Studenta
        monte_carlo.py     # parametryczne MC na zwrotach portfela
        portfolio.py       # dispatch wysokiego poziomu → RiskResult
        backtest.py        # test Kupca POF
    ui/
        main_window.py     # QMainWindow komponujące widgety
        workers.py         # QRunnable dla asynchronicznego pipeline'u
        widgets/           # tabela tickerów, panel parametrów,
                           # panel wyników, wykresy
tests/                     # zestaw pytest (19 testów, oracle analityczne)
main.py                    # punkt wejścia
```

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Testowane na Pythonie 3.12.

## Konfiguracja Stooq (opcjonalna)

Stooq od 2024 r. wymaga klucza API. Aplikacja czyta go ze zmiennej
środowiskowej `STOOQ_API_KEY`:

1. Otwórz w przeglądarce dowolny link postaci
   `https://stooq.com/q/d/?s=wig20&get_apikey`.
2. Wpisz captchę i skopiuj parametr `apikey` z linku do pobrania CSV
   na dole strony.
3. Ustaw zmienną środowiskową przed uruchomieniem aplikacji:

   ```bash
   # PowerShell
   $env:STOOQ_API_KEY = "twoj_klucz_z_stooq"
   python main.py

   # Linux/macOS
   export STOOQ_API_KEY=twoj_klucz_z_stooq
   python main.py
   ```

Bez klucza wybór źródła „Stooq" w tabeli portfela skutkuje czytelnym
komunikatem błędu. **Polskie tickery są też dostępne przez yfinance**
z sufiksem `.WA` (np. `CDR.WA`, `PKO.WA`, `ALE.WA`, `^WIG20`), więc dla
typowych zastosowań klucz Stooq nie jest konieczny.

## Uruchomienie

```bash
python main.py
```

Aplikacja startuje z małym portfelem demo (AAPL, MSFT, ^GSPC). Możesz
edytować tabelę tickerów, wybrać daty i metodę, po czym wcisnąć
**Oblicz ryzyko**.

## Testy

```bash
pytest tests/ -v
```

Każdy moduł matematyczny jest walidowany względem analitycznego oracle
tam, gdzie on istnieje (np. historyczny VaR vs. formuła z kwantyla rozkładu
normalnego dla i.i.d. N(0, σ²); zbieżność MC do zamkniętej formy VaR
normalnego przy 200 tys. symulacji; statystyka LR testu Kupca vs.
ręczne obliczenie).

## Notatki metodologiczne

- **Konwencja strat.** VaR i CVaR raportujemy jako *dodatnie* wartości
  strat — tzn. `var_relative = 0.023` oznacza stratę 2,3% portfela przy
  zadanym poziomie ufności, zgodnie z praktyką branżową.
- **Skalowanie horyzontu.** Aplikacja stosuje regułę pierwiastka z czasu
  (`VaR_h = VaR_1 · √h`), która jest dokładna przy i.i.d. normalnych
  zwrotach i standardową aproksymacją regulatora dla krótkich horyzontów
  (np. 10-dniowy VaR bazylejski).
- **Zakres backtestu.** Test Kupca liczy się in-sample dla statycznej
  estymaty VaR. W środowisku produkcyjnym standardem jest backtest
  out-of-sample z oknem kroczącym, uzupełniony o test niezależności
  Christoffersena — oba są naturalnymi kierunkami rozszerzenia.

## Literatura

- McNeil, Frey, Embrechts — *Quantitative Risk Management*, wyd. 2 (2015).
- Kupiec, P. (1995). *Techniques for verifying the accuracy of risk
  measurement models*.
- Basel Committee on Banking Supervision (1996). *Supervisory framework for
  the use of "backtesting" in conjunction with the internal models approach
  to market risk capital requirements*.

## Licencja

MIT.
