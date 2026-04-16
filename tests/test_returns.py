"""Testy dla obliczania zwrotów i agregacji portfelowej."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.returns import compute_returns, portfolio_returns


def test_log_returns_against_formula():
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]})
    r = compute_returns(prices, kind="log")
    assert r["X"].iloc[0] == pytest.approx(np.log(110 / 100))
    assert r["X"].iloc[1] == pytest.approx(np.log(99 / 110))


def test_simple_returns_against_formula():
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]})
    r = compute_returns(prices, kind="simple")
    assert r["X"].iloc[0] == pytest.approx(0.1)
    assert r["X"].iloc[1] == pytest.approx(-0.1)


def test_weighted_portfolio_return(two_asset_prices):
    rets = compute_returns(two_asset_prices, kind="simple")
    w = np.array([0.7, 0.3])
    port = portfolio_returns(rets, w)
    manual = 0.7 * rets["A"] + 0.3 * rets["B"]
    pd.testing.assert_series_equal(port, manual.rename("portfolio"), check_names=False)


def test_weight_length_mismatch_raises(two_asset_prices):
    rets = compute_returns(two_asset_prices, kind="log")
    with pytest.raises(ValueError):
        portfolio_returns(rets, np.array([1.0]))
