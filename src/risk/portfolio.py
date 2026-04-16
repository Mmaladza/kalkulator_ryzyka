"""Orkiestracja VaR / CVaR na poziomie portfela.

Ten moduł spaja poszczególne metody VaR z parą PortfolioSpec / RiskParams
i produkuje typowany RiskResult. UI i testy korzystają z tego samego wejścia,
dzięki czemu cały switch po metodzie siedzi w jednym miejscu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import (
    PortfolioSpec,
    RiskMethod,
    RiskParams,
    RiskResult,
)
from src.risk import historical, monte_carlo, parametric
from src.risk.returns import compute_returns, portfolio_returns


def compute_portfolio_returns(
    prices: pd.DataFrame,
    spec: PortfolioSpec,
) -> pd.Series:
    """Liczy pojedynczy szereg zwrotów portfela z wieloaktywowego panelu cen."""
    if list(prices.columns) != list(spec.tickers):
        # Zmień kolejność kolumn tak, by pasowała do wag (ceny mogą przyjść
        # w dowolnej kolejności).
        prices = prices[list(spec.tickers)]
    rets = compute_returns(prices, kind=spec.return_type)
    weights = np.array(spec.normalized_weights)
    return portfolio_returns(rets, weights)


def compute_risk(
    prices: pd.DataFrame,
    spec: PortfolioSpec,
    params: RiskParams,
) -> RiskResult:
    """Wejście wysokiego poziomu — dispatch na wybraną metodę i pakowanie wyniku."""
    port_rets = compute_portfolio_returns(prices, spec)
    extras: dict[str, float] = {}

    if params.method == RiskMethod.HISTORICAL:
        var_1d, cvar_1d = historical.historical_var_cvar(port_rets, params.confidence)
        var = historical.scale_horizon(var_1d, params.horizon_days)
        cvar = historical.scale_horizon(cvar_1d, params.horizon_days)

    elif params.method == RiskMethod.PARAMETRIC_NORMAL:
        var_1d, cvar_1d = parametric.parametric_var_cvar_normal(
            port_rets, params.confidence
        )
        var = historical.scale_horizon(var_1d, params.horizon_days)
        cvar = historical.scale_horizon(cvar_1d, params.horizon_days)

    elif params.method == RiskMethod.PARAMETRIC_T:
        var_1d, cvar_1d, df = parametric.parametric_var_cvar_t(
            port_rets, params.confidence
        )
        var = historical.scale_horizon(var_1d, params.horizon_days)
        cvar = historical.scale_horizon(cvar_1d, params.horizon_days)
        extras["fitted_df"] = df

    elif params.method == RiskMethod.MONTE_CARLO:
        var, cvar = monte_carlo.monte_carlo_var_cvar(
            port_rets,
            params.confidence,
            horizon_days=params.horizon_days,
            n_sims=params.mc_simulations,
            seed=params.mc_seed,
        )
        extras["n_sims"] = float(params.mc_simulations)

    else:
        raise ValueError(f"Nieznana metoda: {params.method}")

    return RiskResult(
        method=params.method,
        confidence=params.confidence,
        horizon_days=params.horizon_days,
        var_relative=var,
        cvar_relative=cvar,
        var_absolute=var * spec.portfolio_value,
        cvar_absolute=cvar * spec.portfolio_value,
        portfolio_value=spec.portfolio_value,
        n_observations=len(port_rets),
        extras=extras,
    )
