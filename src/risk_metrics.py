import numpy as np
import pandas as pd

def var_cvar(returns, confidence_level=0.05):
    var = np.percentile(returns, confidence_level*100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def rolling_metrics(returns, window=30):
    rolling_vol = returns.rolling(window=window).std()
    rolling_sharpe = returns.rolling(window=window).mean() / rolling_vol
    return rolling_vol, rolling_sharpe

def max_drawdown(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def stress_test(portfolio_returns, scenarios):
    """
    Apply different scenario shocks to portfolio returns.
    scenarios: dict of {scenario_name: shock_vector}
    """
    results = {}
    for name, shock in scenarios.items():
        shocked_returns = portfolio_returns + shock
        results[name] = shocked_returns.cumsum()
    return results
