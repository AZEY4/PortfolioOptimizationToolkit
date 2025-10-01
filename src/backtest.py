import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cumulative_returns(returns):
    """Compute cumulative returns from daily returns."""
    return (1 + returns).cumprod()

def annualized_return(cumulative_returns, periods_per_year=252):
    """Compute CAGR from cumulative returns."""
    total_periods = len(cumulative_returns)
    total_return = cumulative_returns[-1] / cumulative_returns[0] - 1
    return (1 + total_return) ** (periods_per_year / total_periods) - 1

def annualized_volatility(returns, periods_per_year=252):
    """Compute annualized volatility."""
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Compute annualized Sharpe Ratio."""
    excess_return = returns - risk_free_rate / periods_per_year
    return excess_return.mean() / excess_return.std() * np.sqrt(periods_per_year)

def max_drawdown(cumulative_returns):
    """Compute the maximum drawdown."""
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def backtest_portfolio(prices, weights, benchmark_prices=None, risk_free_rate=0.0):
    """
    Backtest portfolio against a benchmark.
    prices: DataFrame of asset prices
    weights: Portfolio weights (array-like)
    benchmark_prices: DataFrame or Series of benchmark prices
    """
    # Compute daily returns
    asset_returns = prices.pct_change().dropna()
    portfolio_returns = asset_returns.dot(weights)
    portfolio_cum = cumulative_returns(portfolio_returns)

    # Metrics
    cagr = annualized_return(portfolio_cum)
    vol = annualized_volatility(portfolio_returns)
    sharpe = sharpe_ratio(portfolio_returns, risk_free_rate)
    mdd = max_drawdown(portfolio_cum)

    print("Portfolio Performance Metrics:")
    print(f"CAGR: {cagr:.2%}")
    print(f"Annualized Volatility: {vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")

    # Benchmark
    if benchmark_prices is not None:
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_cum = cumulative_returns(benchmark_returns)
        plt.figure(figsize=(12,6))
        plt.plot(portfolio_cum, label='Portfolio')
        plt.plot(benchmark_cum, label='Benchmark')
        plt.title('Portfolio vs Benchmark Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.show()

    return portfolio_cum
