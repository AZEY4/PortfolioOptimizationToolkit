import pandas as pd
from src.data_fetcher import fetch_data
from src.portfolio import optimize_portfolio, portfolio_stats
from src.factor_model import fama_french_adjustment, momentum_volatility_adjustment
from src.backtest import backtest_portfolio
from src.rebalance_simulator import simulate_rebalance
from src.risk_metrics import var_cvar, rolling_metrics, max_drawdown
from src.utils import generate_random_portfolios
from src.visualizer import plot_efficient_frontier, plot_interactive_frontier

print("Step 1: Fetching market data...")
tickers = ['AAPL','MSFT','GOOG','AMZN']
data = fetch_data(tickers)
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
print(f"Fetched {len(tickers)} tickers, data shape: {data.shape}")

print("\nOptional: Fetching benchmark data (S&P 500)...")
benchmark = fetch_data(['^GSPC'])
print(f"Benchmark data shape: {benchmark.shape}")

print("\nStep 2: Applying factor adjustments (momentum + volatility)...")
# Ensure we pass returns, not prices
adjusted_returns = momentum_volatility_adjustment(returns, momentum_window=20, vol_window=20)
mean_returns_adj = adjusted_returns.mean()
print("Factor-adjusted mean returns calculated.")


print("\nStep 3: Optimizing portfolio...")
opt_result = optimize_portfolio(mean_returns_adj, cov_matrix, robust=True)
weights = opt_result.x
ret, risk, sharpe = portfolio_stats(weights, mean_returns_adj, cov_matrix)
print("Optimal Weights:", weights)
print(f"Expected Return: {ret:.2%}")
print(f"Portfolio Risk: {risk:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

print("\nStep 4: Generating Efficient Frontier...")
random_portfolios = generate_random_portfolios(5000, mean_returns_adj, cov_matrix)
plot_efficient_frontier(random_portfolios[0], random_portfolios[1], random_portfolios[2])
print("Efficient Frontier plotted.")

print("\nStep 5: Backtesting portfolio vs benchmark...")
portfolio_cum = backtest_portfolio(data, weights, benchmark_prices=benchmark['^GSPC'])
print("Backtest complete, cumulative returns calculated.")

print("\nStep 6: Simulating rebalancing with transaction costs...")
rebalance_series = simulate_rebalance(data, weights, freq='M', starting_capital=100000, transaction_cost=0.001)
print("Rebalancing simulation complete.")
print("Rebalancing portfolio values (monthly):")
print(rebalance_series)

print("\nStep 7: Calculating risk metrics...")
portfolio_returns = data.pct_change().dropna().dot(weights)
var, cvar = var_cvar(portfolio_returns)
rolling_vol, rolling_sharpe = rolling_metrics(portfolio_returns)
mdd = max_drawdown((1+portfolio_returns).cumprod())

print(f"Portfolio VaR (5%): {var:.2%}")
print(f"Portfolio CVaR (5%): {cvar:.2%}")
print(f"Max Drawdown: {mdd:.2%}")

print("\nAll steps completed successfully!")
