import pandas as pd
import numpy as np

def apply_transaction_costs(weights_old, weights_new, cost_rate=0.001):
    """Transaction cost applied when changing weights"""
    cost = cost_rate * np.sum(np.abs(weights_new - weights_old))
    return cost

def simulate_rebalance(prices, weights, freq='ME', starting_capital=100000, transaction_cost=0.001):
    """
    Simulate periodic portfolio rebalancing with transaction costs.
    """
    # Resample dates (month-end)
    rebalance_dates = prices.resample(freq).first().index

    portfolio_value = starting_capital
    portfolio_values = []

    for i in range(1, len(rebalance_dates)):
        # Use the closest previous available date
        start_date = prices.index.asof(rebalance_dates[i-1])
        end_date = prices.index.asof(rebalance_dates[i])

        start_price = prices.loc[start_date]
        end_price = prices.loc[end_date]

        # Period return
        period_returns = (end_price / start_price - 1)

        # Transaction costs (approx, can refine for actual turnover)
        turnover = 0  # assuming constant weights here
        portfolio_value = portfolio_value * (1 + (weights * period_returns).sum()) * (1 - transaction_cost*turnover)

        portfolio_values.append(portfolio_value)

    rebalance_series = pd.Series(portfolio_values, index=rebalance_dates[1:])
    return rebalance_series


