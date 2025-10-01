import pandas as pd
import numpy as np

def fama_french_adjustment(asset_returns, factor_returns, betas):
    """
    Adjust returns using Fama-French factors (3 or 5 factors)
    asset_returns: Series or DataFrame
    factor_returns: DataFrame
    betas: DataFrame of factor exposures
    """
    adjusted = asset_returns + factor_returns.dot(betas.T)
    return adjusted

def momentum_volatility_adjustment(returns, momentum_window=20, vol_window=20):
    """
    Adjust returns using momentum and volatility factors.
    
    Parameters:
        returns: pd.DataFrame of daily returns (not prices)
        momentum_window: int, rolling window for momentum
        vol_window: int, rolling window for volatility

    Returns:
        pd.DataFrame of adjusted returns
    """
    # Ensure input is returns
    if (returns > 1).any().any():  # checks if any value > 1 in the entire DataFrame
        raise ValueError("Input must be returns (not prices)")

    # Momentum factor: rolling mean
    momentum = returns.rolling(window=momentum_window).mean()

    # Volatility factor: rolling std
    vol = returns.rolling(window=vol_window).std()

    # Adjusted returns: add momentum, penalize by volatility
    adjusted = returns + momentum - 0.5 * vol

    # Drop all rows with NaN values
    adjusted = adjusted.dropna(how='any')

    return adjusted


