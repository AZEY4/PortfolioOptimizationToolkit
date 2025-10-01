import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / std
    return ret, std, sharpe

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0, 
                       bounds=None, constraints=None, target_return=None, robust=False):
    """
    Multi-objective portfolio optimization:
    - Max Sharpe
    - Min Risk
    - Optional target return
    - Optional sector/ESG constraints
    - Optional robust shrinkage covariance
    """
    num_assets = len(mean_returns)
    
    # Robust shrinkage covariance
    if robust:
        avg_var = np.mean(np.diag(cov_matrix))
        cov_matrix = 0.5 * cov_matrix + 0.5 * np.eye(num_assets) * avg_var

    # Default bounds and constraints
    if bounds is None:
        bounds = tuple((0,1) for _ in range(num_assets))
    if constraints is None:
        constraints = [{'type':'eq','fun': lambda x: np.sum(x)-1}]
        if target_return is not None:
            constraints.append({'type':'eq','fun': lambda x: np.dot(x, mean_returns)-target_return})

    # Optimize Sharpe ratio
    def neg_sharpe(weights):
        return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

    result = minimize(neg_sharpe, num_assets*[1./num_assets,], method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result
