import numpy as np

def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.0):
    results = np.zeros((3, num_portfolios))
    num_assets = len(mean_returns)
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret = np.dot(weights, mean_returns)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / std
        results[0, i], results[1, i], results[2, i] = ret, std, sharpe
    return results
