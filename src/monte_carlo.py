import numpy as np

def monte_carlo_simulation(mean_returns, cov_matrix, weights, num_simulations=10000):
    num_assets = len(mean_returns)
    results = []
    for _ in range(num_simulations):
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
        port_return = np.dot(weights, simulated_returns)
        results.append(port_return)
    return np.array(results)
