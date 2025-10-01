def apply_transaction_costs(weights_old, weights_new, transaction_cost=0.001):
    cost = transaction_cost * sum(abs(weights_new - weights_old))
    return cost
