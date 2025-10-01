import matplotlib.pyplot as plt
import plotly.express as px

def plot_efficient_frontier(returns, risks, sharpe_ratios):
    plt.figure(figsize=(10,6))
    sc = plt.scatter(risks, returns, c=sharpe_ratios, cmap='viridis')
    plt.colorbar(sc, label='Sharpe Ratio')
    plt.xlabel('Risk (Std Dev)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.show()

def plot_interactive_frontier(returns, risks, sharpe_ratios):
    fig = px.scatter(x=risks, y=returns, color=sharpe_ratios,
                     labels={'x':'Risk (Std Dev)', 'y':'Expected Return', 'color':'Sharpe Ratio'},
                     title='Interactive Efficient Frontier')
    fig.show()
