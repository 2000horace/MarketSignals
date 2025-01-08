import pandas as pd
import matplotlib.pyplot as plt

from ..data import OrderBookData

__all__ = [
    'plot_snapshot'
]

def plot_snapshot(df_snapshot: OrderBookData) -> None:
    ask_prices = [df_snapshot[f'askp{i}'].values[0] for i in range(1, 11)]
    ask_quantities = [df_snapshot[f'askq{i}'].values[0] for i in range(1, 11)]
    bid_prices = [df_snapshot[f'bidp{i}'].values[0] for i in range(1, 11)]
    bid_quantities = [df_snapshot[f'bidq{i}'].values[0] for i in range(1, 11)]

    # Plot the ask and bid sides of the order book
    plt.figure(figsize=(10, 6))

    # Ask side
    plt.step(ask_prices, ask_quantities, where='post', color='red', label='Ask')
    plt.fill_between(ask_prices, ask_quantities, step='post', color='red', alpha=0.2)

    # Bid side
    plt.step(bid_prices, bid_quantities, where='post', color='green', label='Bid')
    plt.fill_between(bid_prices, bid_quantities, step='post', color='green', alpha=0.2)

    # Labeling the plot
    plt.xticks(bid_prices[::-1] + ask_prices, rotation=45)  # Add price levels for clarity
    plt.xlabel('Price')
    plt.ylabel('Quantity')
    plt.title('Limit Order Book Snapshot at {}'.format(df_snapshot.index[0]))
    plt.legend()
    plt.grid(True)
    plt.show()