import pandas as pd
from .base_data import BaseData

class OrderBookData(BaseData):
    """
    A class representing order book data, initialized with DataFrames for bids and asks.
    """

    def __init__(self, symbol: str, bids_df: pd.DataFrame, asks_df: pd.DataFrame):
        """
        Initialize the order book data object.

        :param symbol: The asset symbol.
        :param timestamp: The timestamp of the order book snapshot.
        :param bids_df: A DataFrame containing bid levels with two columns per level (price, quantity).
        :param asks_df: A DataFrame containing ask levels with two columns per level (price, quantity).
        """
        
        self.bids_df = bids_df
        self.asks_df = asks_df
        self._validate_dataframes()
        self._assign_levels()

        super().__init__(symbol, pd.concat([bids_df, asks_df], axis=1))

    def _validate_dataframes(self):
        """
        Validate the structure of the bids and asks DataFrames.
        """
        for df, side in [(self.bids_df, "bids"), (self.asks_df, "asks")]:
            if df.shape[1] % 2 != 0:
                raise ValueError(f"{side.capitalize()} DataFrame must have an even number of columns (price and quantity pairs).")
            if df.shape[0] != 1:
                raise ValueError(f"{side.capitalize()} DataFrame must contain exactly one row representing the snapshot.")

    def _assign_levels(self):
        """
        Assign bid and ask levels as attributes.
        """
        bid_levels = self.bids_df.columns
        ask_levels = self.asks_df.columns

        for i in range(1, len(bid_levels) // 2 + 1):
            setattr(self, f"bidp{i}", self.bids_df.iloc[0, (i - 1) * 2])
            setattr(self, f"bidq{i}", self.bids_df.iloc[0, (i - 1) * 2 + 1])

        for i in range(1, len(ask_levels) // 2 + 1):
            setattr(self, f"askp{i}", self.asks_df.iloc[0, (i - 1) * 2])
            setattr(self, f"askq{i}", self.asks_df.iloc[0, (i - 1) * 2 + 1])

    def validate(self):
        """
        Validate the order book data object.
        """
        # Ensure no prices are negative
        assert all(self.bids_df.iloc[0, ::2] > 0), "All bid prices must be positive."
        assert all(self.asks_df.iloc[0, ::2] > 0), "All ask prices must be positive."
        # Ensure no volumes are negative
        assert all(self.bids_df.iloc[0, 1::2] >= 0), "All bid quantities must be non-negative."
        assert all(self.asks_df.iloc[0, 1::2] >= 0), "All ask quantities must be non-negative."
        # Ensure bid prices are lower than ask prices for corresponding levels
        for i in range(1, len(self.bids_df.columns) // 2 + 1):
            assert getattr(self, f"bidp{i}") < getattr(self, f"askp{i}"), \
                f"Bid price at level {i} must be less than ask price."

    def to_dict(self):
        """
        Convert the order book data to a dictionary representation.
        """
        result = {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
        }
        for attr in dir(self):
            if attr.startswith(("bidp", "bidq", "askp", "askq")):
                result[attr] = getattr(self, attr)
        return result
