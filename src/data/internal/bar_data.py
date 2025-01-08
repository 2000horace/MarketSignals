import pandas as pd
from .base_data import BaseData

class BarData(BaseData):
    """
    A class representing bar data (OHLC or partial), with mandatory close price and optional open, high, and low prices.
    """

    def __init__(
        self,
        symbol: str,
        timestamps: pd.DatetimeIndex,
        close_price: pd.Series,
        open_price: pd.Series = None,
        high_price: pd.Series = None,
        low_price: pd.Series = None,
    ):
        """
        Initialize the BarData object.

        :param symbol: The asset symbol (e.g., 'AAPL', 'BTC-USD').
        :param timestamps: A pandas DatetimeIndex representing the timestamps of the bars.
        :param close_price: A pandas Series containing the closing prices.
        :param open_price: A pandas Series containing the opening prices (optional).
        :param high_price: A pandas Series containing the highest prices (optional).
        :param low_price: A pandas Series containing the lowest prices (optional).
        """
        # Create a DataFrame to hold all the bar data
        data = pd.DataFrame({"close": close_price}, index=timestamps)
        if open_price is not None:
            data["open"] = open_price
        if high_price is not None:
            data["high"] = high_price
        if low_price is not None:
            data["low"] = low_price

        # Initialize the parent class
        super().__init__(symbol, data)

        # Validate the data
        self.validate()

    def validate(self):
        """
        Validate the bar data.
        """
        # Ensure the index is a pandas DatetimeIndex
        assert isinstance(self.data.index, pd.DatetimeIndex), "Timestamps must be a pandas DatetimeIndex."

        # Ensure close price exists
        assert "close" in self.data.columns, "Close price is mandatory."

        # Ensure high is greater than or equal to max(open, close)
        if "high" in self.data.columns:
            assert (self.data["high"] >= self.data[["open", "close"]].max(axis=1)).all(), \
                "High price must be greater than or equal to open and close prices."

        # Ensure low is less than or equal to min(open, close)
        if "low" in self.data.columns:
            assert (self.data["low"] <= self.data[["open", "close"]].min(axis=1)).all(), \
                "Low price must be less than or equal to open and close prices."

    def get_summary(self):
        """
        Return a summary of the bar data as a dictionary.
        """
        return {
            "symbol": self.symbol,
            "start": self.data.index.min(),
            "end": self.data.index.max(),
            "num_bars": len(self.data),
            "columns": list(self.data.columns),
        }
