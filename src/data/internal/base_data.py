import pandas as pd

class BaseData:
    """
    A generic data object representing financial data.
    """

    def __init__(self, symbol: str, df: pd.DataFrame):
        """
        Initialize the base data object.

        :param symbol: The asset symbol (e.g., 'AAPL', 'BTC-USD').
        :param timestamp: The timestamp of the data.
        """
        self.symbol = symbol
        self.df = df

    def validate(self):
        """
        Validate the data object.
        """
        raise NotImplementedError("This method should be implemented in child classes.")