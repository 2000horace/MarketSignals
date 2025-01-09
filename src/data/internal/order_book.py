import pandas as pd
from .base_data import BaseData

class OrderBookData(BaseData):
    """
    A class representing order book data, initialized with DataFrames for bids and asks.
    """

    def __init__(self, symbol: str, bbo_df: pd.DataFrame, msg_df: pd.DataFrame = None):
        """
        Initialize the order book data object.

        :param symbol: The asset symbol.
        :param timestamp: The timestamp of the order book snapshot.
        :param bbo_df: A DataFrame containing the ask price, ask quantity, bid price, bid quantity levels.
        :param msg_df: A DataFrame containing the modifications to the order book (assumed to match snapshot).
        """
        
        self.bbo_df = bbo_df
        self.msg_df = None if msg_df.empty else msg_df

        self._validate_dataframes()
        self._assign_levels()

        super().__init__(symbol, pd.concat([bbo_df, msg_df], axis=1))

    def _validate_dataframes(self):
        """
        Validate the structure of the bids and asks DataFrames.
        TODO: change validation specific for bbo and msg
        """
        pass

    def validate(self):
        """
        Validate the order book data object.
        TODO: add checks
        """
        pass

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
