import pandas as pd

from .base_data import BaseData
from sklearn.pipeline import Pipeline
from typing import Optional


_REQUIRED_COLUMNS = [
            'event', 'id', 'q', 'p', 'dir', 'event_str'
        ] + [f"{side}{attr}{level}" for level in range(1, 11) for side, attr in [('askp', ''), ('askq', ''), ('bidp', ''), ('bidq', '')]]


class OrderBookData(BaseData):
    """
    A class representing order book data, initialized with DataFrames for bids and asks.
    """

    def __init__(self, symbol: str, bbo_df: pd.DataFrame, msg_df: pd.DataFrame = pd.DataFrame()):
        """
        Initialize the order book data object.

        :param symbol: The asset symbol.
        :param timestamp: The timestamp of the order book snapshot.
        :param bbo_df: A DataFrame containing the ask price, ask quantity, bid price, bid quantity levels.
        :param msg_df: A DataFrame containing the modifications to the order book (assumed to match snapshot).
        """
        
        self.bbo_df = bbo_df
        self.msg_df = msg_df

        self.validate()

        super().__init__(symbol, pd.concat([bbo_df, msg_df], axis=1))

    def validate(self):
        """
        Validate the order book data object.
        """
        # 1. Check required columns
        missing_columns = set(_REQUIRED_COLUMNS) - set(self.bbo_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 2. Check data types
        numeric_columns = [
            'nanoseconds_since_midnight', 'q', 'p'
        ] + [f"{side}{attr}{level}" for level in range(1, 11) for side, attr in [('askp', ''), ('askq', ''), ('bidp', ''), ('bidq', '')]]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(self.bbo_df[col]):
                raise TypeError(f"Column {col} must have numeric data type.")

        # 3. Check non-negative prices and sizes
        price_columns = [f"{side}p{level}" for level in range(1, 11) for side in ['ask', 'bid']]
        size_columns = [f"{side}q{level}" for level in range(1, 11) for side in ['ask', 'bid']]

        if not ((self.bbo_df[price_columns] >= 0) | self.bbo_df[price_columns].isna()).all().all():
            raise ValueError("Prices must be non-negative or NaN.")
        if not ((self.bbo_df[size_columns] >= 0) | self.bbo_df[size_columns].isna()).all().all():
            raise ValueError("Sizes must be non-negative.")

        # 4. Check ask and bid price ordering
        for level in range(1, 10):  # Only check up to the second-last level
            next_level = level + 1
            ask_col = f"askp{level}"
            next_ask_col = f"askp{next_level}"
            bid_col = f"bidp{level}"
            next_bid_col = f"bidp{next_level}"

            if not (self.bbo_df[ask_col] <= self.bbo_df[next_ask_col]).all():
                raise ValueError(f"Asks are not increasing at levels {level} and {next_level}.")
            if not (self.bbo_df[bid_col] >= self.bbo_df[next_bid_col]).all():
                raise ValueError(f"Bids are not decreasing at levels {level} and {next_level}.")

        # 5. Validate presence of size if price exists
        for level in range(1, 11):
            ask_price_col = f"askp{level}"
            ask_size_col = f"askq{level}"
            bid_price_col = f"bidp{level}"
            bid_size_col = f"bidq{level}"

            ask_missing_size = self.bbo_df[(self.bbo_df[ask_price_col] > 0) & (self.bbo_df[ask_size_col] <= 0)]
            bid_missing_size = self.bbo_df[(self.bbo_df[bid_price_col] > 0) & (self.bbo_df[bid_size_col] <= 0)]

            if not ask_missing_size.empty:
                raise ValueError(f"Ask prices exist without corresponding sizes at depth {level}.")
            if not bid_missing_size.empty:
                raise ValueError(f"Bid prices exist without corresponding sizes at depth {level}.")

        print("Validation passed: Order book data is valid.")

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
    
    def fit_transform(self, pipeline, inplace = True, *args, **kwargs) -> Optional[pd.DataFrame]:
        if not isinstance(pipeline, Pipeline):
            raise TypeError("The 'pipeline' argument must be an instance of sklearn.pipeline.Pipeline.")

        # Apply the pipeline
        transformed = pipeline.fit_transform(self.bbo_df)

        # Ensure the output is a DataFrame
        if not isinstance(transformed, pd.DataFrame):
            raise ValueError("The pipeline must return a pandas DataFrame.")

        # Replace the current dataframe or return the transformed one
        if inplace:
            self.bbo_df = transformed
        else:
            return transformed
