import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List

__all__ = [
    'Resample',
    'ColumnFilter'
]


class Resample(BaseEstimator, TransformerMixin):
    """
    Transformer to resample a DataFrame.
    Conforms to sklearn pipeline structure.
    """

    def __init__(self, window_length: Union[str, int], aggregation_method: str = "mean"):
        """
        Initialize the resampling transformer.

        Args:
            window_length (str or int): The length of the resampling window.
                - If the DataFrame has a DatetimeIndex, this should be a time-based string (e.g., "5T" for 5 minutes).
                - If the DataFrame has an integer-based index, this should be an integer (e.g., 10 for 10 rows).
            aggregation_method (str): The aggregation method to apply during resampling.
                Supported methods: "mean", "sum", "first", "last", "min", "max", etc.
        """
        self.window_length = window_length
        self.aggregation_method = aggregation_method

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit method (no action needed for this transformer).

        :param X: The input DataFrame.
        :param y: Optional target values (ignored).
        :return: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the input DataFrame.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Resampled data.
        """
        if isinstance(X.index, pd.DatetimeIndex):
            # Time-based resampling
            resampled_df = X.resample(self.window_length).agg(self.aggregation_method)
        else:
            # Integer-based resampling
            resampled_df = X.groupby(X.index // self.window_length).agg(self.aggregation_method)
        return resampled_df
    

class ColumnFilter(BaseEstimator, TransformerMixin):

    def __init__(self, col_filter: Union[List[str], pd.Index], keep: bool = True, lenient_match: bool = True):
        """
        Initialize the ColumnFilter transformer.

        Args:
            col_filter (list of strings or index): 
                columns to be filtered
            keep (bool): 
                - If True, will keep the columns indicated by col_filter
                - If False, will remove the columns indiated by col_filter
            lenient_match (bool):
                - If True, will ignore columns in col_filter that is not present
                - If False, will throw KeyError if col_filter consists of non-existent column name
        """

        self.col_filter = list(col_filter)
        self.keep = keep
        self.lenient_match = lenient_match

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        if self.lenient_match:
            self.col_filter = [c for c in self.col_filter if c in X.columns]

        if not self.keep:
            new_filter = [c for c in X.columns if c not in self.col_filter]
            self.col_filter = new_filter
            self.keep = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.col_filter].copy()
