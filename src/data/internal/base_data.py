import pandas as pd

from sklearn.pipeline import Pipeline
from typing import Optional

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
    
    def fit_transform(self, pipeline: Pipeline, inplace: bool = True, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        Transforms the stored dataframe(s) given the sklearn pipeline.

        :param pipeline: The sklearn.pipeline Pipeline object.
        :param inplace: bool. If True, will replace the self.df object. Else returns the transformed dataframe.
        """
        raise NotImplementedError()