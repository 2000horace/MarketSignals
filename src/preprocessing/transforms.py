import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class RollingMean(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate rolling mean for each column in a DataFrame.
    Conforms to sklearn pipeline structure.
    """

    def __init__(self, rolling_window_minutes: int = 5):
        """
        Initialize the RollingMean transformer.

        :param rolling_window_minutes: Size of the rolling window in minutes.
        """
        self.rolling_window_minutes = rolling_window_minutes

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
        Transform the input DataFrame by applying a rolling mean.

        :param X: The input DataFrame, with a datetime index.
        :return: Transformed DataFrame with rolling means.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        
        rolling_window = f"{self.rolling_window_minutes}T"
        return X.rolling(rolling_window).mean()