import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Union, List

__all__ = [
    'RollingMean',
    'LOBMultivariateRegressor'
]

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


class LOBMultivariateRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 target_df: pd.DataFrame, 
                 factor_dfs: Dict[str, pd.DataFrame], 
                 columns_to_use: Union[List[str], str], 
                 depth: int = 10):
        """
        Initialize the LOBMultivariateRegressor.

        Parameters:
        target_df: DataFrame representing the dependent variable (Y).
        factor_dfs: List of DataFrames representing independent variables (Xs).
        columns_to_use: List of columns to use, or a pattern for generating columns (e.g., 'askq{i}').
        depth: Number of levels to consider if columns_to_use is a pattern.
        """
        self.target_df = target_df
        self.factor_dfs = factor_dfs
        self.models = {}
        self.depth = depth
        self.X_columns_to_use = None

        if isinstance(columns_to_use, str):
            self.columns_to_use = [columns_to_use.format(i=i) for i in range(1, depth + 1)]
        elif isinstance(columns_to_use, list):
            self.columns_to_use = columns_to_use
        else:
            raise ValueError("columns_to_use must be a list or a formattable string.")

    def fit(self):
        """
        Fit the multivariate regression model between two dataframes using OLS from statsmodels.
        """
        # Add prefixes to all dataframes
        # Add the ticker as a prefix to the columns
        X_columns_to_use = []
        for ticker, _x in self.factor_dfs.items():
            _x = _x[self.columns_to_use].copy()
            _x.columns = [f"{ticker}_{col}" for col in _x.columns]
            X_columns_to_use += list(_x.columns)

        self.X_columns_to_use = X_columns_to_use

        # Filter only the relevant columns
        X = pd.concat(self.factor_dfs.values(), axis=1)
        X = sm.add_constant(X)  # Add intercept
        Y = self._y[self.columns_to_use]

        # Fit an OLS model for each meta level
        for col in self.columns_to_use:
            model = sm.OLS(Y[col], X).fit()
            self.models[col] = model

        return self

    def transform(self):
        """
        Apply the regression model and return the residuals as a DataFrame.

        Parameters:
        df_a: DataFrame representing the first order book.
        df_b: DataFrame representing the second order book.

        Returns:
        residuals: DataFrame of residuals with the same structure as the input order books.
        """
        if self.columns_to_use is None or not self.models:
            raise ValueError("The model must be fitted before calling transform().")

        # Extract relevant columns
        X = pd.concat(self.factor_dfs.values(), axis=1)
        X = sm.add_constant(X)  # Add intercept
        Y = self._y[self.columns_to_use]

        # Compute residuals for each meta level
        residuals = {}
        for col in self.columns_to_use:
            Y_pred = self.models[col].predict(X)
            residuals[col] = Y[col] - Y_pred

        # Convert residuals back to a DataFrame
        residuals_df = pd.DataFrame(
            residuals,
            index=X.index  # Preserve the index of df_a
        )

        return residuals_df
    
    def get_model_parameters(self):
        """
        Extract regression parameters for each model.

        Returns:
        parameters: Dictionary where keys are column names and values are dictionaries containing
                    model summaries, coefficients (betas), p-values, and R-squared values.
        """
        if not self.models:
            raise ValueError("The model must be fitted before extracting parameters.")

        parameters = {}
        for col, model in self.models.items():
            parameters[col] = {
                "summary": model.summary(),
                "params": model.params,
                "pvalues": model.pvalues,
                "rsquared": model.rsquared
            }

        return parameters
    

class MetaLevels(BaseEstimator, TransformerMixin):
    def __init__():
        pass
    