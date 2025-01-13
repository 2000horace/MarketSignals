import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Union, List, Literal, Tuple
from src.data.utils import UndefinedKwargDefaultUsedWarning

__all__ = [
    'RollingMean',
    'LOBMultivariateRegressor',
    'MetaLevels',
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

    def __init__(self, mode: Literal['spread'] = 'spread', max_meta_level: int = 10, keep_book_msgs: bool = False, **kwargs):
        """
        meta_lvl_width: float
            The width of each meta level is calculated by meta_width = avg_spread * meta_lvl_width.
            Subsequent meta levels for ask prices would be [0, meta_width), [meta_width, 2*meta_width)...
            Levels that fall into the first bucket will have their quantities combined.

        max_meta_level: int
            if -1, will not impose bound. Otherwise any meta level > max_meta_level will be combined to one value

        keep_book_msgs: bool
            if True, will keep message updates such as price, quantity, update_type etc.
        """
        self.mode = mode
        self.max_meta_level = max_meta_level
        self.keep_book_msgs = keep_book_msgs

        self.fit_metadata = None        # Contains any bookkeeping values after transformation

        if self.mode == 'spread':
            self.meta_lvl_width = kwargs.get('meta_lvl_width', 2)
            if 'meta_lvl_width' not in kwargs:
                raise UndefinedKwargDefaultUsedWarning('meta_lvl_width', 2)
        else:
            raise ValueError('Unknown mode "{}".'.format(mode))


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
        Creates meta levels on a limit orderbook and returns the dataframe with the new meta levels.

        :param X: The input DataFrame, with a datetime index.
        :return: Transformed DataFrame with rolling means.
        """
        if self.mode == 'spread':
            new_df, interval_map = MetaLevels.label_based_on_spread(X, self.meta_lvl_width, self.max_meta_level, self.keep_book_msgs)
            self.fit_metadata = interval_map
            return new_df
        else:
            raise ValueError('Unknown mode "{}".'.format(self.mode))

    @staticmethod
    def label_based_on_spread(book: pd.DataFrame, 
                              meta_lvl_width: float = 2, 
                              max_meta_level: int = 10, 
                              keep_book_msgs: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.Interval]]:
        """
        book: pd.DataFrame
            Dataframe containing the order book. Assume to have columns ask and bid prices up to 10 levels.

        meta_lvl_width: float
            The width of each meta level is calculated by meta_width = avg_spread * meta_lvl_width.
            Subsequent meta levels for ask prices would be [0, meta_width), [meta_width, 2*meta_width)...
            Levels that fall into the first bucket will have their quantities combined.

        max_meta_level: int
            if -1, will not impose bound. Otherwise any meta level > max_meta_level will be combined to one value

        keep_book_msgs: bool
            if True, will keep message updates such as price, quantity, update_type etc.
        """

        book['mid'] = (book['askp1'] + book['bidp1'])/2

        # 1. calculate average spread in bps
        spread_df = ((book['askp1'] - book['bidp1'])/book['mid'])
        avg_spread = spread_df.mean()

        # 2. compute meta level width
        #   Define interval size based on avg_spread and meta_lvl_width
        interval_size = avg_spread * meta_lvl_width

        #   DataFrame to store results with aggregated meta levels
        meta_asks, meta_bids = {}, {}
        meta_name_to_interval = {}

        # 3. map meta levels to intervals and store in dataframe
        ask_price_cols = [c for c in book.columns if c.startswith('askp')]
        bid_price_cols = [c for c in book.columns if c.startswith('bidp')]
        ask_qty_cols = [c for c in book.columns if c.startswith('askq')]
        bid_qty_cols = [c for c in book.columns if c.startswith('bidq')]

        #   Calculate ask meta level quantities and boundaries
        ask_distances = book[ask_price_cols].div(book['mid'], axis=0) - 1
        ask_levels = (ask_distances / interval_size).apply(np.floor).astype(int)
        ask_levels.rename(columns=dict(zip(ask_price_cols, ask_qty_cols)), inplace=True)
        for meta_lvl in np.unique(ask_levels.values.flatten()):
            lvl_name = 'meta_askq{}'.format(meta_lvl+1)
            meta_asks.update({lvl_name: book[ask_qty_cols].where(ask_levels == meta_lvl).sum(axis=1)})
            if max_meta_level == -1 or meta_lvl < max_meta_level-1:
                interval = pd.Interval(left=meta_lvl * interval_size, 
                                    right=(meta_lvl+1) * interval_size, 
                                    closed='left')
                meta_name_to_interval.update({lvl_name: interval})
            else:
                interval = pd.Interval(left=meta_lvl * interval_size,
                                    right=float('inf'),
                                    closed='left')
                meta_name_to_interval.update({lvl_name: interval})
                break
        
        del ask_distances, ask_levels

        #   Calculate bid meta level quantities and boundaries
        bid_distances = 1 - book[bid_price_cols].div(book['mid'], axis=0)
        bid_levels = (bid_distances / interval_size).apply(np.floor).astype(int)
        bid_levels.rename(columns=dict(zip(bid_price_cols, bid_qty_cols)), inplace=True)
        for meta_lvl in np.unique(bid_levels.values.flatten()):
            lvl_name = 'meta_bidq{}'.format(meta_lvl+1)
            meta_bids.update({lvl_name: book[bid_qty_cols].where(bid_levels == meta_lvl).sum(axis=1)})
            if max_meta_level == -1 or meta_lvl < max_meta_level-1:
                interval = pd.Interval(left= -(meta_lvl+1) * interval_size, 
                                    right= -meta_lvl * interval_size, 
                                    closed='right')
                meta_name_to_interval.update({lvl_name: interval})
            else:
                interval = pd.Interval(left=float('-inf'),
                                    right= -meta_lvl * interval_size,
                                    closed='right')
                meta_name_to_interval.update({lvl_name: interval})
                break

        if keep_book_msgs:
            # q	p	dir	notes	event_str
            msg_cols = ['q', 'p', 'dir', 'notes', 'event_str', 'mid']
            return pd.concat([pd.DataFrame(meta_bids), pd.DataFrame(meta_asks), book[msg_cols]], axis=1), meta_name_to_interval
        else:
            return pd.concat([pd.DataFrame(meta_bids), pd.DataFrame(meta_asks)], axis=1), meta_name_to_interval