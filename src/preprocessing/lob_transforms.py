import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Union, List, Literal, Tuple
from src.data.utils import lob_depth

__all__ = [
    'LOBMultivariateRegressor',
    'MetaLevelsBySpread',
    'MetaLevelsByVolume',
    'MetaLevelsByPriceImpact'
]

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
    

class MetaLevelsBySpread(BaseEstimator, TransformerMixin):

    def __init__(self, max_meta_level: int = 10, keep_book_msgs: bool = False, meta_lvl_width: float = 2):
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
        self.max_meta_level = max_meta_level
        self.keep_book_msgs = keep_book_msgs
        self.meta_lvl_width = meta_lvl_width

        self.fit_metadata = None        # Contains any bookkeeping values after transformation

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
        new_df, interval_map = MetaLevelsBySpread.label_based_on_spread(X, self.meta_lvl_width, self.max_meta_level, self.keep_book_msgs)
        self.fit_metadata = interval_map
        return new_df

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


class MetaLevelsByVolume(BaseEstimator, TransformerMixin):

    def __init__(self, max_meta_levels: int = 5, keep_original: bool = False):
        self.keep_original = keep_original
        self.max_meta_levels = max_meta_levels

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        levels = lob_depth(X.columns)
        
        # Function to process one side
        def process_side(df, side, price_prefix, qty_prefix):
            # Melt the price and quantity columns into long format
            price_cols = [f"{price_prefix}{i}" for i in range(1, levels + 1)]
            qty_cols = [f"{qty_prefix}{i}" for i in range(1, levels + 1)]
            df_long = pd.DataFrame()

            for i in range(1, levels + 1):
                df_level = df[['nanoseconds_since_midnight']].copy()
                df_level['side'] = side
                df_level['level'] = i
                df_level['price'] = df[f"{price_prefix}{i}"]
                df_level['qty'] = df[f"{qty_prefix}{i}"]
                df_long = pd.concat([df_long, df_level], ignore_index=True)
            
            # Sort levels by price (asks increasing, bids decreasing)
            df_long = df_long.sort_values(by=['nanoseconds_since_midnight', 'price'], ascending=(side == 'ask'))

            # Compute cumulative volume
            df_long['cum_qty'] = df_long.groupby('nanoseconds_since_midnight')['qty'].cumsum()
        
            # Determine meta-level breakpoints (assign 0, 1, 2, 3... groupings)
            df_long['meta_level'] = df_long.groupby('nanoseconds_since_midnight')['cum_qty'].transform(
                lambda x: pd.qcut(x, self.max_meta_levels, labels=False, duplicates='drop')
            )
        
            # Calculate weighted average price for each meta-level
            df_meta = df_long.groupby(['nanoseconds_since_midnight', 'side', 'meta_level']).apply(
                lambda x: (x['price'] * x['qty']).sum() / x['qty'].sum() if x['qty'].sum() != 0 else 0).reset_index()
            df_meta.rename(columns={0: 'meta_price'}, inplace=True)

            # Sum quantities for each meta-level
            df_qty = df_long.groupby(['nanoseconds_since_midnight', 'side', 'meta_level'])['qty'].sum().reset_index()

            # Merge price and quantity
            df_meta = df_meta.merge(df_qty, on=['nanoseconds_since_midnight', 'side', 'meta_level'])

            return df_meta
        
        # Process ask and bid sides
        ask_df = process_side(X, 'ask', 'askp', 'askq')
        bid_df = process_side(X, 'bid', 'bidp', 'bidq')
        
        # Pivot to wide format for asks
        ask_pivot = ask_df.pivot_table(index='nanoseconds_since_midnight', columns=['side', 'meta_level'], values=['meta_price', 'qty'])
        ask_pivot.columns = ['meta_ask{}{}'.format('q' if c[0] == 'qty' else 'p', c[2]+1) for c in ask_pivot.columns]
        
        # Pivot to wide format for bids
        bid_pivot = bid_df.pivot_table(index='nanoseconds_since_midnight', columns=['side', 'meta_level'], values=['meta_price', 'qty'])
        bid_pivot.columns = ['meta_bid{}{}'.format('q' if c[0] == 'qty' else 'p', c[2]+1) for c in bid_pivot.columns]
        
        # Combine ask and bid pivots
        transformed_df = ask_pivot.join(bid_pivot).reset_index()

        # If keep original, join with original X as well
        if self.keep_original:
            transformed_df = X.join(transformed_df, on='nanoseconds_since_midnight').reset_index()

        # Inherit original datetime index
        transformed_df.index = X.index.copy()
        
        return transformed_df
    

class MetaLevelsByPriceImpact(BaseEstimator, TransformerMixin):

    def __init__(self, n_meta_levels: int = -1, manual_thr: List[float] = None, keep_original: bool = False):
        self.n_meta_levels = n_meta_levels
        self.manual_thr = manual_thr
        
        if self.n_meta_levels < 0:
            assert self.manual_thr is not None, "Must specify n_meta_levels or provide manual_thr, got neither."
        if manual_thr is not None:
            self.n_meta_levels = len(manual_thr)

        self.keep_original = keep_original

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        # No fitting required
        return self
    
    def _assign_meta_level_quantile(self, group) -> pd.DataFrame:
        # Determine price impact thresholds using quantiles, returns a dataframe of group assignments
        if group['cum_price_impact'].max() == 0:
            group['meta_level'] = 0
        else:
            quantiles = group['cum_price_impact'].quantile([i / self.n_meta_levels for i in range(1, self.n_meta_levels + 1)])
            thresholds = quantiles.tolist()
            group['meta_level'] = 0
            for i in range(1, self.n_meta_levels + 1):
                group.loc[group['cum_price_impact'] >= thresholds[i-1], 'meta_level'] = i
        return group

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        def process_side(df, side, price_prefix, qty_prefix, manual_thr):
            # Function to process one side (ask or bid)
            levels = []
            for i in range(1, 11):
                level_df = df[['nanoseconds_since_midnight']].copy()
                level_df['level'] = i
                level_df['price'] = df[f"{price_prefix}{i}"]
                level_df['qty'] = df[f"{qty_prefix}{i}"]
                levels.append(level_df)
            levels_df = pd.concat(levels, ignore_index=True)

            # Sort levels by price
            if side == 'ask':
                levels_df = levels_df.sort_values(by=['nanoseconds_since_midnight', 'price'])
            else:
                levels_df = levels_df.sort_values(by=['nanoseconds_since_midnight', 'price'], ascending=False)
            
            # Calculate individual price impacts
            levels_df['price_next'] = levels_df.groupby('nanoseconds_since_midnight')['price'].shift(-1)
            if side == 'ask':
                levels_df['price_impact'] = levels_df['price_next'] - levels_df['price']
            else:
                levels_df['price_impact'] = levels_df['price'] - levels_df['price_next']
            
            # Handle NaNs in price_next
            levels_df['price_impact'] = levels_df['price_impact'].fillna(0)

            # Calculate cumulative price impact
            levels_df['cum_price_impact'] = levels_df.groupby('nanoseconds_since_midnight')['price_impact'].cumsum()

            # Assign meta levels based on manual thresholds or quantiles
            if manual_thr is not None:
                # Ensure thresholds are sorted
                manual_thr_sorted = sorted(manual_thr)

                # Define bins including all possible cumulative impacts
                bins = [0] + manual_thr_sorted + [levels_df['cum_price_impact'].max()]
                labels = range(1, len(bins))
                levels_df['meta_level'] = pd.cut(levels_df['cum_price_impact'], bins=bins, labels=labels, right=False)

                # Fill NaNs with the highest meta level
                levels_df['meta_level'] = levels_df['meta_level'].fillna(max(labels))
            else:
                levels_df = levels_df.groupby('nanoseconds_since_midnight').apply(self._assign_meta_level_quantile)
            
            def _aggregate_meta(group: pd.DataFrame) -> pd.DataFrame:
                # Aggregate price and quantity for each meta-level
                meta_dfs = []
                for meta in range(1, self.n_meta_levels + 1):
                    meta_group = group[group['meta_level'] == meta]
                    if not meta_group.empty:
                        total_qty = meta_group['qty'].sum()
                        if total_qty > 0:
                            weighted_price = (meta_group['price'] * meta_group['qty']).sum() / total_qty
                        else:
                            weighted_price = 0
                        meta_df = pd.DataFrame({
                            'nanoseconds_since_midnight': [group.name],
                            f'meta_{side}p{meta}': [weighted_price],
                            f'meta_{side}q{meta}': [total_qty]
                        })
                        meta_dfs.append(meta_df)
                return pd.concat(meta_dfs, ignore_index=True)
            
            meta_df = levels_df.groupby('nanoseconds_since_midnight').apply(_aggregate_meta).reset_index(drop=True)
            return meta_df

        # Process ask and bid sides
        ask_meta = process_side(X, 'ask', 'askp', 'askq', self.manual_thr)
        bid_meta = process_side(X, 'bid', 'bidp', 'bidq', self.manual_thr)

        # forward/backward fill based on nanoseconds_since_midnight
        ask_meta = ask_meta.groupby('nanoseconds_since_midnight').first()
        bid_meta = bid_meta.groupby('nanoseconds_since_midnight').first()

        # Combine ask and bid meta-levels
        combined_meta = ask_meta.merge(bid_meta, on='nanoseconds_since_midnight', how='outer').reset_index()

        if self.keep_original:
            # Merge back into the original DataFrame
            combined_meta = X.merge(combined_meta, on='nanoseconds_since_midnight', how='left')

        combined_meta.index = X.index
        
        return combined_meta