import pandas as pd
import numpy as np

from arcticdb import Arctic
from arcticdb.version_store.library import Library
from typing import Literal, List

from src.data.accessors.generic_accessor import GenericAccessor, DfDictType, InternalDataDictType
from src.data.internal.order_book import OrderBookData
from src.data.utils import DateRange, warn_empty_df, LOBSTER_EVENT_MAP


S3_URI_FORMAT = r's3s://omi-rapid-graid.omirapid.oxford-man.ox.ac.uk:{}?region=omi-eaglehouse&port=9000&access={}&secret={}'
_ACCESS_KEY = r'ISvPoXxEfZq8R5NrhteF'
_SECRET_KEY = r'q19I0b8lQvPEwAsJANs3lFUFyVDJpzjpzTTmZTb0'

class OmiArcticAccessor(GenericAccessor):

    def __init__(self, db_name: Literal['lobster', 'pinnacle']):
        super().__init__()
        self.db_name = db_name

        self.server_uri: str = self._fn_str_s3_uri_ro()
        self._db_instance: Arctic = self._get_adb_instance()

    def _fn_str_s3_uri_ro(self) -> str:
        s3_uri = S3_URI_FORMAT.format(
            self.db_name,
            _ACCESS_KEY,
            _SECRET_KEY
        )
        return s3_uri

    def _get_adb_instance(self) -> Arctic:
        """
        Returns an arctic db instance for a specific data set (read_only)
        """
        return Arctic(self._fn_str_s3_uri_ro())    #we return an arctic db instance containing the chosen data set

    def get_libraries(self) -> List[str]:
        return self._db_instance.list_libraries()
    
    def get_library(self, lib_name: str) -> Library:
        return self._db_instance[lib_name]
    
    def get_lib_symbols(self, lib_name: str) -> List[str]:
        lib = self.get_library(lib_name)
        return sorted(lib.list_symbols())
    
    def get_symbol_versions(self, lib_name: str, symb: str) -> None:
        lib = self.get_library(lib_name)
        versions = lib.list_versions(symb)

        print(f"Versions for symbol '{symb}':", versions)

    def get_raw_data(self, 
                     lib_names: List[str], 
                     tickers: List[List[str]], 
                     col_filter: List[List[str]] = None,
                     dt_filter: DateRange = None,
                     _add_prefix: bool = False) -> DfDictType:
        """
        The next function fetches symbols from different libraries and returns dictionary of dataframes indexed by symbol
        """
        res = {}
        le = len(lib_names)

        for i in range(le):
            library_name = lib_names[i]  # Replace with your library name
            library: Library = self.get_library(library_name)
            
            # Retrieve the symbols
            symbol_names = tickers[i]  # Replace with your symbol names
            if not (col_filter is None):
                column_names = col_filter[i]
            else:
                column_names = None

            for str_symbol_name in symbol_names:
                df = library.read(str_symbol_name,
                                  columns=column_names,
                                  date_range=dt_filter)
                df = pd.DataFrame(df.data)
                warn_empty_df(df)

                if _add_prefix:
                    df = df.add_prefix(str_symbol_name+'_')  # Assuming Date and Value columns

                res.update({(library_name, str_symbol_name): df})

        # Concatenate the DataFrames into a combined one which is returned
        return res
    
    def get_data(self, tickers: List[str], col_filter: List[str] = None, dt_filter: DateRange = None) -> InternalDataDictType:
        pass

    def _parse_mbp_and_trades_to_internal(symbol: str, mbp_df: pd.DataFrame, msg_df: pd.DataFrame) -> InternalDataDictType:
        """
        Parse raw data into an OrderBookData object.

        Parameters:
        - data (pd.DataFrame): Raw dataframe containing order book data.
        - symbol (str): The symbol for the financial instrument.

        Returns:
        - OrderBookData: Initialized order book object.
        """
        # ------------ 1. Deal with mbp_df ----------------
        # Extract bid columns
        bid_columns = [col for col in mbp_df.columns if col.startswith('bid_price') or col.startswith('bid_size')]
        bids = mbp_df[bid_columns]
        
        # Rename bid columns to standardized names (bidp1, bidq1, ...)
        bid_rename_map = {f'bid_price_{i}': f'bidp{i}' for i in range(1, 11)}
        bid_rename_map.update({f'bid_size_{i}': f'bidq{i}' for i in range(1, 11)})
        bids.rename(columns=bid_rename_map, inplace=True)

        # Extract ask columns
        ask_columns = [col for col in mbp_df.columns if col.startswith('ask_price') or col.startswith('ask_size')]
        asks = mbp_df[ask_columns]

        # Rename ask columns to standardized names (askp1, askq1, ...)
        ask_rename_map = {f'ask_price_{i}': f'askp{i}' for i in range(1, 11)}
        ask_rename_map.update({f'ask_size_{i}': f'askq{i}' for i in range(1, 11)})
        asks.rename(columns=ask_rename_map, inplace=True)

        mbp_df.replace(9999999999, np.nan, inplace=True)            # indicates nothing happening on this level
        mbp_df.replace(-9999999999, np.nan, inplace=True)           # indicates nothing happening on this level

        # ------------ 2. Deal with mbp_df ----------------
        # remove 'ask_price_1' 'ask_size_1' 'bid_price_1' 'bid_size_1'
        msg_df = msg_df[[
            'nanoseconds_since_midnight', 
            'event', 
            'order_id', 
            'size', 
            'price',
            'direction']]
        
        # Parse event code to string action code
        msg_df['event_str'] = msg_df['event'].map(LOBSTER_EVENT_MAP)

        # Rename columns
        msg_df.columns = ['time', 'event', 'id', 'q', 'p', 'dir', 'notes']

        # ------------ 3. Merge bbos and messages ----------------
        # Merge with trade book, ensure both DataFrames use the same DatetimeIndex
        # if not mbp_df.index.equals(msg_df.index):
        #     raise ValueError("Indices of the two DataFrames do not match perfectly.")
        merged_data = pd.concat([mbp_df, msg_df], axis=1)

        # Create and return OrderBookData object
        return OrderBookData(symbol=symbol, timestamps=merged_data.index, bids=bids, asks=asks)


if __name__ == '__main__':
    from datetime import date

    # 1) CONNECTING TO THE DATA BASE:
    # First we access the arctic db data base wish to access and create an arctic instance (adb) connected to it:
    omi_adb = OmiArcticAccessor('lobster')

    # 2) Let's see what libraries (subtypes of data) the lobster data set has:
    print(omi_adb.get_libraries())

    # check library symbols and metadata
    omi_adb.get_symbol_versions('lobster-trades', 'MSFT')

    # 3) FETCH THE DATA :
    ticker = 'MSFT'
    libs = ['lobster-mbp-10', 'lobster-trades']
    tickers = [[ticker]] * len(libs)
    res = omi_adb.get_raw_data(lib_names=libs, tickers=tickers, dt_filter=(date(2020, 1, 8), date(2020, 1, 9)))

    print(list(res.keys()))

    # 4) PARSE INTO INTERNAL FORMAT
    ob_data = OmiArcticAccessor._parse_mbp_and_trades_to_internal(ticker, 
                                                                  res.get(('lobster-mbp-10', ticker)),
                                                                  res.get(('lobster-trades', ticker)))
    
    print(ob_data.to_dict())
