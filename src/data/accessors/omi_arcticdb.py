import pandas as pd

from arcticdb import Arctic
from typing import Literal, List
from generic_accessor import GenericAccessor


S3_URI_FORMAT = r's3s://omi-rapid-graid.omirapid.oxford-man.ox.ac.uk:{}?region=omi-eaglehouse&port=9000&access={}&secret={}'
_ACCESS_KEY = r'ISvPoXxEfZq8R5NrhteF'
_SECRET_KEY = r'q19I0b8lQvPEwAsJANs3lFUFyVDJpzjpzTTmZTb0'

class OmiArcticAccessor(GenericAccessor):

    def __init__(self, db_name: Literal['lobster', 'pinnacle']):
        super().__init__()
        self.db_name = db_name

        self.server_uri: str = self._fn_str_s3_uri_ro(db_name)
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
        return Arctic(self.fn_str_s3_uri_ro())    #we return an arctic db instance containing the chosen data set

    def get_libraries(self) -> List[str]:
        return self._db_instance.list_libraries()

    def get_data(self, 
                 lib_names: List[str], 
                 tickers: List[str], 
                 col_filter: List[str] = None,
                 dt_filter = None) -> pd.DataFrame:
        """
        The next function fetches symbols from different libraries and collates them into one big pandas data frame:
        """
        dfs = []  # To store the individual DataFrames    
        le = len(lib_names)

        for i in range(le):
            library_name = lib_names[i]  # Replace with your library name
            library = self._db_instance[library_name]
            
            # Retrieve the symbols
            symbol_names = tickers[i]  # Replace with your symbol names
            if not (col_filter is None):
                column_names = col_filter[i]
            else:
                column_names = None
        
            for str_symbol_name in symbol_names:
                symbol = library.read(str_symbol_name,
                                      columns=column_names,
                                      date_range=dt_filter)
                
                df = pd.DataFrame(symbol.data).add_prefix(str_symbol_name+'_')  # Assuming Date and Value columns
                dfs.append(df)

        # Concatenate the DataFrames into a combined one which is returned
        return pd.concat(dfs, axis=1)


if __name__ == '__main__':
    from datetime import date

    # 1) CONNECTING TO THE DATA BASE:
    # First we access the arctic db data base wish to access and create an arctic instance (adb) connected to it:
    omi_adb = OmiArcticAccessor('lobster')
    print('Arctic instance instantiated.')

    # 2) Let's see what libraries (subtypes of data) the lobster data set has:
    print(omi_adb.get_libraries())

    # 3) FETCH THE DATA :
    df = omi_adb.get_data(lib_names=["lobster-mbp-10"],
                          tickers=['COKE'],
                          dt_filter=[(date(2020, 1, 6), date(2020, 1, 7))])

    print(df.head())
    print(f'The columns are:{df.columns.values}')
    print(df.describe())
