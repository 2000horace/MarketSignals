import unittest

from datetime import date
from src.data.accessors import *

class TestOmiArcticDB(unittest.TestCase):
    
    omi_adb = OmiArcticAccessor('lobster')
    _DEF_TICKER = 'MSFT'
    _DEF_LIBS = ['lobster-mbp-10', 'lobster-trades']

    def test_get_libraries(self):
        print(TestOmiArcticDB.omi_adb.get_libraries())

    def test_get_symbol_versions(self):
        # check library symbols and metadata
        TestOmiArcticDB.omi_adb.get_symbol_versions('lobster-trades', 'MSFT')

    def test_fetch_lob(self):
        # fetch the data
        tickers = [[TestOmiArcticDB._DEF_TICKER]] * len(TestOmiArcticDB._DEF_LIBS)
        res = TestOmiArcticDB.omi_adb.get_raw_data(lib_names=TestOmiArcticDB._DEF_LIBS, 
                                                   tickers=tickers, 
                                                   dt_filter=(date(2020, 1, 8), date(2020, 1, 9)))

        keys = [(lib_name, TestOmiArcticDB._DEF_TICKER) for lib_name in TestOmiArcticDB._DEF_LIBS]
        self.assertListEqual(keys, list(res.keys()))

    def test_parse_data_into_order_book_data_object(self):
        tickers = [[TestOmiArcticDB._DEF_TICKER]] * len(TestOmiArcticDB._DEF_LIBS)
        res = TestOmiArcticDB.omi_adb.get_raw_data(lib_names=TestOmiArcticDB._DEF_LIBS, 
                                                   tickers=tickers, 
                                                   dt_filter=(date(2020, 1, 8), date(2020, 1, 9)))

        ob_data = OmiArcticAccessor._parse_mbp_and_trades_to_internal(TestOmiArcticDB._DEF_TICKER, 
                                                                      res.get(('lobster-mbp-10', TestOmiArcticDB._DEF_TICKER)),
                                                                      res.get(('lobster-trades', TestOmiArcticDB._DEF_TICKER)))