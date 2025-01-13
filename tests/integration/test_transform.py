import unittest

from datetime import date
from src.preprocessing.transform import *
from src.data.accessors import OmiArcticAccessor, OrderBookData

class TestLOBMultivariateRegression(unittest.TestCase):
    
    omi_adb = OmiArcticAccessor('lobster')
    _DEF_TICKERS = ['MSFT', 'AAPL']
    _DEF_LIBS = ['lobster-mbp-10']

    tickers = [_DEF_TICKERS] * len(_DEF_LIBS)
    res = omi_adb.get_raw_data(lib_names=_DEF_LIBS, tickers=tickers, dt_filter=(date(2020, 1, 8), date(2020, 1, 9)))
    ob_dict = {}

    for _t in _DEF_TICKERS:
        ob_dict.update({_t: OmiArcticAccessor._parse_mbp_and_trades_to_internal(_t, res.get(('lobster-mbp-10', _t)))})


    def test_bucket_transformation(self):
        pass