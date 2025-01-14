import unittest

from datetime import date
from src.preprocessing.lob_transforms import *
from src.preprocessing.clean import Resample
from src.data.accessors import OmiArcticAccessor, OrderBookData
from sklearn.pipeline import Pipeline


"""
Module-level definitions and data usage
"""
print('Loading data.....')
omi_adb = OmiArcticAccessor('lobster')
_DEF_TICKERS = ['MSFT']
_DEF_LIBS = ['lobster-mbp-10']

tickers = [_DEF_TICKERS] * len(_DEF_LIBS)
res = omi_adb.get_raw_data(lib_names=_DEF_LIBS, tickers=tickers, dt_filter=(date(2020, 1, 8), date(2020, 1, 9)))

ob_dict = {}
for _t in _DEF_TICKERS:
    ob_dict.update({_t: OmiArcticAccessor._parse_mbp_and_trades_to_internal(_t, res.get(('lobster-mbp-10', _t)))})
    print('\tloaded {}'.format(_t))


# class TestLOBMultivariateRegression(unittest.TestCase):
    
#     omi_adb = OmiArcticAccessor('lobster')
#     _DEF_TICKERS = ['MSFT', 'AAPL']
#     _DEF_LIBS = ['lobster-mbp-10']

#     tickers = [_DEF_TICKERS] * len(_DEF_LIBS)
#     res = omi_adb.get_raw_data(lib_names=_DEF_LIBS, tickers=tickers, dt_filter=(date(2020, 1, 8), date(2020, 1, 9)))
#     ob_dict = {}

#     for _t in _DEF_TICKERS:
#         ob_dict.update({_t: OmiArcticAccessor._parse_mbp_and_trades_to_internal(_t, res.get(('lobster-mbp-10', _t)))})


#     def test_bucket_transformation(self):
#         pass


class TestMetaLevelsByVolume(unittest.TestCase):

    def test_appropriate_shape_after_resampling(self):
        META_LEVELS = 4

        df = ob_dict.get('MSFT').bbo_df.copy()
        df = df.resample('1T').first()

        transform = MetaLevelsByVolume(META_LEVELS)
        res = transform.fit_transform(df)

        # assert length of df unch.
        self.assertEqual(len(df), len(res))

        expected_cols = [f'meta_askq{i}' for i in range(1, META_LEVELS+1)] 
        expected_cols += [f'meta_bidq{i}' for i in range(1, META_LEVELS+1)]
        expected_cols += [f'meta_askp{i}' for i in range(1, META_LEVELS+1)]
        expected_cols += [f'meta_bidp{i}' for i in range(1, META_LEVELS+1)]

        self.assertTrue(set(expected_cols).issubset(set(res.columns)))

        print(res.head())
        print(res.tail())