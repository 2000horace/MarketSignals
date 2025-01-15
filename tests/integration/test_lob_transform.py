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

        # assert required columns
        self.assertTrue(set(expected_cols).issubset(set(res.columns)))

        # assert number of data points remain the same
        self.assertEqual(df.shape[0], res.shape[0])

    def test_value_after_processing(self):
        """
        First row of MSFT.

        nanoseconds_since_midnight    34200008588129
        event                                      2
        id                                   5903403
        q                                        900
        p                                     158.25
        dir                                        1
        askp1                                 158.97
        askq1                                      8
        bidp1                                 158.94
        bidq1                                    200
        askp2                                 158.98
        askq2                                    100
        bidp2                                 158.76
        bidq2                                      5
        askp3                                  159.0
        askq3                                     64
        bidp3                                 158.69
        bidq3                                      1
        askp4                                  159.1
        askq4                                      2
        bidp4                                 158.65
        bidq4                                    300
        askp5                                 159.15
        askq5                                    233
        bidp5                                 158.52
        bidq5                                    397
        askp6                                 159.18
        askq6                                     40
        bidp6                                  158.5
        bidq6                                    100
        askp7                                  159.2
        askq7                                    101
        bidp7                                 158.48
        bidq7                                      1
        askp8                                 159.21
        askq8                                    587
        bidp8                                 158.38
        bidq8                                     25
        askp9                                 159.25
        askq9                                      2
        bidp9                                  158.3
        bidq9                                    605
        askp10                                 159.3
        askq10                                    50
        bidp10                                158.25
        bidq10                                   100
        event_str                             CANCEL
        Name: 2020-01-08 09:30:00, dtype: object
        """
        META_LEVELS = 4
        pass


class TestMetaLevelsByPriceImpact(unittest.TestCase):

    def test_appropriate_shape_after_resampling(self):
        META_LEVELS = 4

        df = ob_dict.get('MSFT').bbo_df.copy()
        df = df.resample('1T').first()

        transform = MetaLevelsByPriceImpact(META_LEVELS)
        res = transform.fit_transform(df)

        print(res)

        # assert length of df unch.
        self.assertEqual(len(df), len(res))

        expected_cols = [f'meta_askq{i}' for i in range(1, META_LEVELS+1)] 
        expected_cols += [f'meta_bidq{i}' for i in range(1, META_LEVELS+1)]
        expected_cols += [f'meta_askp{i}' for i in range(1, META_LEVELS+1)]
        expected_cols += [f'meta_bidp{i}' for i in range(1, META_LEVELS+1)]

        # assert required columns
        self.assertTrue(set(expected_cols).issubset(set(res.columns)))
        print(res.tail())

        # assert number of data points remain the same
        self.assertEqual(df.shape[0], res.shape[0])