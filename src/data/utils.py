import pandas as pd
import numpy as np
import warnings

from src.data.internal.base_data import BaseData
from datetime import date, datetime
from typing import Union, Dict, Type, Tuple, Hashable

"""
Type Definitions
"""
DateTimeType = Union[date, datetime, pd.Timestamp, np.datetime64]
DateRange = Tuple[DateTimeType, DateTimeType]

DfDictType = Dict[Hashable, pd.DataFrame]
InternalDataType = Type[BaseData]
InternalDataDictType = Dict[str, InternalDataType]

"""
Variable Definitions
"""
LOBSTER_EVENT_MAP = {
    1: 'ADD',
    2: 'CANCEL',
    3: 'DEL',
    4: 'TRADE_V',
    5: 'TRADE_H',
    6: 'AUC',
    7: 'HALT'
}

"""
Custom Warnings
"""
class MissingDataWarning(Warning):
    """Warning raised when a pandas DataFrame is empty."""
    pass

def warn_empty_df(df: pd.DataFrame, message: str = "The DataFrame is empty.") -> None:
    if df.empty:
        warnings.warn(message, MissingDataWarning)