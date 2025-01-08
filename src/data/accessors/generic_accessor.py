import pandas as pd

from abc import ABC, abstractmethod

__all__ = [
    'GenericAccessor'
]


class GenericAccessor(ABC):
    """
    Abstract base class for data accessors.

    All data accessor classes should inherit from this class and implement
    the `get_data` method, which retrieves data and returns it as a pandas DataFrame.
    """

    @abstractmethod
    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Retrieve data and return it as a pandas DataFrame.

        :param args: Positional arguments specific to the accessor implementation.
        :param kwargs: Keyword arguments specific to the accessor implementation.
        :return: A pandas DataFrame containing the retrieved data.
        """
        raise NotImplementedError
