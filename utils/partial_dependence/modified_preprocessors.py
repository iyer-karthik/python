# -*- coding: utf-8 -*-
# -*- author: Karthik Iyer -*-
"""
Most scikit-learn transformers output a numpy ndarray regardless of the
original datatype. If put in a pipeline this datatype is lost. In this script
we aim modify the transform() method for some of these transformers so that
they return a pandas dataframe if the fit() method is called on a dataframe. 
This is especially important if we wish to do some feature importance analysis
later on. 

A manual solution would be to use sklearn's transform() method and assign
the transformed data back into a dataframe. If we want to stick the usual
transformers into a pipeline then we need to modify the behavior of these
transformers, which is what this .py file does. 

TO DO: Add implementations for other preprocessors.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def custom_transform(obj, X):
    """
    Modifies scikit-learn's pre-processors/ transformers return type 
    for `transform()` method
    
    Custom implementation that modifies the return type of 
    some of scikit-learn's preprocessors. The default return type for 
    `transform()` method for scklearn's preprocessors/ transformers 
    is a numpy ndarray, regardless of how it was fit with.  This helper function 
    changes that behavior slightly and returns a pandas dataframe if the preprocessor/
    transformer was fit with a dataframe, otherwise it returns
    a numpy ndarray. 
    
    
    Parameters
    ----------
    obj : An instance of an object from sklearn.preprocessing or sklearn.impute
        
    X : array_like
        Input data that will be transformed
    
    """
    if isinstance(X, pd.core.frame.DataFrame):
        z = super(type(obj), obj).transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)
    return super(type(obj), obj).transform(X)


class BinarizerDf(Binarizer):
    """Wrapper around Binarizer

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.randint(0,100,size=(10, 4)), 
                          columns=list('ABCD'))
        A   B   C   D
    0  51  51  61  35
    1  35  81  67   8
    2  78  37  68  41
    >>> transformer = BinarizerDf(threshold=40.0).fit(df) # fit does nothing.
    >>> transformer
    BinarizerDf(copy=True, threshold=40.0)
    >>> transformer.transform(df)
        A  B  C  D
    0   1  1  1  0
    1   0  1  1  0
    2   1  0  1  1
    """

    def __init__(self, copy=True, threshold=0.0):
        super().__init__(copy=copy, threshold=threshold)

    def transform(self, X):
        return custom_transform(self, X)


class MaxAbsScalerDf(MaxAbsScaler):
    """Wrapper around MaxAbsScaler"""
    
    def __init__(self, copy=True):
        super().__init__(copy=copy)
    
    def transform(self, X):
        return custom_transform(self, X)


class MinMaxScalerDf(MinMaxScaler):
    """Wrapper around MinMaxScaler"""

    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__(copy=copy, feature_range=feature_range)
    
    def transform(self, X):
        return custom_transform(self, X)


class StandardScalerDf(StandardScaler):
    """Wrapper around StandardScaler"""

    def __init__(self, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
    
    def transform(self, X):
        return custom_transform(self, X)

class SimpleImputerDf(SimpleImputer):
    """Wrapper around SimpleImputer"""

    def __init__(self, copy=True, fill_value=None, missing_values=np.nan, 
                 strategy='mean', verbose=0):
        super().__init__(copy=copy, fill_value=fill_value,
                         missing_values=missing_values,
                         strategy=strategy,
                         verbose=verbose)
    
    def transform(self, X):
        return custom_transform(self, X)
