#import packages
import numpy as np
import pandas as pd

class Data(object):

    '''
    This class can be used to process the data so it can be used with several algorithms. 
    Once this class is  initialized it will prepare the dataset. 
    - 'data' consists of a Pandas dataframe which represents your dataset
    - 'Features' represents a list of columns.
    - 'cols_to_init' a list of string values which are initialized as new columns (values will be 
    set to numpy's version of NaN)
    - 'test_size' size of the test dataset (train_data will be determined by calculating 1-test_size)
    '''

    def __init__(self, data, features, cols_to_init=[], test_size=0.6, class_col=None):
        self.data = data
        self.test_size = test_size
        self.features = features
        self.cols_to_init = cols_to_init
        self.train_data, self.test_data = self.__process()

    def __process(self):
        train_data = self.data.sample(frac=(1-self.test_size))
        test_data = data[data.index.notin(test_data.index.values)]

        return train_data, test_data