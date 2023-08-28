
# Libraries
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Tuple

# --------------------------- Read data files --------------------------- #
class DataLoader:
    """ An object to load and prepare data for AE

    Parameters
    ----------
    data_path : str
        Path to input data
    rand_state : dict
        A random number
    x_transform : bool
        To apply input data transformation

    Returns
    ----------
    None

    Example
    --------
    >>> DataLoader('data_path', 115, True)
        """
    def __init__(self, data_path: str, rand_state: int, 
                 x_transform: bool = False) -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        self.data                       = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        self.x_transform                = x_transform
        self.train                      = pd.DataFrame([])
        self.test                       = pd.DataFrame([])
        self.model_features = json.load(open('data/ae_model_feature_names.json'))
        
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),"model/")):
            os.mkdir(os.path.join(os.getcwd(),"model/"))

    def readFiles(self) -> pd.DataFrame:
        """ Read input for AE

        Returns
        ----------
        data : pd.DataFrame
            input data

        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')

        except:
            print('Wrong address or data format. Please use parquet file.')   
        return self.data
      
    # --------------------------- Split train and test --------------------------- #
    def splitData(self) -> pd.array:
        """ Split to train and test

        Returns
        ----------
        msk : np.array
            Index of splited data

        """
        df_mask = self.data[self.model_features]

        msk = np.random.rand(len(df_mask)) < 0.85
        self.train = df_mask[msk]
        self.test = df_mask[~msk]
        return msk

    # --------------------------- Transformation --------------------------- #

    def transformData(self, type: str ='power') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Apply tranformation to data
        
        Parameters
        ----------
        type: str
            Type of transformation
            Options:
            - "power" 
            - "quntile"

        Returns
        ----------
        train_x : pd.DataFrame
            Traning data
        test_x : pd.DataFrame
            Testing data

        """
        if self.x_transform:
            if type=='power':
                t_x = PowerTransformer()
            else:
                t_x = QuantileTransformer(
                    n_quantiles=500, output_distribution="normal", 
                    random_state=self.rand_state
                )
            scaler_x = StandardScaler()
            train_x = self.train.reset_index(drop=True)
            train_x_t = t_x.fit_transform(train_x)
            pickle.dump(t_x, open('model/'+'ae_train_x_normalization.pkl', "wb"))
            train_x_pt = scaler_x.fit_transform(train_x_t)
            pickle.dump(t_x, open('model/'+'ae_train_x_scaling.pkl', "wb"))
            train_x = pd.DataFrame(data=train_x_pt,
                    columns=train_x.columns)

            test_x = self.test.reset_index(drop=True)
            test_x_t = t_x.transform(test_x)
            test_x_pt = scaler_x.transform(test_x_t)
            test_x = pd.DataFrame(data=test_x_pt,
                    columns=test_x.columns)

        return train_x, test_x