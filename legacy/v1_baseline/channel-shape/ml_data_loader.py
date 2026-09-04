# Libraries
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer
import scipy
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import Tuple

# FHG dataset
# --------------------------- Read data files --------------------------- #
class DataLoader:
    """ Main body of the data loader for preparing data for ML models

    Parameters
    ----------
    data_path : str
        The path to data that is used in ML model
    rand_state : int
        A random state number
    out_feature : str
        The name of the FHG coeficent to be used
    custom_name : str
        A custom name defiend by user to name modeling task
    x_transform : str
        Whether to apply transformation to predictor variables or not 
        Opptions are:
        - True
        - False
    x_transform : str
        Whether to apply transformation to predictor variables or not 
        Opptions are:
        - True
        - False
        - defaults to False
    y_transform : bool
        Whether to apply transformation to target variable or not 
        Opptions are:
        - True
        - False
        - defaults to False
    R2_thresh : float
        The desired coeficent of determation to filter out bad measurments
        Opptions are:
        - any value between 0.0 - 100.0
        - defaults to 0.0
    Example
    --------
    >>> DataLoader(data_path = 'data/test.parquet', out_feature = 'b', rand_state = 115,
        custom_name = 'test', x_transform = False, y_transform = False, R2_thresh = 0.0)
        
    """
    def __init__(self, data_path: str, rand_state: int, out_feature: str, 
                 custom_name: str, x_transform: bool = False, 
                 y_transform: bool = False, R2_thresh: float = 0.0) -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        self.data                       = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        self.in_features                = []
        self.out_feature                = out_feature
        self.custom_name                = custom_name
        self.x_transform                = x_transform
        self.y_transform                = y_transform
        self.train                      = pd.DataFrame([])
        self.test                       = pd.DataFrame([])
        self.R2_thresh                  = R2_thresh

        # ___________________________________________________
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,"model/")):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,"model/"))

    def readFiles(self) -> None:
        """ Read files from the directories
        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')
            self.data.astype({'siteID': 'string'})
        except:
            print('Wrong address or data format. Please use parquet file.')   
        
        # ___________________________________________________
        # Filter bad stations
        adcp = pd.read_parquet('data/Processed_adcp.parquet', engine='pyarrow')
        adcp.astype({'siteID': 'string'})
        adcp = adcp.loc[(adcp['Q']>0)&(adcp['V_x']>0)&(adcp['Ymean']>0)&(adcp['TW_x']>0)]
        adcp['Y_prime'] = adcp['c']*(adcp['Q']**adcp['f'])
        adcp['TW_prime'] = adcp['a']*(adcp['Q']**adcp['b'])
        adcp['V_prime'] = adcp['k']*(adcp['Q']**adcp['m'])
        adcp.loc[adcp['Y_prime'] > 10e+06, 'Y_prime'] = 10e+06
        adcp.loc[adcp['TW_prime'] > 10e+06, 'TW_prime'] = 10e+06
        adcp.loc[adcp['V_prime'] > 10e+06, 'V_prime'] = 10e+06

        def compRsquared(df: pd.DataFrame) -> pd.DataFrame:
            """ 
            R2 based on linear regression 
            Parameters:
            ----------
            df pd.DataFrame
                it contains the following
                y_true ([pd.series]): Observations 
                y_pred ([pd.series]): Predictions
            Returns:
            -------
                pd.DataFrame
                coeficnet of determination for depth, width, and velocity
            """
            def calR2(y_true, y_pred):
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_pred)
                return r_value**2

            df['Y_R2'] = calR2(df['Ymean'], df['Y_prime'])
            df['TW_R2'] = calR2(df['TW_x'], df['TW_prime'])
            df['V_R2'] = calR2(df['V_x'], df['V_prime'])
            return df
        
        R2_df2 = adcp.groupby('siteID').apply(compRsquared).reset_index()
        del adcp
        r2_epochs = np.arange(0, 1.05, 0.05)
        grouped_r2 = R2_df2.groupby('siteID').agg('mean')
        count_listY = [3543]
        count_listTW = [3543]
        count_listV = [3543]
        count_listT = [3543]
        for epoch in r2_epochs:
            count_Y = len(grouped_r2.loc[grouped_r2['Y_R2']>=epoch])
            count_TW = len(grouped_r2.loc[grouped_r2['TW_R2']>=epoch])
            count_V= len(grouped_r2.loc[grouped_r2['V_R2']>=epoch])
            count_T = len(grouped_r2.loc[(grouped_r2['TW_R2'] >= epoch) &
                                    (grouped_r2['Y_R2'] >= epoch) &
                                    (grouped_r2['V_R2'] >= epoch)])
            count_listY.append(count_Y)
            count_listTW.append(count_TW)
            count_listV.append(count_V)
            count_listT.append(count_T)
        r2_epochs = np.insert(r2_epochs, 0, -0.05, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        scale = 30
        ax.grid(True)
        ax.scatter(np.array(count_listY)/3543, r2_epochs, c='r', s=scale, label='Y',
                    alpha=0.6, edgecolors='k')
        ax.scatter(np.array(count_listTW)/3543, r2_epochs, c='b', s=scale, label='TW',
                    alpha=0.6, edgecolors='k')
        ax.scatter(np.array(count_listV)/3543, r2_epochs, c='g', s=scale, label='V',
                    alpha=0.6, edgecolors='k')
        ax.scatter(np.array(count_listT)/3543, r2_epochs, c='k', s=scale, label='Total',
                    alpha=0.6, edgecolors='k')
        plt.vlines(x=self.R2_thresh, ymin=0, ymax=1, colors='purple', ls='--', lw=2, label='Threshold')
        ax.legend()
        ax.set_ylim([0, 1])
        plt.xlabel("R2")
        plt.ylabel("% stations greater than or equal")
        my_plot = plt.gcf()
        plt.savefig(self.custom_name+'/img/model/'+str(self.custom_name)+'_R2_cut.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()

        good_stations = grouped_r2.loc[(grouped_r2['TW_R2'] >= self.R2_thresh) &
                               (grouped_r2['Y_R2'] >= self.R2_thresh)]
                            #    (grouped_r2['V_R2'] >= self.R2_thresh)]
        good_stations = good_stations.reset_index()
        good_stations.astype({'siteID': 'string'})
        stations = good_stations['siteID'].tolist()
        del good_stations
        self.data = self.data[self.data['siteID'].isin(stations)].reset_index(drop=True)
        self.data['r'] = abs(self.data['r'])
        
        return 
        
 # --------------------------- Split train and test --------------------------- #
    
    def splitData(self, sample_type: str) -> None:
        """ 
        To split data to train and test, and whether to use all 
        features or few 

        Parameters:
        ----------
        sample_type: str
            For limiting feature space
            Options are:
            - ``All``
            - ``Sub``
            - ``test`
        Example
            --------
            >>> splitData("All")
        """
        if sample_type == "All":
            temp = json.load(open('data/ml_model_feature_names.json'))
            model_features = temp.get('out_features')+temp.get('in_features')+temp.get('id_features')+temp.get('in_features_NWM')+temp.get('in_features_flow_freq')
            # ___________________________________________________
            # to dump variables
            # dump_list = ["BFICat","CatAreaSqKm","ElevCat","PctWaterCat","PrecipCat",
            # "RckDepCat","RockNCat","RunoffCat","WaterInputCat","WetIndexCat","WtDepCat",
            # "scat_nlcd_feature1","scat_nlcd_feature2","scat_nlcd_feature3",
            # "scat_ant_feature1","scat_ant_feature2","scat_ant_feature3",
            # "scat_lith_feature1","scat_lith_feature2","scat_lith_feature3",
            # "scat_hydra_feature1","scat_hydra_feature2","SM_ave","SM_max","SM_min","Q_mean","Qb_mean",
            # "Q_max","Qb_max","Q_min","Qb_min","ST_ave","ST_max","ST_min","ET_ave","AI","LAI_max","LAI_min",
            # "LAI_ave","Precip_ave","Precip_max","Precip_min","NDVI_max","NDVI_min","NDVI_ave","aspect_ave",
            # "slope_ave","elevation_ave"]
            # model_features = list(set(model_features) - set(dump_list))
            self.in_features = model_features.copy()
            self.in_features = list(set(self.in_features) - set(temp.get('id_features')) - set(temp.get('out_features')))
        else:
            temp = json.load(open('model_space/feature_space.json'))
            temp = temp.get(sample_type).get(self.out_feature+'_feats')
            model_features = temp
            self.in_features = model_features.copy()
            temp = json.load(open('data/ml_model_feature_names.json'))
            model_features += temp.get('out_features')+temp.get('id_features')

        # to moderate HydroSwat discharge values
        model_features.append('nwis_25')
        # added coordinates for spatial corralations 
        model_features.append('lat')
        model_features.append('lng')
        self.in_features.append('lat')
        self.in_features.append('lng')
        del temp

        # Convert full negative columns to positive
        def convPositive(arr):
            if (arr < 0).values.sum() == len(arr):
                arr = np.abs(arr)
            return arr
        self.data[self.in_features] = self.data[self.in_features].apply(convPositive)
        df_mask = self.data[model_features]

        df_mask = df_mask.fillna(0) # // to be changed (compensating for EE features in cities that can be set to 0)
        msk = np.random.rand(len(df_mask)) < 0.85
        self.train = df_mask[msk]
        self.test = df_mask[~msk]
        return

# --------------------------- Transformation --------------------------- #

    def transformData(self, type: str = 'quant') -> Tuple[pd.DataFrame,
                                                          np.array,
                                                          pd.DataFrame,
                                                          pd.DataFrame,
                                                          np.array,
                                                          pd.DataFrame]:
        """ 
        To split data to train and test, and whether to use all 
        features or few 

        Parameters:
        ----------
        type: str
            Type of transformation
            Options are:
            - ``power`` for power transformation
            - ``any``   for quantile transformation
        
        Returns:
        ----------
        train_x: pd.DataFrame
            A dataframe containg predictor data for training 
        train_y: np.array
            An array containg target data for training 
        train_id: pd.DataFrame
            A dataframe containg site id and nwis_25 of the stations for training 
        test_x: pd.DataFrame
            A dataframe containg predictor data for testing 
        test_y: np.array
            An array containg target data for testing 
        test_id: pd.DataFrame
            A dataframe containg site id and nwis_25 of the stations for testing 

        Example
            --------
            >>> train_x, train_y, train_id, test_x, test_y, test_id = transformData("power")
        """
        # Scaler function
        def applyLearnScaler(arr):
            """
            Applies minmax scaler to data with min set to 0.001 and max is the maximum of data
            
            Parameters:
            ------------
            arr: pd.Series()
                represnting the target column

            Returns:
            ------------
            np.array()
                The scaled data 
            """
            # Some like min NDVI is negative and should be convert to abs 
            # try:
            min_max_scaler = MinMaxScaler(feature_range = (100, 500))
            data_minmax = min_max_scaler.fit_transform(arr.values.reshape(-1, 1))
            # except:
            #     min_max_scaler = MinMaxScaler(feature_range = (0.00001, np.max(np.abs(arr))))
            #     data_minmax = min_max_scaler.fit_transform(np.abs(arr).values.reshape(-1, 1))
            # Save scaling
            pickle.dump(min_max_scaler, open(self.custom_name+'/model/'+'train_'+arr.name+'_scaled.pkl', "wb"))
            return data_minmax.flatten()
        
        def applyScaler(arr):
            """
            Applies learned minmax scaler to data with min set to 0.001 and max is the maximum of data
            
            Parameters:
            ------------
            arr: pd.Series()
                represnting the target column

            Returns:
            ------------
            np.array()
                The scaled data 
            """
            min_max_scaler = pickle.load(open(self.custom_name+'/model/'+'train_'+arr.name+'_scaled.pkl', "rb"))
            data_minmax = min_max_scaler.transform(arr.values.reshape(-1, 1))
            return data_minmax.flatten()
        
        if self.x_transform:
            if type=='power':
                # t_x = MinMaxScaler(feature_range=(0, 1))
                t_x = PowerTransformer(method='box-cox', standardize=False)
            else:
                t_x = QuantileTransformer(
                    n_quantiles=5000, output_distribution="normal", 
                    random_state=self.rand_state
                )

            train_x = self.train[self.in_features].reset_index(drop=True)
            scaler_x = train_x.apply(applyLearnScaler)
            train_x_t = t_x.fit_transform(scaler_x)
            pickle.dump(t_x, open(self.custom_name+'/model/'+'train_x_'+self.out_feature+'_tansformation.pkl', "wb"))
            # train_x_pt = scaler_x.fit_transform(train_x_pt)
            train_x = pd.DataFrame(data=train_x_t,
                    columns=train_x.columns)
            train_id =  self.train[['siteID', 'nwis_25']].reset_index(drop=True)

            test_x = self.test[self.in_features].reset_index(drop=True)
            
            scaler_x = test_x.apply(applyScaler)
            test_x_t = t_x.transform(scaler_x)
            # test_x_pt = scaler_x.transform(test_x_pt)
            test_x = pd.DataFrame(data=test_x_t,
                    columns=test_x.columns)
            test_id =  self.test[['siteID', 'nwis_25']].reset_index(drop=True)

        else:
            train_x = self.train[self.in_features].reset_index(drop=True)
            train_id =  self.train[['siteID', 'nwis_25']].reset_index(drop=True)
            test_x = self.test[self.in_features].reset_index(drop=True)
            test_id =  self.test[['siteID', 'nwis_25']].reset_index(drop=True)

        if self.y_transform:
            if type=='power':
                # t_y = MinMaxScaler(feature_range=(0, 1))
                t_y = PowerTransformer(method='box-cox', standardize=False)
            else:    
                t_y = QuantileTransformer(
                    n_quantiles=5000, output_distribution="normal", 
                    random_state=self.rand_state
                )
            scaler_y = self.train[[self.out_feature]].apply(applyLearnScaler)
            train_y = scaler_y.reset_index(drop=True)
            train_y_t = t_y.fit_transform(train_y)
            pickle.dump(t_y, open(self.custom_name+'/model/'+'train_y_'+self.out_feature+'_tansformation.pkl', "wb"))
            # train_y_pt = scaler_x.fit_transform(train_y_pt)
            train_y = train_y_t.ravel()

            scaler_y = self.test[[self.out_feature]].apply(applyScaler)
            test_y = scaler_y.reset_index(drop=True)
            test_y_t = t_y.transform(test_y)
            # test_y_pt = scaler_y.transform(test_y_pt)
            test_y = test_y_t.ravel()
        else:
            train_y = self.train[[self.out_feature]].reset_index(drop=True)
            train_y = train_y.values.ravel()
            test_y = self.test[[self.out_feature]].reset_index(drop=True)
            test_y = test_y.to_numpy().reshape((-1,))
        
        return train_x, train_y, train_id, test_x, test_y, test_id
