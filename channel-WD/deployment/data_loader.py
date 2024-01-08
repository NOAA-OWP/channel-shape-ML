# Libraries
import pandas as pd
import numpy as np
import pickle
import os
import json
import re
import fnmatch
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer


# Custom dataset
# --------------------------- Read data files --------------------------- #
class DataLoader:
    """ Main body of the data loader for preparing data for ML models

    Parameters
    ----------
    data_path : str
        The path to data that is used in ML model
    rand_state : int
        A random state number
    Example
    --------
    >>> DataLoader(rand_state = 105, data_path = 'data/input.parquet')
        
    """
    def __init__(self, rand_state: int, data_path: str = 'data/nwm_conus_input.parquet') -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        self.data                       = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        # ___________________________________________________
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),"models/")):
            os.mkdir(os.path.join(os.getcwd(),"models/"))
        if not os.path.isdir(os.path.join(os.getcwd(),"data/")):
            os.mkdir(os.path.join(os.getcwd(),"data/"))
        if not os.path.isdir(os.path.join(os.getcwd(),'model_space/')):
            os.mkdir(os.path.join(os.getcwd(),'model_space/'))

    def readFiles(self) -> None:
        """ Read files from the directories
        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')
            self.data = self.data.iloc[2500000:2647455]
            self.data.reset_index(drop=True, inplace=True)
        except:
            print('Wrong address or data format. Please use correct parquet file.')
        
        self.data = self.data[set(self.data.columns.to_list()) - set(['geometry','NHDFlowline','full_cats',
                                                                      'gridcode','number_unique_peaks','non_zero_years',
                                                                      'toCOMID','Hydroseq','RPUID','FromNode',
                                                                      'ToNode','VPUID','hy_cats','geometry_poly',
                                                                      'REACHCODE','sourcefc','comid'])]
        # Find string columns (debug)
        # string_columns = []
        # # Iterate through each column and check if it contains string values
        # for col in self.data.columns:
        #     if self.data[col].dtype == 'O':  # 'O' represents object type (strings) in Pandas
        #         string_columns.append(col)  
        # print(string_columns)
        return

    # --------------------------- Add Binary Features --------------------------- #
    def addExtraFeatures(self, target_name: str) -> None:
        # Add VAA dummy
        self.data['vaa_dummy'] = self.data['roughness'].isnull().values
        self.data['vaa_dummy'] = self.data['vaa_dummy'] * 1

        # Add Scat dummy
        self.data['scat_dummy'] = self.data['BFICat'].isnull().values
        self.data['scat_dummy'] = self.data['scat_dummy'] * 1

        # Add discharge dummy
        if target_name.endswith("bf"):
            self.data['bf_ff'] = np.nan
            self.data['NWM'] = self.data['rp_2']
            self.data['discharge_dummy'] = 3
        else:
            self.data['in_ff'] = np.nan
            self.data['NWM'] = self.data['rp_1.5']
            self.data['discharge_dummy'] = 3
        return

    # --------------------------- Imputation --------------------------- #
    def imputeData(self) -> None:
        # Data imputation 
        impute = "median"
        if impute == "zero":
            self.data = self.data.fillna(-1) # a temporary brute force way to deal with NAN
        if impute == "median":
            median_values_df = pd.read_parquet('models/median_imput.parquet')
            for col in self.data.columns:
                if col in median_values_df.index:
                    median_value = median_values_df.loc[col, 'Median']
                    self.data[col].fillna(median_value, inplace=True)
        return

    # --------------------------- Dimention Reduction --------------------------- #     
    # PCA model
    def buildPCA(self, variable)  -> None:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        matching_files = []
        folder_path = 'models'
        full_path = os.path.join(os.getcwd(), folder_path)
        # Iterate through the files in the folder
        for root, dirs, files in os.walk(full_path):
            for filename in files:
                # Check if both "PCA" and "Y_bf" are present in the file name
                search_pattern = f'*PCA*{variable}*'
                if all(fnmatch.fnmatch(filename, f'*{part}*') for part in search_pattern.split('*')):
                    matching_files.append(os.path.join(root, filename))

        # Extract the text between "PCA" and the 'vars' value using regular expressions
        pattern = f'{re.escape(variable)}(.*?)PCA'
        captured_texts = []
        for filename in matching_files:
            match = re.search(pattern, filename)
            if match:
                captured_texts.append(match.group(1))
            else:
                captured_texts.append("No match found")

        temp = json.load(open('model_space/dimension_space.json'))
        
        # Print the list of matching files
        for pca_item, text in zip(matching_files, captured_texts):
            pca =  pickle.load(open(pca_item, "rb"))
            temp_data = self.data[temp.get(text[1:-1])]
            new_data_pca = pca.transform(temp_data)
            max_n = min(5, len(temp.get(text[1:-1])))
            for i in range(0, max_n, 1):
                self.data[str(text[1:-1])+"_"+str(i)] = new_data_pca[:, i]

        return self.data
    
    # --------------------------- Data transformation --------------------------- #     
    # Input and output transformation
    def transformXData(self, out_feature: str, trans_feats : list,
                        t_type: str = 'power', x_transform: bool = False)  -> pd.DataFrame:
        """ Apply scaling and normalization to data
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        print('transforming and plotting ...')
        self.data.reset_index(drop=True)
        trans_data = self.data[trans_feats]

        # Always perform transformation on PCA
        in_features = self.data.columns.tolist()
        in_feats = set(in_features) - set(trans_feats)

        if t_type!='log':
            trans =  pickle.load(open('models/train_x_'+out_feature+'_tansformation.pkl', "rb"))
            min_max_scaler = pickle.load(open('models/train_x_'+out_feature+'_scaler_tansformation.pkl', "rb"))
            scaler_data = min_max_scaler.transform(trans_data)
            data_transformed = trans.transform(scaler_data)
            data_transformed = pd.DataFrame(data_transformed, columns=trans_data.columns)
            if not x_transform:
                self.data = pd.concat([data_transformed, self.data[in_feats]], axis=1)
            else:
                self.data = data_transformed.copy()
        else:
            # Replace NA and inf
            self.data = np.log(np.abs(self.data)).fillna(0)
            self.data.replace([np.inf, -np.inf], -100, inplace=True)

        # Tests
        is_inf_data = self.data.isin([np.inf, -np.inf]).any().any()
        if is_inf_data:
            print('---- found inf in X {0} !!!!'.format(out_feature))
        has_missing_data  = self.data.isna().any().any()
        if has_missing_data:
            print('---- found nan in X {0} !!!!'.format(out_feature))

        return 
    
    def transformYData(self, out_feature, data, t_type: str = 'power', y_transform: bool = False)  -> np.array:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        print('transforming and plotting ...')

        if y_transform:
            if t_type!='log':
                def applyScalerY(arr):
                    min_max_scaler = pickle.load(open('models/train_y_'+out_feature+'_scaler_tansformation.pkl', "rb"))
                    # min_max_scaler = pickle.load(open('/mnt/d/Lynker/R2_out/New/'+folder+'/conus-fhg/'+file+'/model/'+'train_'+arr.name+'_scaled.pkl', "rb"))
                    data_minmax = min_max_scaler.transform(arr.values.reshape(-1, 1))
                    return data_minmax.flatten()
                
                trans =  pickle.load(open('models/train_y_'+out_feature+'_tansformation.pkl', "rb"))
                #scaler_data = data.apply(applyScalerY)
                data = pd.DataFrame(data, columns=[out_feature])
                data = trans.inverse_transform(data) # .reshape(-1,1)

            else:
                # Replace NA and inf
                data = np.log(np.abs(data)).fillna(0)
                data.replace([np.inf, -np.inf], -100, inplace=True)

        print('--------------- End of Y transformation ---------------')
        
        # Tests
        is_inf_y = np.isinf(data).any()
        if is_inf_y:
            print('---- found inf in Y {0} !!!!'.format(out_feature))
        
        has_missing_y = np.isnan(data).any()
        if has_missing_y:
            print('---- found nan in Y {0} !!!!'.format(out_feature))

        return data
        