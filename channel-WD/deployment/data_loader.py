# Libraries
import pandas as pd
import numpy as np
import pickle
import os
import json
import re
import fnmatch

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
    def __init__(self, rand_state: int, data_path: str = 'data/input.parquet') -> None:
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
        except:
            print('Wrong address or data format. Please use parquet file.')
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
            self.data = self.data.replace(-1, np.nan)
            column_medians = self.data.median()
            self.data = self.data.fillna(column_medians)
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
            for i in range(0, 5, 1):
                self.data[str(text[1:-1])+"_"+str(i)] = new_data_pca[:, i]

        return 
        