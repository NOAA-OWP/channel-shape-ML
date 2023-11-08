# Libraries 
# Catch warnings 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from typing import Tuple
import data_loader as dataloader

import pickle
import json
import pandas as pd
import numpy as np
import os
import sys 
from tqdm import tqdm
from joblib import Parallel, delayed


# WD
# ---------------------------initialize --------------------------- #
class DPModel:
    """ Main body of the machine learning model for estimating bankfull width 
        and depth

        Parameters
        ----------
        rand_state : int
            A custom integer for randomness
    """
    def __init__(self, rand_state) -> None:
        # os.chdir(b'/home/arash.rad/river_3d/conus-fhg/')
        pd.options.display.max_columns = 30
        self.rand_state                = rand_state

# --------------------------- Load trained models  --------------------------- #    
    def loadModel(self, out_feature: str, vote_flag: bool = False, meta_flag: bool = False,
                    best_flag: bool = True, file: str = 'bf', model_type: str = 'xgb',
                    ) -> Tuple[any, list]:
        """ Load the trained transformed models

        Parameters
        ----------
        out_feature : str
            Name of the FHG coefficients
        vote_flag : bool
            Whether to use vote model 
            Options are:
            - True
            - False
        meta_flag : bool
            Whether to use meta model
            Options are:
            - True
            - False
        best_flag : bool
            Whether to use best model
            Options are:
            - True
            - False
        file : str
            Prefix of trained models
            Options are:
            - any string
        model_type: str
            The best model choices
            Options are:
            - xgb
            - lgb
        
        Example
        --------
        >>> DPModel.loadData(out_feature = 'Y-bf', vote_flag = False, meta_flag = False,
                    best_flag = True, file = 'bf', model_type = 'xgb')
        """
        
        # Load ML models
        if meta_flag:
            model = pickle.load(open('models/'+file+'_'+out_feature+'_final_Meta_Model.pickle.dat', "rb"))
            best_model = pickle.load(open('models/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        elif vote_flag:
            model = pickle.load(open('models/'+file+'_'+out_feature+'_final_Voting_Model.pickle.dat', "rb"))
            best_model = pickle.load(open('models/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        elif best_flag:
            model = best_model = pickle.load(open('models/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        
        # Extract feature names
        if model_type == 'xgb':
            feats = best_model.get_booster().feature_names
        elif model_type == 'lgb':
            feats = best_model.feature_name_
        else:
            feats = best_model.feature_name()
        return model, feats
    
    def process_target(self, dl_obj, target_name: str, vote_flag: bool=False, meta_flag: bool=False, 
                        best_flag: bool=True, file: str='bf', model_type: str='xgb') -> Tuple[str, np.array]:
        
        dl_obj.addExtraFeatures(target_name)
        dl_obj.buildPCA(target_name)
        
        if target_name.startswith("Y"):
            model_type = 'xgb'
        else:
            model_type = 'lgb'

        model, feats = self.loadModel(target_name, vote_flag=False, meta_flag=False, best_flag=True, file='bf', model_type=model_type)

        data_in = dl_obj.data[feats]
        y_pred_label = 'ML_' + target_name + '(m)'
        preds_all = model.predict(data_in)
        
        return y_pred_label, preds_all


# --------------------------- A driver class --------------------------- #           
class RunMlModel:
    @staticmethod
    def main(argv):
        """ The driver class to run ML models
        
        Parameters 
        ----------
        argv: list
            taken from bash script
        """
        nthreads     = int(argv[0])
        SI           = True
        rand_state   = 105

        # Load data
        dl_obj = dataloader.DataLoader(rand_state)
        dl_obj.readFiles()
        dl_obj.imputeData()

        # Load targets
        temp        = json.load(open('data/model_feature_names.json'))
        target_list = temp.get('out_features')
        out_vars    = []
        del temp

        # Additional parameters
        vote_flag   = False
        meta_flag   = False
        best_flag   = True
        file        = 'bf'
        model_type  = 'xgb'
        log_y_t     = True
        
        deploy_obj = DPModel(rand_state)

        # Parallelize the for loop
        results = Parallel(n_jobs=nthreads, backend="multiprocessing")(delayed(deploy_obj.process_target)(dl_obj, target_name, vote_flag, meta_flag, best_flag, file, model_type) for target_name in tqdm(target_list))
        
        # Unpack the results
        for y_pred_label, preds_all in results:
            out_vars.append(y_pred_label)
            if log_y_t:
                dl_obj.data[y_pred_label] = np.exp(preds_all)
            else:
                dl_obj.data[y_pred_label] = preds_all
            if SI:
                dl_obj.data[y_pred_label] = dl_obj.data[y_pred_label] * 0.3048
    
        out_vars.append('comid')
        out_df = dl_obj.data[out_vars]
        out_df.to_parquet('data/ml_exports.parquet')
        print("\n ------------- ML estimates complete ----------- \n")
        return

if __name__ == "__main__":
    RunMlModel.main(sys.argv[1:])