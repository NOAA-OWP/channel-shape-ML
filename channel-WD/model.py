# Libraries 
# Catch warnings 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from typing import Tuple
import data_loader as dataloader
import feature_importance as fimp
import save_output as sd
# import comp_plot as pcml
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, ARDRegression, OrthogonalMatchingPursuit
from sklearn.metrics import mean_absolute_error, max_error, explained_variance_score, mean_squared_error, mean_absolute_percentage_error
import lightgbm as ltb
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import load_model, save_model, Model
# from keras.models import load_model

import tempfile
import pickle
import joblib
import json
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import os
import sys 
from tqdm import tqdm


# FHG
# ---------------------------initialize --------------------------- #
class MlModel:
    """ Main body of the machine learning model for estimating FHG
        Parameters

        Parameters
        ----------
        custom_name : str
            A custom name for the model to be extucuted 
    """
    def __init__(self, custom_name: str) -> None:
        # os.chdir(b'/home/arash.rad/river_3d/conus-fhg/')
        pd.options.display.max_columns = 30
        self.custom_name               = custom_name
        self.rand_state                = 105
        self.grid_searches             = {}
        temp                           = json.load(open('data/model_feature_names.json'))
        self.target_data_path          = ""
        self.train_x                   = 0 
        self.train_y                   = 0 
        self.train_id                  = 0 
        self.test_x                    = 0
        self.test_y                    = 0
        self.test_id                   = 0
       
        self.train_sub_id              = 0 
        self.x_train                   = 0 
        self.eval_id                   = 0 
        self.x_eval                    = 0

        # ___________________________________________________
        # Free memory
        del temp

        # ___________________________________________________
        # Check directories if not present create one
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name)):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,'model/')):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,'model/'))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,'metrics/')):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,'metrics/'))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,'img/')):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,'img/'))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,'img/model/')):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,'img/model/'))
        if not os.path.isdir(os.path.join(os.getcwd(),'cache/')):
            os.mkdir(os.path.join(os.getcwd(),'cache/'))
        
# --------------------------- Load train and test data files --------------------------- #    
    def loadData(self, out_feature: str, x_transform: bool = False, 
                 y_transform: bool = False, R2_thresh: float = 0.0, count_thresh: int = 3,
                 sample_type: str = "All", pci: bool = True, t_type: str = 'log') -> None:
        """ Load the data and apply data filtering, transformation and 
        feature selection if nessassery

        Parameters
        ----------
        out_feature : str
            Name of the FHG coeficients
        x_transform : bool
            Whether to apply transformation to predictor variables or not 
            Opptions are:
            - True
            - False
        y_transform : bool
            Whether to apply transformation to target variable or not 
            Opptions are:
            - True
            - False
        R2_thresh : float
            The desired coeficent of determation to filter out bad measurments
            Opptions are:
            - any value between 0.0 - 100.0
        count_thresh: int
            The desired number of observations in each station to filter out bad measurments
        sample_type : str
            The type of predictor feature selection
            Opptions are:
            - "All": for considering all features
            - "Sub": for considering pre selected features
            - "test": a test case for unit testing
        pci: bool
            Whether to apply PCA or not 
            Opptions are:
            - True
            - False
        t_type: str
            type of transformation
            Opptions are:
            - log
            - power
            - quant
        
        Example
        --------
        >>> MlModel.loadData(out_feature = 'b', x_transform = False, 
                 y_transform = False, R2_thresh = 0.0,
                 sample_type = "Sub", pci = False, t_type = 'log')
        """
        # Bulid an instance of DataLoader object

        if "TW_" in out_feature:
            data_path = 'data/width_predictor_test.parquet'
            self.target_data_path = 'data/width_target.parquet'
        else:
            data_path = 'data/depth_mean_predictor_test.parquet'
            self.target_data_path = 'data/depth_mean_target.parquet'

        data_loader = dataloader.DataLoader(data_path=data_path,
                                            target_data_path=self.target_data_path,
                                            rand_state=self.rand_state, 
                                            # in_features=self.in_features, 
                                            out_feature=out_feature, 
                                            custom_name=self.custom_name, 
                                            x_transform=x_transform, y_transform=y_transform,
                                            R2_thresh=R2_thresh, count_thresh=count_thresh) 
        data_loader.readFiles()
        if pci:
            data_loader.reduceDim()
        data_loader.splitData(sample_type=sample_type)
        self.train_x, self.train_y, self.train_id, self.test_x, self.test_y, self.test_id = data_loader.transformData(t_type=t_type, plot_dist=False)

# --------------------------- Grid Search --------------------------- #
    def findBestParams(self, out_features: str = 'TW_bf', nthreads: int = -1, space: str = 'actual_space',
                        weighted: bool = False) -> Tuple[str, dict, pd.DataFrame]:
        """ Find the best parameters of the all ML models through k-fold
        cross validation and prevent overfit

        Parameters
        ----------
        out_features : str
            Name of the FHG coeficients
        nthreads : int
            Number of cores to be engaged in k-fold
        space : str
            For unit testing: Whether to use complete paramter space or a 
            small subset of it
            Opptions are:
            - "actual_space"
            - "test_space"
        weighted : bool
            Whether to apply weighted learning or not
            Opptions are:
            - True
            - False

        Outputs
        ----------
        best_model : str
            The name of the best model identified during k-fold
        best_params : dict
            A dictionary containg all parametrs of the best model after training
        best_models : pandas.DataFrame
            A dataframe containg names and all parametrs of the top models after training
        

        Example
        --------
        >>> MlModel.findBestParams(out_features = 'b', nthreads = -1, space: 'actual_space',
                        weighted = False)
        """
        # Check if it is a weighted learning
        if weighted:
            t_y = pickle.load(open(self.custom_name+'/model/'+'train_y_'+out_features+'_tansformation.pkl', "rb"))
            temp = t_y.inverse_transform(self.train_y.reshape(-1,1)).ravel()
            y_weights = abs(temp - temp.mean())+0.1
        else: 
            y_weights = None
        fit_params = dict(sample_weight=y_weights)#, base_margin=np.abs(self.train_x[:, 1]))
        
        # ___________________________________________________
        # Build an isntance of each model with defaults
        xgb_reg = xgb.XGBRegressor(seed=self.rand_state, nthread=nthreads)
        
        rf_reg = RandomForestRegressor(random_state=self.rand_state, n_jobs=nthreads)
       
        hgb_reg = HistGradientBoostingRegressor(random_state=self.rand_state)
       
       
        lgb_reg = ltb.LGBMRegressor(random_state=self.rand_state, n_jobs=nthreads)
 
        bsvr_reg = BaggingRegressor(estimator=SVR(),
                                    n_jobs=nthreads, random_state=self.rand_state)
    
        knr_reg = KNeighborsRegressor(n_jobs=nthreads)
        ard_reg = ARDRegression()
        enet_reg = ElasticNet(random_state=self.rand_state)
        mlp_reg = MLPRegressor(random_state=self.rand_state)
        bays_reg = BayesianRidge()
        # orth_reg = OrthogonalMatchingPursuit()
       
       # ___________________________________________________
       # Define models and paramters
        params_space = json.load(open('model_space/params_space.json')) 
        models = { 
            'xgb': xgb_reg,
            'rf': rf_reg,
            'hgb': hgb_reg,
            'lgb': lgb_reg,
            'bsvr': bsvr_reg,
            'knr': knr_reg,
            'ard': ard_reg,
            'enet': enet_reg,
            'mlp': mlp_reg,
            'bays': bays_reg
            # 'orth': orth_reg
        }
        params = { 
            'xgb': params_space.get(space).get('xgb_params'),
            'rf': params_space.get(space).get('rf_params'),
            'hgb': params_space.get(space).get('hgb_params'),
            'lgb': params_space.get(space).get('lgb_params'),
            'bsvr': params_space.get(space).get('bsvr_params'),
            'knr': params_space.get(space).get('knr_params'),
            'ard': params_space.get(space).get('ard_params'),
            'enet': params_space.get(space).get('enet_params'),
            'mlp': params_space.get(space).get('mlp_params'),
            'bays': params_space.get(space).get('bays_params')
            # 'orth': params_space.get(space).get('orth_params')
        }

        # ___________________________________________________
        # Do a k-fold cross validation on models
        cv = RepeatedKFold(n_splits = 5, n_repeats = 3, random_state = self.rand_state)
        for model_key in models.keys():
            print('Running GridSearchCV for model: %s.' % model_key)
            model = models[model_key]
            param = params[model_key]
            def deval_f(x):
                try:
                    ans = eval(str(x))
                except:
                    ans = x
                return ans
            for k,v in param.items():
                temp = []
                for x in v:
                    temp.append(deval_f(x))
                param[k] = temp
                del temp
            grid_search = GridSearchCV(estimator=model, param_grid=param, n_jobs = nthreads, cv = cv,
                                       scoring="neg_root_mean_squared_error",)
            if model_key == 'ard' or model_key == 'knr' or model_key == 'mlp':
                grid_search.fit(self.train_x, self.train_y)
            else:
                grid_search.fit(self.train_x, self.train_y, **fit_params)
            joblib.dump(grid_search, self.custom_name+'/model/'+str(self.custom_name)+'_'+out_features+'_'+str(model_key)+'_gridsearch.pkl')
            self.grid_searches[model_key] = grid_search
        print('GridSearchCV complete.')
        
        # ___________________________________________________
        # Find best model and parmaters to pass on
        frames = []
        sort_by = 'mean_test_score'
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]

        # ___________________________________________________
        # retrun the best
        best_model = df.iloc[0].get('estimator')
        best_params = df.iloc[0].get('params')
        print("best model {0}".format(best_model))
        print("best params {0}".format(best_params))

        # ___________________________________________________
        # get best of all models
        best_models = df.loc[df.groupby("estimator")[sort_by].idxmax()]
        best_models = best_models[['estimator', sort_by, 'params']]
        return best_model, best_params, best_models

# --------------------------- Run Best Model --------------------------- #
    def runMlModel(self, best_model: str, best_params: dict, best_models: pd.DataFrame, 
                   weighted: bool, out_features: str, nthreads: int = -1) -> Tuple[any, 
                                                                                   VotingRegressor,
                                                                                   StackingRegressor,
                                                                                   pd.DataFrame,
                                                                                   np.array,
                                                                                   pd.DataFrame,
                                                                                   np.array]:
        """ Train the ML models based on k-fold results
        Parameters
        ----------
        best_model : str
            Name of the best ML model
        best_params : dict
            A dictionary containing estiamted parameters
        best_models : pd.DataFrame
            A dataframe containg top models and their parameter space
        weighted : bool
            Whether to apply weighted learning or not
            Opptions are:
            - True
            - False
        out_features : str
            Name of the FHG coeficients
        nthreads : int
            Number of cores to be engaged in training

        Outputs
        ----------
        loaded_model : any
            model structure and weights
        voting_model : VotingRegressor
            model structure and weights
        meta_model : StackingRegressor
            model structure and weights
        self.train_x : pd.DataFrame
            splited predictor data for training
        self.train_y : np.array
            splited target data for training
        self.test_x : pd.DataFrame
            splited predictor data for testing
        self.test_y : np.array
            splited target data for testing

        Example
        --------
        >>> MlModel.runMlModel(best_model, best_params, best_models, 
                   weighted = True, out_features = 'b', nthreads = -1)
        """
        def rsquared(obs: np.array, pred: np.array) -> float:
            """ Return R^2 where obs and pred are array 
            Parameters
            ----------
            obs : np.array
                A numpy array containing true values
            pred : np.array
                A numpy array containing estimated values

            Outputs
            ----------
            r_value**2 : flaot
                The value of coeficient of determination

            Example
            --------
            >>> r2 = rsquared(obs, pred)
            """
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obs, pred)
            return r_value**2
        # ___________________________________________________
        # Custom objective function
        # def custom_eval(y_pred, dtrain):
        #     y_true = dtrain.get_label()
        #     err = 1-f1_score(y_true, np.round(y_pred))
        #     return 'custom_eval', err
        
        # ___________________________________________________
        # Prepare models and split data
        model_obj = ModelSwitch(self.rand_state, nthreads)
        loaded_model = model_obj.modelName(best_model, best_params)
        concated_x = pd.concat([self.train_x, self.train_id], axis=1)
        # concated_y = pd.concat([self.train_y, self.train_id], axis=1)
        self.x_train, self.x_eval, self.y_train, self.y_eval = train_test_split(concated_x, self.train_y, test_size=0.15,
                                                                random_state=self.rand_state)
        self.train_sub_id = self.x_train[['siteID', 'R2']]
        self.train_sub_id = self.train_sub_id.reset_index(drop=True)
        self.x_train = self.x_train.loc[:, ~self.x_train.columns.isin(['siteID', 'R2'])]
        self.x_train = self.x_train.reset_index(drop=True)
        self.eval_id = self.x_eval[['siteID', 'R2']]
        self.eval_id = self.eval_id.reset_index(drop=True)

        self.x_eval = self.x_eval.loc[:, ~self.x_eval.columns.isin(['siteID', 'R2'])]
        self.x_eval = self.x_eval.reset_index(drop=True)
        # ___________________________________________________
        # Out of the box evaluation of models
        # Fit all models
        reg_models = lazypredict.Supervised.REGRESSORS
        lazypredict.Supervised.REGRESSORS = [t for t in reg_models if not t[0].startswith('Quantile')]
        ob_reg = LazyRegressor(predictions=True)
        models, predictions = ob_reg.fit(self.x_train, self.x_eval, self.y_train, self.y_eval)
        print('\n out of the box evaluation of models for target: '+str(self.custom_name)+ '\n')
        print(models)

        # ___________________________________________________
        # Check witch models are used with weights and fit
        eval_set = [(self.x_train, self.y_train), (self.x_eval, self.y_eval)]
        if weighted:
            t_y = pickle.load(open(self.custom_name+'/model/'+'train_y_'+out_features+'_tansformation.pkl', "rb"))
            temp = t_y.inverse_transform(self.y_train.reshape(-1,1)).ravel()
            y_weights = abs(temp - temp.mean())+0.1
        else: 
            y_weights = None
        fit_params = dict(sample_weight=y_weights)

        if best_model == 'xgb':
            loaded_model.fit(self.x_train, self.y_train, eval_metric=["mae", "rmse"], eval_set=eval_set, 
                             early_stopping_rounds=0.1*best_params['n_estimators'], verbose=False, **fit_params)
        elif best_model == 'ard' or best_model == 'knr' or best_model == 'mlp':
            loaded_model.fit(self.x_train, self.y_train)
        else:
            loaded_model.fit(self.x_train, self.y_train, **fit_params)
        
        # ___________________________________________________
        # Predict
        preds_t = loaded_model.predict(self.x_train)
        rs_DNN_t = round(rsquared(self.y_train, preds_t.flatten()), 2)
        print("best Training acc {0}".format(rs_DNN_t))
        
        # ___________________________________________________
        # save model to file
        pickle.dump(loaded_model, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_"+str(best_model)+"_Best_Model.pickle.dat", "wb"))
        # load model from file
        # loaded_model = pickle.load(open("model/"+out_features+"_gb_model.pickle.dat", "rb"))
        
        # ___________________________________________________
        # Meta learner & voting
        def loadBaseModel(model_df):
            temp_model_obj = ModelSwitch(self.rand_state, nthreads)
            return temp_model_obj.modelName(model_df["estimator"].values[0], model_df["params"].values[0])
        
        base_model = list()
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'xgb'])
        base_model.append(('xgb', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'rf'])
        base_model.append(('rf', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'hgb'])
        base_model.append(('hgb', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'lgb'])
        base_model.append(('lgb', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'bsvr'])
        base_model.append(('bsvr', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'knr'])
        base_model.append(('knr', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'ard'])
        base_model.append(('ard', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'enet'])
        base_model.append(('enet', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'mlp'])
        base_model.append(('mlp', temp))
        temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'bays'])
        base_model.append(('bays', temp))
        # temp = loadBaseModel(best_models.loc[best_models['estimator'] == 'orth'])
        # base_model.append(('orth', temp))


        top_model = RandomForestRegressor(random_state=self.rand_state, n_jobs=nthreads,
                                          max_depth=9, max_features='log2', max_samples=0.6, n_estimators=13000)#ExtraTreesRegressor(random_state=self.rand_state, n_jobs=nthreads)#LinearRegression()
        voting_model = VotingRegressor(estimators=base_model, n_jobs=nthreads)
        meta_model = StackingRegressor(estimators=base_model, final_estimator=top_model, cv=5, 
                                       passthrough=True, n_jobs=nthreads)
        voting_model.fit(self.x_train, self.y_train)
        meta_model.fit(self.x_train, self.y_train)

        # ___________________________________________________
        # Save meta learner & voting
        pickle.dump(voting_model, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_Voting_Model.pickle.dat", "wb"))
        pickle.dump(meta_model, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_Meta_Model.pickle.dat", "wb"))

        return loaded_model, voting_model, meta_model, self.train_x, self.train_y, self.test_x, self.test_y
    
    def finalFits(self, ml_model: any, voting_model: VotingRegressor, meta_model: StackingRegressor, 
                  out_features: str, best_model: str) -> Tuple[any, 
                                                                VotingRegressor,
                                                                StackingRegressor]:
        concated_x = pd.concat([self.train_x, self.test_x], axis=0)
        concated_x = concated_x.reset_index(drop=True)
        concated_y = np.concatenate([self.train_y, self.test_y])
        ml_model.fit(concated_x, concated_y)
        voting_model.fit(concated_x, concated_y)
        meta_model.fit(concated_x, concated_y)
        # Save models
        pickle.dump(ml_model, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_"+str(best_model)+"_final_Best_Model.pickle.dat", "wb"))
        pickle.dump(voting_model, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_final_Voting_Model.pickle.dat", "wb"))
        pickle.dump(meta_model, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_final_Meta_Model.pickle.dat", "wb"))

        return 

# --------------------------- Model Switcher --------------------------- #           
class ModelSwitch:
    def __init__(self, rand_state : int, nthreads : int) -> None:
        """ A calss object to load best parameters into
        ML models

        Parameters
        ----------
        rand_state : int
            Random state ssnumber
        nthreads : int
            Number of cores to be used
        """
        self.rand_state = rand_state
        self.nthreads = nthreads

    def modelName(self, model, best_params):
        default = "Incorrect model"
        return getattr(self, str(model), lambda: default)(best_params)
 
    def xgb(self, best_params):
        return xgb.XGBRegressor(random_state = self.rand_state, learning_rate = best_params['learning_rate'],
                max_depth = best_params['max_depth'], n_estimators = best_params['n_estimators'],
                colsample_bytree = best_params['colsample_bytree'], nthread=self.nthreads,
                min_child_weight = best_params['min_child_weight'], gamma = best_params['gamma'],
                subsample = best_params['subsample'])
    def rf(self, best_params):
        return RandomForestRegressor(random_state=self.rand_state, n_jobs=self.nthreads,
                max_depth = best_params['max_depth'], max_features = best_params['max_features'], 
                n_estimators = best_params['n_estimators'], max_samples = best_params['max_samples'])
    def hgb(self, best_params):
        return HistGradientBoostingRegressor(random_state=self.rand_state,
                learning_rate = best_params['learning_rate'], max_iter = best_params['max_iter'],
                max_depth = best_params['max_depth'], l2_regularization = best_params['l2_regularization'], 
                min_samples_leaf = best_params['min_samples_leaf'])
    def lgb(self, best_params):
        return ltb.LGBMRegressor(random_state=self.rand_state, n_jobs=self.nthreads,
                learning_rate = best_params['learning_rate'], n_estimators = best_params['n_estimators'],
                max_depth = best_params['max_depth'], reg_alpha = best_params['reg_alpha'], 
                reg_lambda = best_params['reg_lambda'], colsample_bytree = best_params['colsample_bytree'],
                subsample = best_params['subsample'])
    def bsvr(self, best_params):
        return BaggingRegressor(estimator=SVR(), n_jobs=self.nthreads, random_state=self.rand_state,
                n_estimators = best_params['n_estimators'], max_features = best_params['max_features'],
                max_samples = best_params['max_samples'])
    def knr(self, best_params):
        return KNeighborsRegressor(n_jobs=self.nthreads, 
                n_neighbors = best_params['n_neighbors'], algorithm = best_params['algorithm'])
    def ard(self, best_params):
        return ARDRegression(n_iter = best_params['n_iter'], tol = best_params['tol'],
                             alpha_1 = best_params['alpha_1'], alpha_2 = best_params['alpha_2'],
                             lambda_1 = best_params['lambda_1'], lambda_2 = best_params['lambda_2'],
                             fit_intercept = best_params['fit_intercept'])
    def orth(self, best_params):
        return OrthogonalMatchingPursuit(n_nonzero_coefs = best_params['n_nonzero_coefs'], tol = best_params['tol'],
                             fit_intercept = best_params['fit_intercept'],# normalize = best_params['normalize'],
                             precompute = best_params['precompute'])
    def enet(self, best_params):
        return ElasticNet(l1_ratio = best_params['l1_ratio'], 
                             alpha = best_params['alpha'], fit_intercept = best_params['fit_intercept'],
                             max_iter = best_params['max_iter'], tol = best_params['tol'])    
    def mlp(self, best_params):
        return MLPRegressor(hidden_layer_sizes = best_params['hidden_layer_sizes'], activation = best_params['activation'],
                             solver = best_params['solver'], alpha = best_params['alpha'],
                             batch_size = best_params['batch_size'], learning_rate = best_params['learning_rate'],
                             learning_rate_init = best_params['learning_rate_init'], max_iter = best_params['max_iter'])    
    def bays(self, best_params):
        return BayesianRidge(n_iter = best_params['n_iter'], tol = best_params['tol'],
                             alpha_1 = best_params['alpha_1'], alpha_2 = best_params['alpha_2'],
                             lambda_1 = best_params['lambda_1'], lambda_2 = best_params['lambda_2'],
                             fit_intercept = best_params['fit_intercept'])    

# --------------------------- A driver class --------------------------- #           
class RunMlModel:
    @staticmethod
    def main(argv):
        """ The driver class to run ML model
        
        Parameters 
        ----------
        argv: list
            taken from bash script
        """
        custom_name = argv[0]
        # target_name = argv[1] 
        nthreads     = int(argv[1])
        x_transform  = eval(argv[2])
        y_transform  = eval(argv[3])
        R2_thresh    = float(argv[4])
        count_thresh = int(argv[5])
        space        = 'actual_space' # actual_space / test_space
        SI           = False # SI system
        sample_type  = "Sub" #"All", "Sub", "test"
        weighted     = False
        pci          = True 
        t_type       = 'power' # 'log', 'power', 'quant' 
        if sample_type == "Sub" and pci:
            sample_type = "Sub_pca"

        # List of traget varaibles
        # temp        = json.load(open('data/ml_model_feature_names.json'))
        # del temp

        # ___________________________________________________
        # Bulid an instance of MlModel object and itterate through targets
        model = MlModel(custom_name) 
        # temporary holder
        temp        = json.load(open('data/model_feature_names.json'))
        target_list = temp.get('out_features')
        del temp

        for target_name in tqdm(target_list):
            # ___________________________________________________
            # Train models 
            print('\n******************* modeling parameter {0} starts here *******************\n'.format(target_name))
            model.loadData(out_feature=target_name, x_transform=x_transform,
                                y_transform=y_transform, R2_thresh=R2_thresh, count_thresh=count_thresh,
                                sample_type=sample_type, pci=pci, t_type=t_type)     
            print('end')
            best_model, best_params, best_models = model.findBestParams(out_features=target_name, nthreads=nthreads, 
                                                                                    space=space, weighted=weighted)
            best_model_orig = best_model
            ml_model, voting_model, meta_model, train_x, train_y, _, _, = model.runMlModel(best_model=best_model, best_params=best_params, 
                                                best_models=best_models, weighted=weighted, out_features=target_name, nthreads=nthreads)
            
            print('\n----------------- Results for best model -------------------\n')
            # # ___________________________________________________
            # # # save best model fit
            save_obj = sd.SaveOutput(train_id=model.train_sub_id, eval_id=model.eval_id, test_id=model.test_id,
                                        x_train=model.x_train, x_eval=model.x_eval, test_x=model.test_x,
                                        y_train=model.y_train, y_eval=model.y_eval, test_y=model.test_y,
                                        target_data_path = model.target_data_path, best_model=best_model, loaded_model=ml_model, 
                                        x_transform=x_transform, y_transform=y_transform, t_type=t_type,
                                        out_feature=target_name, custom_name=custom_name, SI=SI)
            save_obj.processData()

            print('\n----------------- Results for vote model -------------------\n')
            # # ___________________________________________________
            # # save best model fit
            best_model = 'vote'
            save_obj = sd.SaveOutput(train_id=model.train_sub_id, eval_id=model.eval_id, test_id=model.test_id,
                                        x_train=model.x_train, x_eval=model.x_eval, test_x=model.test_x,
                                        y_train=model.y_train, y_eval=model.y_eval, test_y=model.test_y,
                                        target_data_path = model.target_data_path, best_model=best_model, loaded_model=voting_model, 
                                        x_transform=x_transform, y_transform=y_transform, t_type=t_type,
                                        out_feature=target_name, custom_name=custom_name, SI=SI)
            save_obj.processData()

            print('\n----------------- Results for meta model -------------------\n')
            # # ___________________________________________________
            # # plot meta model fit
            best_model = 'meta'
            save_obj = sd.SaveOutput(train_id=model.train_sub_id, eval_id=model.eval_id, test_id=model.test_id,
                                        x_train=model.x_train, x_eval=model.x_eval, test_x=model.test_x,
                                        y_train=model.y_train, y_eval=model.y_eval, test_y=model.test_y,
                                        target_data_path = model.target_data_path, best_model=best_model, loaded_model=meta_model, 
                                        x_transform=x_transform, y_transform=y_transform, t_type=t_type,
                                        out_feature=target_name, custom_name=custom_name, SI=SI)
            save_obj.processData()
            
            # ___________________________________________________
            # Final training
            model.finalFits(ml_model, voting_model, meta_model, target_name, best_model_orig)

            print('\n----------------- Feature importance -------------------\n')
            # # ___________________________________________________
            # # plot feature importance
            try:
                fimp_object = fimp.FeatureImportance(custom_name, best_model)
                fimp_object.plotImportance(model=ml_model, out_features=target_name,
                                            train_x=train_x, train_y=train_y)
                fimp_object.plotShapImportance(model=ml_model, out_features=target_name, 
                                                train_x=train_x)
            except Exception as e:       
                print("An exception occurred due to shap internal errors!")  
                print(e)      
            print('\n**************** modeling parameter {0} ends here ****************\n'.format(target_name))
            print('end')

if __name__ == "__main__":
    # RunMlModel.main(['test2', -1, "True", "True", 0.8, 5])
    RunMlModel.main(sys.argv[1:])

