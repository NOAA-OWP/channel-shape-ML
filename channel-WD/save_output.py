import pandas as pd
import numpy as np
import pickle
import scipy
# --------------------------- Save data and predictions --------------------------- #
class SaveOutput:
    """ 
    A calss object to save model outputs
    Parameters
    ----------
    train_id : pd.DataFrame
        Training set positional IDs
    eval_id : pd.DataFrame
        Evaluation set positional IDs
    test_id : pd.DataFrame
        Test set positional IDs
    out_features : str
        Name of the FHG coeficent
    custom_name: str
        A custom name for the running instance
    x_train : pd.DataFrame
        Predictor variables for training
    x_eval: pd.DataFrame 
        Predictor variables for evaluation
    test_x: pd.DataFrame    
        Predictor variables for testing
    y_train: np.array: 
        Target variables for training
    y_eval: np.array
        Target variables for evaluation
    test_y: np.array
        Target variables for testing 
    best_model: str
        Name of the best model
    loaded_model: any
        Model structure and weights 
    x_transform: bool
        Apply transformation to predictors
    y_transform: bool 
        Apply transformation to target 
    SI: bool
        Consider sientific system when plotting 
    """
    def __init__(self, train_id: pd.DataFrame, eval_id: pd.DataFrame,
                 test_id: pd.DataFrame, out_feature: str, custom_name: str,
                 x_train: pd.DataFrame, x_eval: pd.DataFrame, test_x: pd.DataFrame,
                 train_columns: list, m_x_train: pd.DataFrame, m_x_eval: pd.DataFrame, m_x_test: pd.DataFrame,
                 y_train: np.array, y_eval: np.array, test_y: np.array,
                 target_data_path: str, best_model: str, loaded_model: any, 
                 x_transform: bool, y_transform: bool, t_type: str, SI: bool) -> None:
        
        self.train_id               = train_id
        self.eval_id                = eval_id
        self.test_id                = test_id
        self.out_feature            = out_feature
        self.custom_name            = custom_name
        self.x_train                = x_train
        self.x_eval                 = x_eval
        self.test_x                 = test_x
        self.train_columns          = train_columns
        self.m_x_train              = m_x_train
        self.m_x_eval               = m_x_eval
        self.m_x_test               = m_x_test
        self.y_train                = y_train
        self.y_eval                 = y_eval
        self.test_y                 = test_y
        self.target_data_path       = target_data_path
        self.best_model             = best_model
        self.loaded_model           = loaded_model
        self.y_trans                = y_transform
        self.x_trans                = x_transform
        self.t_type                 = t_type
        self.SI                     = SI

        self.predictions_train      = 0
        self.predictions_valid      = 0
        self.predictions_test       = 0
        self.merged_data            = pd.DataFrame([])
        
    def processData(self) -> None:
        """ 
        A preprocessing step
        """
        # data transformation ----------------------------------------
        # min_length = min(len(self.loaded_model.feature_names_in_), len(self.train_columns))

        # # Print values side by side
        # for i in range(min_length):
        #     print(f'{self.loaded_model.feature_names_in_[i]}  {self.train_columns[i]}')

        self.predictions_train = self.loaded_model.predict(self.m_x_train)
        self.predictions_valid = self.loaded_model.predict(self.m_x_eval)
        self.predictions_test = self.loaded_model.predict(self.m_x_test)

        pc_columns = [col for col in self.x_train.columns if '_pc' in col]
        pc_columns = pc_columns + ["R2", "siteID"]
        self.x_train = self.x_train.drop(columns=pc_columns, axis=1)
        self.x_eval = self.x_eval.drop(columns=pc_columns, axis=1)
        self.test_x = self.test_x.drop(columns=pc_columns, axis=1)
        # self.predictions_train_orig = self.predictions_train.copy()
        # self.predictions_valid_orig = self.predictions_valid.copy()
        # self.predictions_test_orig = self.predictions_test.copy()
        # self.y_train_orig = self.y_train.copy()
        # self.y_eval_orig = self.y_eval.copy()
        # self.test_y_orig = self.test_y.copy()

        if self.y_trans:
            if self.t_type != 'log':
                t_y = pickle.load(open(self.custom_name+'/model/'+'train_y_'+self.out_feature+'_tansformation.pkl', "rb"))
                self.predictions_train = t_y.inverse_transform(self.predictions_train.reshape(-1,1)).ravel()
                self.predictions_valid = t_y.inverse_transform(self.predictions_valid.reshape(-1,1)).ravel()
                self.predictions_test = t_y.inverse_transform(self.predictions_test.reshape(-1,1)).ravel()
                # train_temp = self.y_train.copy()
                # eval_temp = self.y_eval.copy()
                # test_temp = self.test_y.copy()
                self.y_train = t_y.inverse_transform(self.y_train.reshape(-1,1)).ravel()
                self.y_eval = t_y.inverse_transform(self.y_eval.reshape(-1,1)).ravel()
                self.test_y = t_y.inverse_transform(self.test_y.reshape(-1,1)).ravel()
            else:
                self.y_train = np.exp(self.y_train)
                self.y_eval = np.exp(self.y_eval)
                self.test_y = np.exp(self.test_y)

        # if self.x_trans:
        #     if self.t_type != 'log':
        #         t_x = pickle.load(open(self.custom_name+'/model/'+'train_x_'+self.out_feature+'_tansformation.pkl', "rb"))
        #         col_names = self.x_train.columns
        #         self.x_train = t_x.inverse_transform(self.x_train)
        #         self.x_train = pd.DataFrame(data=self.x_train,
        #                                 columns=col_names).reset_index(drop=True)
        #         self.x_eval = t_x.inverse_transform(self.x_eval)
        #         self.x_eval = pd.DataFrame(data=self.x_eval,
        #                                 columns=col_names).reset_index(drop=True)
        #         self.test_x = t_x.inverse_transform(self.test_x)
        #         self.test_x = pd.DataFrame(data=self.test_x,
        #                                     columns=col_names).reset_index(drop=True)
        #     else:
        #         self.x_train = np.exp(self.x_train)
        #         self.x_eval = np.exp(self.x_eval)
        #         self.test_x = np.exp(self.test_x)

        # ___________________________________________________
        # Build complete dataframe
        train_attr = self.x_train.copy()
        eval_attr = self.x_eval.copy()
        test_attr = self.test_x.copy()

        train_attr['split'] = 'train'
        eval_attr['split'] = 'eval'
        test_attr['split'] = 'test'

        train_attr['predicted'] = self.predictions_train
        eval_attr['predicted'] = self.predictions_valid
        test_attr['predicted'] = self.predictions_test

        train_attr['target'] = self.y_train
        eval_attr['target'] = self.y_eval
        test_attr['target'] = self.test_y

        # Add additional attributes
        train_attr = pd.concat([train_attr, self.train_id[['siteID']]], axis=1)
        eval_attr = pd.concat([eval_attr, self.eval_id[['siteID']]], axis=1) 
        test_attr = pd.concat([test_attr, self.test_id[['siteID']]], axis=1)  
        self.merged_data = pd.concat([train_attr, eval_attr, test_attr], axis=0)

        data_attr = pd.read_parquet(self.target_data_path, engine='pyarrow')
        data_attr.astype({'siteID': 'string'})
        #list_attr = list(set(data_attr.columns.to_list()) - set(['lat', 'long']))
        #data_attr = data_attr[list_attr]
        if self.out_feature.startswith("Y"):

            data_attr = data_attr[['siteID','Count','coe','exp','R2','Y_bf','Y_in']]
        else:
            data_attr = data_attr[['siteID','Count','coe','exp','R2','TW_bf','TW_in']]

        self.merged_data = self.merged_data.merge(data_attr, on='siteID', how='inner')

        if self.SI:
            self.merged_data['predicted'] = self.merged_data['predicted'] * 0.3048
            self.merged_data['target'] = self.merged_data['target'] * 0.3048
        
                # Spitout some metrics
        # ___________________________________________________
        def calRsquared(y_true, y_pred):
            """ 
            R2 based on linear regression 
            rgs:
                y_true ([pd.series]): Observations 
                y_pred ([pd.series]): Predictions
            Returns:
                [float]: normalized root mean square error
            """
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_pred)
            cof2 = r_value**2
            return cof2

        def rSquared2(df):
            return calRsquared(df['target'], df['predicted'])
        
        print("\n __________________ Stats for "+str(self.custom_name)+" "+self.best_model+" "+self.out_feature+" _________________________ \n")
        print("Training Accuracy: %.2f%%" % (rSquared2(train_attr)*100))
        print("Validation Accuracy: %.2f%%" % (rSquared2(eval_attr)*100))
        print("Testing Accuracy: %.2f%%" % (rSquared2(test_attr)*100))
        
        # ___________________________________________________
        # Save dataframe
        self.merged_data = self.merged_data.loc[:, ~self.merged_data.columns.duplicated()]
        self.merged_data.to_parquet(self.custom_name+'/metrics/'+str(self.custom_name)+'_'+self.best_model+'_'+self.out_feature+'.parquet')
        print("\n __________________ Saved _________________________ \n")
        return
