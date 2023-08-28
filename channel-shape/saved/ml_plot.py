# Libraries
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import pyplot
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import pickle
from statsmodels.distributions.empirical_distribution import ECDF
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, max_error, explained_variance_score, mean_squared_error, mean_absolute_percentage_error, r2_score
# --------------------------- Plot fits and metrics --------------------------- #
class PlotMlFit:
    """ 
    A calss object to plot model performance
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
    val_method: str
        Fit to observed or curve
        Options:
        "Fit" 
        "Observed"
    SI: bool
        Consider sientific system when plotting 
    """
    def __init__(self, train_id: pd.DataFrame, eval_id: pd.DataFrame,
                 test_id: pd.DataFrame, out_features: str, custom_name: str,
                 x_train: pd.DataFrame, x_eval: pd.DataFrame, test_x: pd.DataFrame,
                 y_train: np.array, y_eval: np.array, test_y: np.array,
                 best_model: str, loaded_model: any, x_transform: bool, y_transform: bool,
                 val_method: str, SI: bool) -> None:
        
        self.train_id               = train_id
        self.eval_id                = eval_id
        self.test_id                = test_id
        self.out_features           = out_features
        self.custom_name            = custom_name
        self.x_train                = x_train
        self.x_eval                 = x_eval
        self.test_x                 = test_x
        self.y_train                = y_train
        self.y_eval                 = y_eval
        self.test_y                 = test_y
        self.best_model             = best_model
        self.loaded_model           = loaded_model
        self.y_trans                = y_transform
        self.x_trans                = x_transform
        self.val_method             = val_method
        self.SI                     = SI

        self.predictions_train      = 0
        self.predictions_valid      = 0
        self.predictions_test       = 0
        self.predictions_train_orig = 0
        self.predictions_valid_orig = 0
        self.predictions_test_orig  = 0
        self.y_train_orig           = 0
        self.y_eval_orig            = 0
        self.test_y_orig            = 0
        

    
    def processData(self) -> None:
        """ 
        A preprocessing step
        """
        # data transformation ----------------------------------------
        self.predictions_train = self.loaded_model.predict(self.x_train)
        self.predictions_valid = self.loaded_model.predict(self.x_eval)
        self.predictions_test = self.loaded_model.predict(self.test_x)

        self.predictions_train_orig = self.predictions_train.copy()
        self.predictions_valid_orig = self.predictions_valid.copy()
        self.predictions_test_orig = self.predictions_test.copy()
        self.y_train_orig = self.y_train.copy()
        self.y_eval_orig = self.y_eval.copy()
        self.test_y_orig = self.test_y.copy()
        
        def applyInvScaler(arr):
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
            data_original = min_max_scaler.inverse_transform(arr.values.reshape(-1, 1))
            return data_original.flatten()
        
        if self.y_trans:
            # Transform
            t_y = pickle.load(open(self.custom_name+'/model/'+'train_y_'+self.out_features+'_tansformation.pkl', "rb"))
            self.predictions_train = t_y.inverse_transform(self.predictions_train.reshape(-1,1))
            self.predictions_valid = t_y.inverse_transform(self.predictions_valid.reshape(-1,1))
            self.predictions_test = t_y.inverse_transform(self.predictions_test.reshape(-1,1))
            self.y_train = t_y.inverse_transform(self.y_train.reshape(-1,1))
            self.y_eval = t_y.inverse_transform(self.y_eval.reshape(-1,1))
            self.test_y = t_y.inverse_transform(self.test_y.reshape(-1,1))
            # Scale
            min_max_scaler = pickle.load(open(self.custom_name+'/model/'+'train_'+self.out_features+'_scaled.pkl', "rb"))
            self.predictions_train = min_max_scaler.inverse_transform(self.predictions_train).ravel()
            self.predictions_valid = min_max_scaler.inverse_transform(self.predictions_valid).ravel()
            self.predictions_test = min_max_scaler.inverse_transform(self.predictions_test).ravel()
            self.y_train = min_max_scaler.inverse_transform(self.y_train).ravel()
            self.y_eval = min_max_scaler.inverse_transform(self.y_eval).ravel()
            self.test_y = min_max_scaler.inverse_transform(self.test_y).ravel()

        if self.x_trans:
            t_x = pickle.load(open(self.custom_name+'/model/'+'train_x_'+self.out_features+'_tansformation.pkl', "rb"))
            col_names = self.x_train.columns
            self.x_train = t_x.inverse_transform(self.x_train)
            self.x_train = pd.DataFrame(data=self.x_train,
                                    columns=col_names).reset_index(drop=True)
            self.x_train = self.x_train.apply(applyInvScaler)
            self.x_eval = t_x.inverse_transform(self.x_eval)
            self.x_eval = pd.DataFrame(data=self.x_eval,
                                    columns=col_names).reset_index(drop=True)
            self.x_eval = self.x_eval.apply(applyInvScaler)
            self.test_x = t_x.inverse_transform(self.test_x)
            self.test_x = pd.DataFrame(data=self.test_x,
                                        columns=col_names).reset_index(drop=True)
            self.test_x = self.test_x.apply(applyInvScaler)
        return
    
    def processADCP(self) -> pd.DataFrame:
        """ 
        merges ADCP data with predicted values form ML model 

        Returns 
        ----------
        adcp_pred: pd.DataFrame
            A dataframe containg observed and predicted vbalues
        """

        # adcp calculations ------------------------------------------
        fhg = pd.read_parquet('data/Processed_merged_fhg.parquet', engine='pyarrow')
        
        # ___________________________________________________
        # restore coordiantes 
        fhg = gpd.GeoDataFrame(
            fhg, geometry=gpd.points_from_xy(fhg.lng, fhg.lat))

        adcp = pd.read_parquet('data/Processed_adcp.parquet', engine='pyarrow')
        adcp = adcp[['date','siteID','Q','TW_x','V_x','Ymean',"c","f","a","b","k","m","r"]]
        adcp = adcp.drop_duplicates(subset=['Q', 'TW_x'], keep='last')
        adcp = adcp.rename(columns={"TW_x": "TW", "V_x": "V"})
        
        # ___________________________________________________
        # remove estuary negative discharges
        adcp = adcp.loc[(adcp['Q']>0)&(adcp['V']>0)&(adcp['Ymean']>0)&(adcp['TW']>0)]
        adcp.to_csv('cache/temp.csv', index=False)
        del(adcp)

        train_pred_pd = pd.DataFrame({"siteID":self.train_id.siteID.ravel(), "InChannelMaxFlow":self.train_id.nwis_25.ravel(),
                                      self.out_features+"_pred":self.predictions_train, "set":len(self.train_id)*[0]})
        train_pred_pd = train_pred_pd.reset_index(drop=True)
        eval_pred_pd = pd.DataFrame({"siteID":self.eval_id.siteID.ravel(), "InChannelMaxFlow":self.eval_id.nwis_25.ravel(),
                                     self.out_features+"_pred":self.predictions_valid, "set":len(self.eval_id)*[1]})
        eval_pred_pd = eval_pred_pd.reset_index(drop=True)
        test_pred_pd = pd.DataFrame({"siteID":self.test_id.siteID.values.reshape((-1,)), "InChannelMaxFlow":self.test_id.nwis_25.values.reshape((-1,)),
                                     self.out_features+"_pred":self.predictions_test, "set":len(self.test_id)*[2]})
        test_pred_pd = test_pred_pd.reset_index(drop=True)
        pred_pd = pd.concat([train_pred_pd, eval_pred_pd, test_pred_pd], axis=0, ignore_index=True)
        pred_pd.astype({'siteID': 'string'})

        def saveMemory(chunk, counter):
            df = pd.merge(chunk.reset_index(), pred_pd.reset_index(), on='siteID', how='inner') 
            df = df.drop_duplicates(subset=['Q', 'TW'], keep='last')
            df = df[df.columns[~df.columns.isin(['index_x','index_y'])]].reset_index(drop = True)
            if counter == 0:
                df.to_csv("cache/store.csv", mode="a", index=False)
            else:
                df.to_csv("cache/store.csv", mode="a", header=False ,index=False)

        adcp_chunk = pd.read_csv("cache/temp.csv", dtype={'siteID': 'string'}, chunksize=50000)
        counter = 0
        for chunk in adcp_chunk:
            saveMemory(chunk, counter)
            counter += 1 

        del(train_pred_pd, eval_pred_pd, test_pred_pd, adcp_chunk)
        adcp_pred = pd.read_csv("cache/store.csv", dtype={'siteID':'string', 'date':'string', 'set':'int32'})

        os.remove("cache/temp.csv")
        os.remove("cache/store.csv")

        # ___________________________________________________
        # drop floodplain flows and keep bankfull values
        adcp_pred = adcp_pred.loc[adcp_pred['Q'] <= adcp_pred['InChannelMaxFlow']]
        
        # ___________________________________________________
        # add coordiantes
        adcp_pred = adcp_pred.merge(fhg[['siteID', 'geometry']].drop_duplicates(subset=['siteID']), on='siteID', how='inner')
        # -----------------------------------------------------------
        return adcp_pred
    
    def plotMetrics(self, adcp_pred: pd.DataFrame, model_name: str) -> Tuple[list, pd.DataFrame]:
        print("\n $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ "+str(self.custom_name)+"_"+self.out_features+" $$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
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
        # 0.44290031596942386
        def calNrmse(y_true, y_pred):
            """
            Normalized Root Mean Square Error.
            Args:
                y_true ([pd.series]): Observations 
                y_pred ([pd.series]): Predictions
            Returns:
                [float]: normalized root mean square error
            """
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            nrmse = abs(rmse / (np.maximum(y_true.max(), y_pred.max())  - np.minimum(y_true.min(), y_pred.min())))
            return nrmse
        #1.1070588457846888
        def rsquared(obs, pred):
            """ Return R^2 where obs and pred are array """
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obs, pred)
            return r_value**2
        
        if self.best_model == 'xgb':
            results = self.loaded_model.evals_result()
            epochs = len(results['validation_0']['mae'])
            #-------------------------- Overfit check --------------------------
            x_axis = range(0, epochs)
            
            # ___________________________________________________
            # plot log loss
            fig, ax = pyplot.subplots()
            ax.plot(x_axis, results['validation_0']['mae'], label='Train')
            ax.plot(x_axis, results['validation_1']['mae'], label='Test')
            ax.legend()
            pyplot.ylabel('Mean absolute error')
            pyplot.title('XGBoost mae')
            plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_mae'+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            pyplot.show()

            # ___________________________________________________
            # plot classification error
            fig, ax = pyplot.subplots()
            ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
            ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
            ax.legend()
            pyplot.ylabel('Regresiion Error')
            pyplot.title('XGBoost rmse')
            plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_rmse'+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            pyplot.show()

        #-------------------------- model performance check ----------------------

        def plotPerformance(observed, prediction, name, ptype="river char") -> None:
            """
            ptype is pty "river char" or "normal"
            """
            prediction_orig = prediction.ravel().copy()
            observed_orig = observed.ravel().copy()
            
            if ptype == "normal":
                if self.y_trans:
                    min_max_scaler = pickle.load(open(self.custom_name+'/model/'+'train_'+self.out_features+'_scaled.pkl', "rb"))
                    prediction_orig = min_max_scaler.transform(prediction)
                    observed_orig = min_max_scaler.transform(observed)
                    t_y = pickle.load(open(self.custom_name+'/model/'+'train_y_'+self.out_features+'_tansformation.pkl', "rb"))
                    prediction_orig = t_y.transform(prediction_orig).ravel()
                    observed_orig = t_y.transform(observed_orig).ravel()

            print("Plotting paramter {0} {1} performance -------------- ".format(str(self.out_features), name))
            fig, axs = plt.subplots(1, 1, figsize=(5,5))
            axs.plot(np.arange(len(observed)), observed, 'o', color='black', label="Observed")  
            axs.plot(np.arange(len(prediction)), prediction, '.', color='red', label="Prediction")
            axs.set_title(str(self.best_model)+'_comparison')

            axs.set(xlabel='Observed points', ylabel=name+' Value')
            axs.label_outer()
            axs.legend(loc="upper left")
            plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_match_'+name+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()

            fig, axs = plt.subplots(1, 1, figsize=(5,5))
            sns.residplot(x=observed, y=prediction, 
                                    ax =axs,
                                    lowess=True,
                                    scatter_kws={'alpha': 0.5},
                                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
            axs.set_title(str(self.best_model)+'_residual')

            axs.set(xlabel=name+' Value', ylabel=name+' Residuals')
            axs.label_outer()
            plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_residual_'+name+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()

            dic = {
                    name+' actual': observed.ravel(),
                    name+' preds': prediction.ravel(),
                    'Model': [str(self.best_model)] * len(observed)
                    }
            data1 = pd.DataFrame(dic)
            b = sns.lmplot(x=name+' actual', y=name+' preds', col="Model", data=data1, x_estimator=np.mean, 
                            scatter_kws={'color': 'blue', 'alpha':0.5, 's':70}, 
                            line_kws={"linewidth": 3, 'color': 'black'})  

            _, ylabels = plt.yticks()
            _, xlabels = plt.xticks()
            b.set_yticklabels(ylabels, size=13)
            b.set_xticklabels(xlabels, size=13)
            # b.set(xlim=(0,10),ylim=(0,10))
            uplim = max(prediction)+0.01
            lowlim = min(prediction)
            b.axes[0,0].text(lowlim, uplim,name+" Accuracy: %.2f%%" % (calRsquared(observed_orig, prediction_orig)*100), fontsize=9)
            plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_fit_'+name+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()

        def plotError(df: pd.DataFrame, y_label: str, y_pred_label: str, name: str) -> list:
            """
            Plot error figures

            Parameters:
            ----------
            df: pd.DataFrame
                Data containing true and predicted 
            y_label: str
                Observed varible 
            y_pred_label: str
                Estimated variable 
            name: str
                A custom string
            
            Returns:
            ----------
            List of all metrics and values
            """
            if self.SI:
                df[y_label] = df[y_label] * 0.3048
                df[y_pred_label] = df[y_pred_label] * 0.3048
            def compMae(df):
                return mean_absolute_error(df[y_label], df[y_pred_label])
            def compMe(df):
                return max_error(df[y_label], df[y_pred_label])
            def compEvar(df):
                return explained_variance_score(df[y_label], df[y_pred_label])
            def compMse(df):
                return mean_squared_error(df[y_label], df[y_pred_label])/1000# to shrink numbers devided by 1000
            def compNrmse(df):
                return calNrmse(df[y_label], df[y_pred_label])
            def compMap(df):
                return mean_absolute_percentage_error(df[y_label], df[y_pred_label])
            def compNnse(df):
                nse = 1-(np.sum((df[y_label]-df[y_pred_label])**2)/np.sum((df[y_label]-np.mean(df[y_label]))**2))
                return 1/(2-nse), nse 
            def rSquared(df):
                return r2_score(df[y_label], df[y_pred_label])
            def rSquared2(df):
                return calRsquared(df[y_label], df[y_pred_label])
            def compKGE(df):
                y_true = np.array(df[y_label])
                y_pred = np.array(df[y_pred_label])
                m1, m2 = np.mean(y_true), np.mean(y_pred)
                r = np.sum((y_true - m1) * (y_pred - m2)) / (np.sqrt(np.sum((y_true - m1) ** 2)) * np.sqrt(np.sum((y_pred - m2) ** 2)))
                beta = m2 / m1
                gamma = (np.std(y_pred) / m2) / (np.std(y_true) / m1)
                return pd.Series(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))
            def extractCol(tup):
                index = np.array(tup.index)
                nse = np.array([item[1] for item in tup])
                nnse = np.array([item[0] for item in tup])
                return  np.column_stack((index, nse, nnse))
            def filterQuartile(arr):
                df = pd.DataFrame(arr, columns = ['siteID', 'metric'])
                df.dropna(inplace=True)
                # ___________________________________________________
                # IQR
                Q1 = np.percentile(df['metric'], 25, method = 'midpoint')
                Q3 = np.percentile(df['metric'], 75, method = 'midpoint')
                IQR = Q3 - Q1
                # ___________________________________________________
                # Above Upper bound
                upper=Q3+1.5*IQR
                # ___________________________________________________
                # Below Lower bound
                lower=Q1-1.5*IQR
                df = df.loc[(df['metric'] >= lower)&(df['metric'] <= upper)]
                return np.array(df['siteID']), np.array(df['metric'])
            
            df_gp = df.groupby(['siteID'])
            mae = np.array(df_gp.apply(compMae))
            me = np.array(df_gp.apply(compMe))
            evar = df_gp.apply(compEvar)
            evar = evar.reset_index()
            evar_idx, evar = filterQuartile(np.array(evar))
            mse = np.array(df_gp.apply(compMse))
            nrmse = np.array(df_gp.apply(compNrmse))
            map = df_gp.apply(compMap)
            map = map.reset_index()
            map_idx, map = filterQuartile(np.array(map))
            nnses = df_gp.apply(compNnse)
            nnsea_arr = extractCol(nnses)
            nse_idx, nse = filterQuartile(nnsea_arr[:,0:2])
            nnse_idx, nnse = filterQuartile(nnsea_arr[:,[0,2]])
            kge = df_gp.apply(compKGE)
            kge = kge.reset_index()
            kge_idx, kge = filterQuartile(np.array(kge))
            r2 = np.array(df_gp.apply(rSquared))
            r2[r2 < 0] = 0
            r2_2 = np.array(df_gp.apply(rSquared2))

            # ___________________________________________________
            # some stats
            print("Percent stations R2 less than 0.1 (bad correlation): {0}".format(len(r2[r2<=0.1])/len(r2)))
            print("Percent stations R2 greater than 0.4 (Good correlation): {0}".format(len(r2[r2>=0.4])/len(r2)))
            print("Percent stations R2 greater than 0.7 (High correlation): {0}".format(len(r2[r2>=0.7])/len(r2)))
            print("Percent stations R2 (linear reg) greater than 0.4 (Good correlation): {0}".format(len(r2_2[r2_2>=0.4])/len(r2_2)))
            print("Percent stations R2 (linear reg) greater than 0.7 (High correlation): {0}".format(len(r2_2[r2_2>=0.7])/len(r2_2)))
            print("Percent stations KGE greater than 0.5 (Suitable simulation): {0}".format(len(kge[kge>=0.5])/len(kge)))
            print("Percent stations KGE greater than 0.75 (Excellent simulation): {0}".format(len(kge[kge>=0.75])/len(kge)))
            print("Percent stations NNSE greater than 0.5 (Satisfactory): {0}".format(len(nnse[nnse>=0.5])/len(nnse)))
            print("Percent stations NNSE greater than 0.666 (Good): {0}".format(len(nnse[nnse>=0.666])/len(nnse)))
            print("Percent stations NNSE greater than 0.77 (Very Good): {0}".format(len(nnse[nnse>=0.77])/len(nnse)))
            print("Percent stations NSE greater than 0 (Satisfactory): {0}".format(len(nse[nse>=0])/len(nse)))
            print("Percent stations NSE greater than 0.5 (Good): {0}".format(len(nse[nse>=0.5])/len(nse)))
            print("Percent stations NSE greater than 0.7 (Very good): {0}".format(len(nse[nse>=0.7])/len(nse)))
            print("Percent stations ExpVar greater than 0.5: {0}".format(len(evar[evar>=0.5])/len(evar)))

            print('Site average performance:')
            site_ave = df_gp.mean()
            plotPerformance(np.array(site_ave[y_label]).reshape((-1,1)), np.array(site_ave[y_pred_label]).reshape((-1,1)), "site_Average_"+str(name)) 
            
            metrics = [(evar, 'Explained variance score', 'EVar', evar_idx, 'rainbow_r'),
                        (mae, 'Mean absolute error', 'MAE', [], 'rainbow'), 
                        (me, 'Max error', 'ME', [], 'rainbow'),
                        (mse, 'Mean squared error','MSE', [], 'rainbow'),
                        (nrmse, 'Normalized root mean squared error','NRMSE', [], 'rainbow'),
                        (map, 'Mean absolute percentage error', 'MAP', map_idx, 'rainbow'),
                        (nnse, 'Normalized NSE', 'NNSE', nnse_idx, 'rainbow_r'), 
                        (nse, 'NSE', 'NSE', nse_idx, 'rainbow_r'),
                        (kge, 'Kling-Gupta Efficiency', 'KGE', kge_idx, 'rainbow_r'),
                        (r2, 'R Squared', 'R2', [], 'rainbow_r'),
                        (r2_2, 'linear R Squared', 'R2_l', [], 'rainbow_r')]
             
            def plotCDF(metric, xlabel, ylabel):
                fig, axs = plt.subplots(1, 1, figsize=(8,4))
                ecdf = ECDF(metric)
                axs.plot(ecdf.x, ecdf.y, linestyle='--', marker='o', color='black')
                ecdf.x[(ecdf.x>1e30) | (ecdf.x<-1e30)]=0
                axs.axvline(x=np.median(ecdf.x), color='r', ls='--', label='median')
                axs.axhline(y=0.5, color='r', ls='--', label='50%')
                if ylabel == 'NNSE':
                    axs.axvline(x=0.66, color='b', ls='--', label='Good')
                axs.set(xlabel=xlabel, ylabel='CDF('+ylabel+')')
                plt.legend(loc="upper left")
                my_plot = plt.gcf()
                plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_CDF_'+str(name)+'_'+ylabel+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                plt.show()   
            
            def plotMap(metric, xlabel, ylabel, idx, cmap):
                temp_df = df.groupby(['siteID', gpd.GeoSeries(df['geometry']).to_wkt()]).agg({y_pred_label:[np.median], 
                                                                                              self.out_features+"_pred":[np.median], 
                                                                                              self.out_features:[np.median]}).reset_index()
                temp_df.columns = list(temp_df.columns)
                if len(temp_df) == len(metric):
                    temp_df[ylabel] = metric
                else:
                    temp2_df = pd.DataFrame(np.column_stack((idx, metric)), columns = ['siteID', ylabel])
                    temp_df = temp_df.rename(columns={temp_df.columns[0]: "siteID"})
                    temp_df = temp_df.merge(temp2_df[['siteID', ylabel]].drop_duplicates(subset=['siteID']), on='siteID', how='inner')
                    del(temp2_df)
                temp_df = temp_df.astype({ylabel:'float'})
                temp_df['geometry'] = gpd.GeoSeries.from_wkt(temp_df[('level_1', '')])
                temp_df = gpd.GeoDataFrame(temp_df)
                temp_df.columns = temp_df.columns.to_flat_index()
                temp_df.columns = ['_'.join(col) for col in temp_df.columns.values]
                temp_df = temp_df.rename(columns={"g_e_o_m_e_t_r_y": "geometry", 
                                                  "s_i_t_e_I_D": "siteID", 
                                                  temp_df.columns[3]: self.out_features+"_pred", 
                                                  temp_df.columns[4]:self.out_features, 
                                                  temp_df.columns[5]: ylabel})
                temp_df.set_geometry("geometry")

                # path = gplt.datasets.get_path("contiguous_usa")
                # contiguous_usa = gpd.read_file(path)
                contiguous_usa = gpd.read_file('maps/contiguous_usa.sqlite')
                fig, ax = pyplot.subplots()
                ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea(), figsize=(20, 20))
                ax.set_title(xlabel, fontsize=30)
                # print(temp_df.dtypes)
                gplt.pointplot(temp_df,
                                ax=ax,
                                hue=ylabel,
                                legend=True,
                                legend_kwargs={'orientation': 'horizontal',
                                               'shrink': 0.7, 'fraction':0.1, 'pad':0},
                                s=7,
                                cmap=cmap
                )
                temp_df = pd.DataFrame(temp_df.drop(['geometry'], axis=1))
                temp_df = temp_df.rename(columns={'level_1_': "geometry"})
                temp_df.to_parquet(self.custom_name+'/metrics/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_'+str(name)+'_'+ylabel+'.parquet')
                del temp_df
                fig = ax.figure
                cb_ax = fig.axes[1] 
                cb_ax.tick_params(labelsize=30)
                # my_plot = plt.gcf()
                plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_cMAP_'+str(name)+'_'+ylabel+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                plt.show()  
                plt.close()
            
            def plotScatter(df: pd.DataFrame) -> None:
                stat_list_obs = [y_label+'_median',(y_label, 'mean'),y_label+'_min',y_label+'_max',(y_label, 'std')]
                stat_list_pred = [y_pred_label,(y_pred_label, 'mean'),y_pred_label,y_pred_label,(y_pred_label, 'std')]
                stat_list = zip(stat_list_obs, stat_list_pred)
                cmap = 'jet'
                df['error'] = df[y_label] - df[y_pred_label]

                df = df.reset_index(inplace=False)

                temp_min = df.copy()
                temp_min[y_label+'_min'] = abs(df.groupby('siteID')[y_label].transform('min'))
                temp_min = temp_min.loc[temp_min[y_label]==temp_min[y_label+'_min']]

                temp_max = df.copy()
                temp_max[y_label+'_max'] = abs(df.groupby('siteID')[y_label].transform('max'))
                temp_max = temp_max.loc[temp_max[y_label]==temp_max[y_label+'_max']]

                temp_med = df.copy()
                temp_med[y_label+'_median'] = abs(df.groupby('siteID')[y_label].transform('median'))
                temp_med = temp_med.loc[temp_med[y_label]==temp_med[y_label+'_median']]
                
                vertical_concat = pd.concat([temp_min, temp_max, temp_med], axis=0)
                del temp_min, temp_max, temp_med

                temp_df = df.copy()
                temp_df = temp_df.groupby(['siteID'], as_index=False).agg({y_label:[np.mean, np.std], y_pred_label:[np.mean, np.std]})
                temp_df = pd.merge(df, temp_df, on=['siteID'])

                temp_df = pd.merge(temp_df, vertical_concat, suffixes=('', '_y'), on="index", how="inner") 
                temp_df.drop([i for i in temp_df.columns if '_y' in i], axis=1, inplace=True)
                del vertical_concat

                for stat_obs, stat_pred in tqdm(stat_list):
                    sub_temp_df = temp_df.dropna(subset=[stat_obs])
                    sub_temp_df = sub_temp_df[~sub_temp_df[stat_obs].isin([np.nan, np.inf, -np.inf])]
                    sub_temp_df = sub_temp_df.sort_values(by=[stat_obs])
                    values = np.vstack([sub_temp_df[stat_obs], sub_temp_df[stat_pred]])
                    kernel = scipy.stats.gaussian_kde(values)(values)
                    norm = plt.Normalize(kernel.min(), kernel.max())
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    # ___________________________________________________
                    # Type 1
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.axline([0, 0], [0.01, 0.01], color = 'k', ls='--')
                    sns.scatterplot(
                        data=sub_temp_df,
                        x=stat_obs,
                        y=stat_pred,
                        c=kernel,
                        cmap=cmap,
                        ax=ax,
                        edgecolor = None,
                        legend='brief',
                    )
                    plt.text(.05, .95, 'R2: {0}'.format(np.round(calRsquared(sub_temp_df[stat_obs], sub_temp_df[stat_pred]), 2)), ha='left', va='top', transform=ax.transAxes)
                    ax.figure.colorbar(sm)
                    my_plot = plt.gcf()
                    
                    def getSubStr(char1, char2, str):
                        return str[str.find(char1)+1:str.find(char2)]
                    if '(' in str(stat_obs):
                        stat = y_label+'_'+getSubStr(' ', ')', str(stat_obs))[1:-1]
                    else:
                        stat = stat_obs
                    plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_scatter_type1_'+stat+'_'+str(name)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                    plt.show() 
                    # ___________________________________________________
                    # Type 2 
                    try:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.axline([0, 0], [0.01, 0.01], color = 'k', ls='--')
                        sns.kdeplot(
                            data=sub_temp_df,
                            x=stat_obs,
                            y=stat_pred,
                            levels=5,
                            fill=True,
                            alpha=0.6,
                            cut=2,
                            ax=ax,
                        )
                        sns.scatterplot(
                            data=sub_temp_df,
                            x=stat_obs,
                            y=stat_pred,
                            color="k",
                            edgecolor = None,
                            ax=ax,
                        )
                        plt.text(.05, .95, 'R2: {0}'.format(np.round(calRsquared(sub_temp_df[stat_obs], sub_temp_df[stat_pred]), 2)), ha='left', va='top', transform=ax.transAxes)
                        my_plot = plt.gcf()
                        plt.savefig(self.custom_name+'/img/model/'+model_name+'_'+str(self.custom_name)+'_'+self.out_features+'_scatter_type2_'+stat+'_'+str(name)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                        plt.show() 

                    except Exception as e: 
                        print("kde plot error")
                        print(e)


            print("Model Eval >> Plotting different metrics")
            for met, xl, yl, idx, cmap in tqdm(metrics):
                plotCDF(met, xl, yl)
                plotMap(met, xl, yl, idx, cmap)
            print("Model Eval >> Plotting different scatter plots")
            plotScatter(df)

            return metrics 
        # ___________________________________________________
        # predict river feature 
        if self.out_features == 'a':
            if self.val_method == "Observed":
                adcp_pred['TW_act'] = adcp_pred['TW']
            else:
                adcp_pred['TW_act'] = adcp_pred['a']*(adcp_pred['Q']**adcp_pred['b'])
            adcp_pred['TW_pred'] = adcp_pred[self.out_features+"_pred"]*(adcp_pred['Q']**adcp_pred['b'])
            # set logical boundaries based on max 
            adcp_pred.loc[adcp_pred['TW_act'] > 10e+06, 'TW_act'] = 10e+06
            adcp_pred.loc[adcp_pred['TW_pred'] > 10e+06, 'TW_pred'] = 10e+06
            adcp_pred = adcp_pred.loc[(adcp_pred['TW_act']<11613*1.5)&(adcp_pred['TW_pred']<11613*1.5)]
            # train
            # plotPerformance(np.array(adcp_pred[adcp_pred["set"]==0]['TW_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==0]['TW_pred']).reshape((-1,1)), "a_TW_Training")  
            print("parameter a TW Training Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==0]['TW_act'], adcp_pred[adcp_pred["set"]==0]['TW_pred'])*100))
            plotError(adcp_pred[adcp_pred["set"]==0], 'TW_act', 'TW_pred', "a_TW_Training")
            # valid
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==1]['TW_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==1]['TW_pred']).reshape((-1,1)), "a_TW_Validation")  
            print("parameter a TW Validation Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==1]['TW_act'], adcp_pred[adcp_pred["set"]==1]['TW_pred'])*100)) 
            # plotError(adcp_pred[adcp_pred["set"]==1], 'TW_act', 'TW_pred', "a_TW_Validation")
            # test
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==2]['TW_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==2]['TW_pred']).reshape((-1,1)), "a_TW_Testing")  
            print("parameter a TW Testing Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==2]['TW_act'], adcp_pred[adcp_pred["set"]==2]['TW_pred'])*100)) 
            metrics = plotError(adcp_pred[adcp_pred["set"]!=0], 'TW_act', 'TW_pred', "a_TW_Testing")
            pred_df = pd.DataFrame(list(zip(np.array(adcp_pred['TW_act'].values), np.array(adcp_pred['TW_pred'].values))), columns =['observed', 'predicted'])
        
        if self.out_features == 'b':
            if self.val_method == "Observed":
                adcp_pred['TW_act'] = adcp_pred['TW']
            else:
                adcp_pred['TW_act'] = adcp_pred['a']*(adcp_pred['Q']**adcp_pred['b'])
            adcp_pred['TW_pred'] = adcp_pred['a']*(adcp_pred['Q']**adcp_pred[self.out_features+"_pred"])
            # set logical boundaries based on max
            adcp_pred.loc[adcp_pred['TW_act'] > 10e+06, 'TW_act'] = 10e+06
            adcp_pred.loc[adcp_pred['TW_pred'] > 10e+06, 'TW_pred'] = 10e+06
            adcp_pred = adcp_pred.loc[(adcp_pred['TW_act']<11613*1.5)&(adcp_pred['TW_pred']<11613*1.5)]
            # train
            # plotPerformance(np.array(adcp_pred[adcp_pred["set"]==0]['TW_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==0]['TW_pred']).reshape((-1,1)), "b_TW_Training")  
            print("parameter b TW Training Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==0]['TW_act'], adcp_pred[adcp_pred["set"]==0]['TW_pred'])*100))
            plotError(adcp_pred[adcp_pred["set"]==0], 'TW_act', 'TW_pred', "b_TW_Training")
            # valid
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==1]['TW_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==1]['TW_pred']).reshape((-1,1)), "b_TW_Validation")  
            print("parameter b TW Validation Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==1]['TW_act'], adcp_pred[adcp_pred["set"]==1]['TW_pred'])*100)) 
            # plotError(adcp_pred[adcp_pred["set"]==1], 'TW_act', 'TW_pred', "b_TW_Validation")
            # test
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==2]['TW_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==2]['TW_pred']).reshape((-1,1)), "b_TW_Testing")  
            print("parameter b TW Testing Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==2]['TW_act'], adcp_pred[adcp_pred["set"]==2]['TW_pred'])*100)) 
            metrics = plotError(adcp_pred[adcp_pred["set"]!=0], 'TW_act', 'TW_pred', "b_TW_Testing")
            pred_df = pd.DataFrame(list(zip(np.array(adcp_pred['TW_act'].values), np.array(adcp_pred['TW_pred'].values))), columns =['observed', 'predicted'])

        if self.out_features == 'c':
            if self.val_method == "Observed":
                adcp_pred['Y_act'] = adcp_pred['Ymean']
            else:
                adcp_pred['Y_act'] = adcp_pred['c']*(adcp_pred['Q']**adcp_pred['f'])
            adcp_pred['Y_pred'] = adcp_pred[self.out_features+"_pred"]*(adcp_pred['Q']**adcp_pred['f'])
            # set logical boundaries based on max
            adcp_pred.loc[adcp_pred['Y_act'] > 10e+06, 'Y_act'] = 10e+06
            adcp_pred.loc[adcp_pred['Y_pred'] > 10e+06, 'Y_pred'] = 10e+06
            adcp_pred = adcp_pred.loc[(adcp_pred['Y_act']<216*1)&(adcp_pred['Y_pred']<216*1.5)]
            # train
            # plotPerformance(np.array(adcp_pred[adcp_pred["set"]==0]['Y_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==0]['Y_pred']).reshape((-1,1)), "c_Y_Training")  
            print("parameter c Y Training Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==0]['Y_act'], adcp_pred[adcp_pred["set"]==0]['Y_pred'])*100))
            plotError(adcp_pred[adcp_pred["set"]==0], 'Y_act', 'Y_pred', "c_Y_Training")
            # valid
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==1]['Y_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==1]['Y_pred']).reshape((-1,1)), "c_Y_Validation")  
            print("parameter c Y Validation Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==1]['Y_act'], adcp_pred[adcp_pred["set"]==1]['Y_pred'])*100)) 
            # plotError(adcp_pred[adcp_pred["set"]==1], 'Y_act', 'Y_pred', "c_Y_Validation")
            # test
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==2]['Y_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==2]['Y_pred']).reshape((-1,1)), "c_Y_Testing")  
            print("parameter c Y Testing Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==2]['Y_act'], adcp_pred[adcp_pred["set"]==2]['Y_pred'])*100)) 
            metrics = plotError(adcp_pred[adcp_pred["set"]!=0], 'Y_act', 'Y_pred', "c_Y_Testing")
            pred_df = pd.DataFrame(list(zip(np.array(adcp_pred['Y_act'].values), np.array(adcp_pred['Y_pred'].values))), columns =['observed', 'predicted'])

        if self.out_features == 'f':
            if self.val_method == "Observed":
                adcp_pred['Y_act'] = adcp_pred['Ymean']
            else:
                adcp_pred['Y_act'] = adcp_pred['c']*(adcp_pred['Q']**adcp_pred['f'])
            adcp_pred['Y_pred'] = adcp_pred['c']*(adcp_pred['Q']**adcp_pred[self.out_features+"_pred"])
            # set logical boundaries based on max
            adcp_pred.loc[adcp_pred['Y_act'] > 10e+06, 'Y_act'] = 10e+06
            adcp_pred.loc[adcp_pred['Y_pred'] > 10e+06, 'Y_pred'] = 10e+06
            adcp_pred = adcp_pred.loc[(adcp_pred['Y_act']<216*1)&(adcp_pred['Y_pred']<216*1.5)]
            # train
            # plotPerformance(np.array(adcp_pred[adcp_pred["set"]==0]['Y_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==0]['Y_pred']).reshape((-1,1)), "f_Y_Training")  
            print("parameter f Y Training Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==0]['Y_act'], adcp_pred[adcp_pred["set"]==0]['Y_pred'])*100))
            plotError(adcp_pred[adcp_pred["set"]==0], 'Y_act', 'Y_pred', "f_Y_Training")
            # valid
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==1]['Y_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==1]['Y_pred']).reshape((-1,1)), "f_Y_Validation")  
            print("parameter f Y Validation Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==1]['Y_act'], adcp_pred[adcp_pred["set"]==1]['Y_pred'])*100)) 
            # plotError(adcp_pred[adcp_pred["set"]==1], 'Y_act', 'Y_pred', "f_Y_Validation")
            # test
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==2]['Y_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==2]['Y_pred']).reshape((-1,1)), "f_Y_Testing")  
            print("parameter f Y Testing Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==2]['Y_act'], adcp_pred[adcp_pred["set"]==2]['Y_pred'])*100)) 
            metrics = plotError(adcp_pred[adcp_pred["set"]!=0], 'Y_act', 'Y_pred', "f_Y_Testing")
            pred_df = pd.DataFrame(list(zip(np.array(adcp_pred['Y_act'].values), np.array(adcp_pred['Y_pred'].values))), columns =['observed', 'predicted'])

        if self.out_features == 'k':
            if self.val_method == "Observed":
                adcp_pred['V_act'] = adcp_pred['V']
            else:
                adcp_pred['V_act'] = adcp_pred['k']*(adcp_pred['Q']**adcp_pred['m'])
            adcp_pred['V_pred'] = adcp_pred[self.out_features+"_pred"]*(adcp_pred['Q']**adcp_pred['m'])
            # set logical boundaries based on max
            adcp_pred.loc[adcp_pred['V_act'] > 10e+06, 'V_act'] = 10e+06
            adcp_pred.loc[adcp_pred['V_pred'] > 10e+06, 'V_pred'] = 10e+06
            adcp_pred = adcp_pred.loc[(adcp_pred['V_act']<10.2667*1)&(adcp_pred['V_pred']<10.2667*1.5)]
            # train
            # plotPerformance(np.array(adcp_pred[adcp_pred["set"]==0]['V_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==0]['V_pred']).reshape((-1,1)), "k_V_Training")  
            print("parameter k V Training Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==0]['V_act'], adcp_pred[adcp_pred["set"]==0]['V_pred'])*100))
            plotError(adcp_pred[adcp_pred["set"]==0], 'V_act', 'V_pred', "k_V_Training")
            # valid
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==1]['V_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==1]['V_pred']).reshape((-1,1)), "k_V_Validation")  
            print("parameter k V Validation Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==1]['V_act'], adcp_pred[adcp_pred["set"]==1]['V_pred'])*100)) 
            # plotError(adcp_pred[adcp_pred["set"]==1], 'V_act', 'V_pred', "k_V_Validatio")
            # test
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==2]['V_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==2]['V_pred']).reshape((-1,1)), "k_V_Testing")  
            print("parameter k V Testing Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==2]['V_act'], adcp_pred[adcp_pred["set"]==2]['V_pred'])*100)) 
            metrics = plotError(adcp_pred[adcp_pred["set"]!=0], 'V_act', 'V_pred', "k_V_Testing")
            pred_df = pd.DataFrame(list(zip(np.array(adcp_pred['V_act'].values), np.array(adcp_pred['V_pred'].values))), columns =['observed', 'predicted'])

        if self.out_features == 'm':
            if self.val_method == "Observed":
                adcp_pred['V_act'] = adcp_pred['V']
            else:
                adcp_pred['V_act'] = adcp_pred['k']*(adcp_pred['Q']**adcp_pred['m'])
            adcp_pred['V_pred'] = adcp_pred['k']*(adcp_pred['Q']**adcp_pred[self.out_features+"_pred"])
            # set logical boundaries based on max
            adcp_pred.loc[adcp_pred['V_act'] > 10e+06, 'V_act'] = 10e+06
            adcp_pred.loc[adcp_pred['V_pred'] > 10e+06, 'V_pred'] = 10e+06
            adcp_pred = adcp_pred.loc[(adcp_pred['V_act']<10.2667*1)&(adcp_pred['V_pred']<10.2667*1.5)]
            # train
            # plotPerformance(np.array(adcp_pred[adcp_pred["set"]==0]['V_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==0]['V_pred']).reshape((-1,1)), "m_V_Training")  
            print("parameter m V Training Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==0]['V_act'], adcp_pred[adcp_pred["set"]==0]['V_pred'])*100))
            plotError(adcp_pred[adcp_pred["set"]==0], 'V_act', 'V_pred', "m_V_Training")
            # valid
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==1]['V_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==1]['V_pred']).reshape((-1,1)), "m_V_Validation")  
            print("parameter m V Validation Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==1]['V_act'], adcp_pred[adcp_pred["set"]==1]['V_pred'])*100)) 
            # plotError(adcp_pred[adcp_pred["set"]==1], 'V_act', 'V_pred', "m_V_Validatio")
            # test
            plotPerformance(np.array(adcp_pred[adcp_pred["set"]==2]['V_act']).reshape((-1,1)), np.array(adcp_pred[adcp_pred["set"]==2]['V_pred']).reshape((-1,1)), "m_V_Testing")  
            print("parameter m V Testing Accuracy: %.2f%%" % (calRsquared(adcp_pred[adcp_pred["set"]==2]['V_act'], adcp_pred[adcp_pred["set"]==2]['V_pred'])*100)) 
            metrics = plotError(adcp_pred[adcp_pred["set"]!=0], 'V_act', 'V_pred', "m_V_Testing")
            pred_df = pd.DataFrame(list(zip(np.array(adcp_pred['V_act'].values), np.array(adcp_pred['V_pred'].values))), columns =['observed', 'predicted'])

        plotPerformance(self.y_train.reshape((-1, 1)), self.predictions_train.reshape((-1, 1)), self.out_features+"_Training", "normal")  
        plotPerformance(self.y_eval.reshape((-1, 1)), self.predictions_valid.reshape((-1, 1)), self.out_features+"_Validation", "normal")         

        print("Training Accuracy: %.2f%%" % (calRsquared(self.y_train_orig, self.predictions_train_orig)*100))
        print("Validation Accuracy: %.2f%%" % (calRsquared(self.y_eval_orig, self.predictions_valid_orig)*100))

        plotPerformance(self.test_y.reshape((-1, 1)), self.predictions_test.reshape((-1, 1)), self.out_features+"_Testing", "normal")  
        print("Testing Accuracy: %.2f%%" % (calRsquared(self.test_y_orig, self.predictions_test_orig)*100))
        
        return metrics, pred_df