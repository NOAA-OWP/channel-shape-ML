# Libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
import shap
import pickle

# Feature Importance
# --------------------------- Read data files --------------------------- #
class FeatureImportance:
    """ 
    A class object to plot feature importances
        
    Parameters: 
    ----------
    custom_name: str
        A custom name represnting the running code instance
    best_model: str
        The name of the best model identfied by k-fold cross validation
    """
    def __init__(self, custom_name: str, best_model: str):
        pd.options.display.max_columns  = 30
        self.custom_name                = custom_name
        self.best_model                 = best_model
        # ___________________________________________________
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,"img/feature_importance/")):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,"img/feature_importance/"))
    
    def plotImportance(self, model: any, out_features: str, 
                       train_x: pd.DataFrame, train_y: pd.DataFrame) -> None:
        """ 
        Plot the feature importances
            
        Parameters: 
        ----------
        model: any
            ML structure and weights
        out_features: str
            Name of the FHG coeficients
        train_x : pd.DataFrame
            splited predictor data for training
        train_y : np.array
            splited target data for training

        Example:
        ----------
        >>>  plotImportance(model, 'b', train_x, train_y)
        """
        # ___________________________________________________
        # model feature importance
        if self.best_model == 'xgb':
            print("Plotting XGB default feature importance for {0} -------------- ".format(str(out_features)))
            xgb.plot_importance(model, grid=False)
            plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            pyplot.show()
            plt.show()

        # ___________________________________________________
        # Permutation feature importance
        print("Plotting Permutation feature importance for {0} -------------- ".format(str(out_features)))
        results = permutation_importance(model, train_x, train_y, scoring='neg_mean_absolute_error')
        importance = results.importances_mean
        col_list = []
        for col in train_x.columns:
            col_list.append(col)
        col_list = np.array(col_list)
        col_list
        feat_permut = pd.DataFrame.from_dict({"features":col_list, "p_score": importance}).sort_values(by=['p_score'])

        fig, ax = plt.subplots(figsize=(30,25))
        fig.tight_layout(pad=5)

        # ___________________________________________________
        # Creating a case-specific function to avoid code repetition
        def plot_feat(x: str, y: str, xlabel: str, ylabel: str, rotation: int,
                            tick_bottom: bool, tick_left: bool) -> None:
            """ 
            A simple bar plot
                
            Parameters: 
            ----------
            x: str
                x axis title
            y: str
                y axis title
            xlabel: str
                Label for x axis
            ylabel: str
                Label for y axis
            rotation: int
                rotation of labels in degrees
            tick_bottom: bool
                To have ticks on x or not
                options:
                - True
                - False
            tick_left: bool
            To have ticks on y or not
                options:
                - True
                - False

            Example:
            ----------
            >>>  plot_feat(x='p_score', y='features',
                    xlabel='Importance', ylabel=None,
                    rotation=None, tick_bottom=True, tick_left=False)
            """

            sns.barplot(x=x, y=y, data=feat_permut, color='slateblue')
            plt.title('Feature importance \\Permutation', fontsize=85)
            plt.xlabel(xlabel, fontsize=60)
            plt.xticks(fontsize=45, rotation=rotation)
            plt.ylabel(ylabel, fontsize=60)
            plt.yticks(fontsize=45)
            sns.despine(bottom=False, left=True)
            plt.grid(False)
            plt.tick_params(bottom=tick_bottom, left=tick_left)
            return None

        plot_feat(x='p_score', y='features',
                    xlabel='Importance', ylabel=None,
                    rotation=None, tick_bottom=True, tick_left=False)
        plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_Permute.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()
        return

    def plotShapImportance(self, model: any, out_features: str, 
                           train_x: pd.DataFrame) -> None:
        """ 
            A shap importance plots

            Parameters: 
            ----------
            model: any
                ML structure and weights
            out_features: str
                Name of the FHG coeficients
            train_x: pd.DataFrame
                splited predictor data for training

            Example:
            ----------
            >>>  plotShapImportance(model, 'b', train_x)
            """
        print("Plotting Shap feature importance {0} performance -------------- ".format(str(out_features)))
        if self.best_model in ['xgb']:
            # best_explainer = shap.Explainer(model, train_x) # Save f names
            # pickle.dump(best_explainer, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_Shap_Best.pickle.dat", "wb")) # Save f names
            # best_shap_values = best_explainer(train_x) # Save f names

            tree_explainer = shap.TreeExplainer(model)
            pickle.dump(tree_explainer, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_Shap_Tree.pickle.dat", "wb"))
            tree_shap_values = tree_explainer.shap_values(train_x, check_additivity=False )
            tree_expected_value = tree_explainer.expected_value
            if isinstance(tree_expected_value, list):
                tree_expected_value = tree_expected_value[1]
            #     agg_shap_values = np.abs(tree_shap_values).mean(0) # Save f names
            # else: # Save f names
            #     agg_shap_values = np.abs(best_shap_values.values).mean(0) # Save f names
            # feature_importance = pd.DataFrame(list(zip(train_x.columns, agg_shap_values)), columns=['feature_name','feature_importance']) # Save f names
            # feature_importance.sort_values(by=['feature_importance'], ascending=False, inplace=True) # Save f names
            # 
            # shap_data = {'columns': train_x.columns.to_list(),
            #              'shap_names': best_shap_values.feature_names,
            #              'shap_values': best_shap_values.data.mean(0)
            #              }
            # shap_data = pd.DataFrame(shap_data)

            best_explainer = shap.Explainer(model, train_x) # del Save f names
            pickle.dump(best_explainer, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_Shap_Best.pickle.dat", "wb")) # del Save f names
            best_shap_values = best_explainer(train_x) # del Save f names

            # best_expected_value = best_explainer.expected_value
            # if isinstance(best_expected_value, list):
            #     best_expected_value = best_expected_value[1]
        else:
            best_explainer = shap.explainers.Permutation(model.predict, train_x)
            pickle.dump(best_explainer, open(self.custom_name+"/model/"+str(self.custom_name)+'_'+out_features+"_Shap_Best.pickle.dat", "wb"))
            best_shap_values = best_explainer(train_x) #test [:10]
            
            # agg_shap_values = np.abs(best_shap_values.values).mean(0) # Save f names
            # feature_importance = pd.DataFrame(list(zip(train_x.columns, agg_shap_values)), columns=['feature_name','feature_importance']) # Save f names
            # feature_importance.sort_values(by=['feature_importance'], ascending=False, inplace=True) # Save f names
            # shap_data = {'columns': train_x.columns.to_list(),
            #              'shap_names': best_shap_values.feature_names,
            #              'shap_values': best_shap_values.data.mean(0)
            #              }
            # shap_data = pd.DataFrame(shap_data)

            # best_expected_value = best_explainer.expected_value
            # if isinstance(best_expected_value, list):
            #     best_expected_value = best_expected_value[1]
        # feature_importance.to_parquet(self.custom_name+'/metrics/shap_'+str(self.custom_name)+'_'+out_features+'.parquet') # Save f names

        # ___________________________________________________
        # Heat maps
        plt.clf()
        plt.figure()
        shap.plots.heatmap(best_shap_values, max_display=13, show=False)
        plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_Heatmap_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()
        plt.clf()

        # ___________________________________________________
        # plot the summery
        if self.best_model in ['xgb']:
            plt.figure()
            shap.summary_plot(tree_shap_values, features=train_x, feature_names=train_x.columns, max_display=60, show=False)
            my_plot = plt.gcf()
            plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_SHAP_Summary_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()
            plt.clf()

            plt.figure()
            shap.summary_plot(tree_shap_values, features=train_x, feature_names=train_x.columns, plot_type='bar', max_display=60, show=False)
            my_plot = plt.gcf()
            plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_SHAP_Bar_Summary_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()
            plt.clf()
        else:
            plt.figure()
            shap.summary_plot(best_shap_values, features=train_x, feature_names=train_x.columns, max_display=60, show=False)  #test [:10]
            my_plot = plt.gcf()
            plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_SHAP_Summary_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()
            plt.clf()

            plt.figure()
            shap.summary_plot(best_shap_values, features=train_x, feature_names=train_x.columns, plot_type='bar', max_display=50, show=False) #test [:10]
            my_plot = plt.gcf()
            plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_SHAP_Bar_Summary_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()
            plt.clf()

        # ___________________________________________________
        # Barplot
        plt.figure()
        shap.plots.bar(best_shap_values.abs.mean(0), max_display=60, show=False)
        #shap.plots.bar(np.abs(best_shap_values.values).mean(0), max_display=60, show=False) # Save f names
        my_plot = plt.gcf()
        plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_Barplot_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()
        plt.clf()

        # ___________________________________________________
        # look at 5 important features
        if self.best_model in ['xgb']:
            imps = train_x.columns[np.argsort(np.abs(tree_shap_values).mean(0))]
            imps = list(imps[-6:-1])
            for feature in imps:
                shap.plots.scatter(best_shap_values[:,feature], show=False)
                my_plot = plt.gcf()
                plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_'+feature+'_Scatter_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                plt.show()
                
                shap.dependence_plot(feature, tree_shap_values, train_x, show=False)
                my_plot = plt.gcf()
                plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_'+feature+'_Scatter_Dependance_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                plt.show() 
        else:
            imps = train_x.columns[np.argsort(np.abs(best_shap_values.values).mean(0))]
            imps = list(imps[-6:-1])
            for feature in imps:
                shap.plots.scatter(best_shap_values[:,feature], show=False)
                my_plot = plt.gcf()
                plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_'+feature+'_Scatter_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                plt.show()
                
                shap.dependence_plot(feature, best_shap_values.values, train_x, show=False) # [:10]
                my_plot = plt.gcf()
                plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_'+feature+'_Scatter_Dependance_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
                plt.show()

        # ___________________________________________________    
        # plot a single instance of observation
        plt.clf()
        obs = 8
        plt.figure()
        a = best_shap_values[obs]
        shap.plots.waterfall(best_shap_values[obs], max_display=60, show=False)  
        my_plot = plt.gcf()
        plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_Obs_num_'+str(obs)+'_Waterfall_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()
        plt.clf()

        # ___________________________________________________
        # plot first 500 varaibles interactions
        if self.best_model in ['xgb']:
            plt.figure()
            shap_interaction_values = tree_explainer.shap_interaction_values(train_x.iloc[:100,:])
            shap.summary_plot(shap_interaction_values, train_x.iloc[:100,:], max_display = 12, show=False)
            my_plot = plt.gcf()
            plt.savefig(self.custom_name+'/img/feature_importance/'+str(self.custom_name)+'_'+out_features+'_Summary_Interactions_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()
            plt.clf()
       
    
        # a custom interaction plot 
        # plt.figure()
        # shap.dependence_plot(
        #     ("slope", "soil_texture_dummy"),
        #     shap_interaction_values, train_x.iloc[:100,:],
        #     display_features=train_x.iloc[:100,:]
        # )
        # my_plot = plt.gcf()
        # plt.savefig('img/feature_importance/'+str(custom_name)+'_'+out_features+'_Custom_Interactions_'+str(self.best_model)+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
        # plt.show()
        # plt.clf()
        print("All plotting complete for {0} -------------- ".format(str(out_features)))