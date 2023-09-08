# Libraries
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os
import json

# FHG dataset
# --------------------------- Read data files --------------------------- #
class DataLoader:
    """ Main body of the data loader for preparing data for ML models

    Parameters
    ----------
    data_path : str
        The path to data that is used in ML model
    target_data_path : str
        The path to target widt/depth data that is used in ML model
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
    count_thresh: int
            The desired number of observations in each station to filter out bad measurments
    Example
    --------
    >>> DataLoader(data_path = 'data/test.parquet', out_feature = 'b', rand_state = 115,
        custom_name = 'test', x_transform = False, y_transform = False, R2_thresh = 0.0, count_thresh = 3)
        
    """
    def __init__(self, data_path: str, target_data_path: str, rand_state: int, out_feature: str, 
                 custom_name: str, x_transform: bool = False, y_transform: bool = False, 
                 R2_thresh: float = 0.0, count_thresh: int = 3) -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        self.target_data_path           = target_data_path
        self.data                       = pd.DataFrame([])
        self.data_target                       = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        self.in_features                = []
        self.add_features               = []
        self.del_features               = []
        self.out_feature                = out_feature
        self.custom_name                = custom_name
        self.x_transform                = x_transform
        self.y_transform                = y_transform
        self.train                      = pd.DataFrame([])
        self.test                       = pd.DataFrame([])
        self.R2_thresh                  = R2_thresh
        self.count_thresh               = count_thresh
        
        # ___________________________________________________
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,"model/")):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,"model/"))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,"img/")):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,"img/"))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,'img/dist/')):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,'img/dist/'))

    def readFiles(self) -> None:
        """ Read files from the directories
        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')
            self.data.astype({'siteID': 'string'})
            self.data_target = pd.read_parquet(self.target_data_path, engine='pyarrow')
            self.data_target.astype({'siteID': 'string'})
        except:
            print('Wrong address or data format. Please use parquet file.')   
        
        # ___________________________________________________
        # Merge data and prepare targets
        self.data_target = self.data_target[set(self.data_target) - set(['lat','long','meas_q_va','stream_wdth_va','max_depth_va'])] # 'meas_q_va',
        self.data = pd.merge(self.data_target, self.data, on='siteID', how = 'inner')

        # ___________________________________________________
        # Filter bad stations
        target_df = pd.read_parquet(self.target_data_path, engine='pyarrow')
        target_df.astype({'siteID': 'string'})
        
        r2_epochs = np.arange(0, 1.05, 0.05)
        grouped_r2 = target_df.groupby('siteID').agg('mean')
        count_list = [len(grouped_r2)]
        for epoch in r2_epochs:
            count_Y = len(grouped_r2.loc[grouped_r2['R2']>=epoch])
            count_list.append(count_Y)

        r2_epochs = np.insert(r2_epochs, 0, -0.05, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        scale = 30
        ax.grid(True)
        ax.scatter(np.array(count_list)/len(grouped_r2), r2_epochs, c='r', s=scale, label='Y',
                    alpha=0.6, edgecolors='k')
    
        plt.vlines(x=self.R2_thresh, ymin=0, ymax=1, colors='purple', ls='--', lw=2, label='Threshold')
        ax.legend()
        ax.set_ylim([0, 1])
        plt.xlabel("R2")
        plt.ylabel("% stations greater than or equal")
        my_plot = plt.gcf()
        plt.savefig(self.custom_name+'/img/model/'+str(self.custom_name)+'_'+str(self.out_feature)+'_R2_cut.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()

        # Filter based on count 
        good_stations = grouped_r2.loc[(grouped_r2['Count'] >= self.count_thresh)]
        good_stations = good_stations.reset_index()
        good_stations.astype({'siteID': 'string'})
        stations = good_stations['siteID'].tolist()
        self.data = self.data[self.data['siteID'].isin(stations)].reset_index(drop=True)
        del good_stations, stations

        # Filter based on R2
        good_stations = grouped_r2.loc[(grouped_r2['R2'] >= self.R2_thresh)]
        good_stations = good_stations.reset_index()
        good_stations.astype({'siteID': 'string'})
        stations = good_stations['siteID'].tolist()
        del good_stations
        self.data = self.data[self.data['siteID'].isin(stations)].reset_index(drop=True)
        self.data = self.data.fillna(0) # a temporary brute force way to deal with NAN
        return 

 # --------------------------- Dimention Reduction --------------------------- #
    def reduceDim(self) -> None:
        """ Reduce the dimention of data some help addressing  multi-colinearity
        """
        print("\n Begin dimention reduction .... \n")
        # Load dimention categories
        temp = json.load(open('model_space/dimention_space.json'))
        
        # PCA model
        def buildPCA(feat_list, name):
            """ Builds a PCA and extracts new dimentions
            
            Parameters:
            ----------
            feat_list: list
                A list containing all feature names to be reduced 
            name: str    
                Reduced feature names

            Returns:
            ----------

            """
            pca = PCA(n_components = 5)
            out_arr = pca.fit_transform(self.data[feat_list])
            pickle.dump(pca, open(self.custom_name+'/model/'+'train_'+self.out_feature+'_'+name+'_PCA.pkl', "wb"))
            explained_variance = pca.explained_variance_ratio_
            components_matrix = pca.components_
            features_pc = set(self.add_features.copy())
            # Find optimum number of PCs
            for i in range(0, 5, 1):
                total_var = np.sum(explained_variance[0:i])
                self.data[str(name)+"_"+str(i)] = out_arr[:, i]
                self.add_features.append(str(name)+"_"+str(i))
                if total_var >= 0.95:
                    # num_pc = i
                    break
            
            features_pc = set(self.add_features) - features_pc
            # Save contibutions
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            ax.grid(color='gray', linewidth=0.5)
            cmap = plt.cm.seismic
            median_value = np.median(components_matrix)
            midpoint = 1 - median_value / (components_matrix.max() - components_matrix.min())
            cmap_adjusted = mcolors.TwoSlopeNorm(vmin=components_matrix.min(), vcenter=0, vmax=components_matrix.max())

            plt.imshow(components_matrix, cmap=cmap, norm=cmap_adjusted, aspect='auto')
            plt.xticks(range(len(feat_list)), feat_list, rotation=45, ha='right')
            plt.yticks(range(len(features_pc)), features_pc, rotation=45, ha='right')
            plt.colorbar(label='Loading Value')
            plt.xlabel('Original Features')
            plt.ylabel('Principal Components')
            plt.title('Contributions of Original Features to Principal Components')
            my_plot = plt.gcf()
            plt.savefig(self.custom_name+'/img/model/'+str(self.custom_name)+'_'+str(self.out_feature)+'_'+str(name)+'_PCA.png', bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()

            # Remove transformed features
            all_col = self.data.columns.tolist()
            new_col = list(set(all_col) - set(feat_list))
            self.data = self.data[new_col]

            # Update varaibles
            self.del_features += feat_list

            return

        # Vegetation
        print('Reducing Vegetation ..')
        feat_list = temp.get('Vegetation_pc')
        buildPCA(feat_list, 'Vegetation_pc')

        # Discharge
        print('Reducing Discharge ..')
        feat_list = temp.get('Discharge_pc')
        buildPCA(feat_list, 'Discharge_pc')

        # Soil_temp_moist
        print('Reducing Soil_temp_moist ..')
        feat_list = temp.get('Soil_temp_moist_pc')
        buildPCA(feat_list, 'Soil_temp_moist_pc')

        # Soil
        print('Reducing Soil ..')
        feat_list = temp.get('Soil_char_pc')
        buildPCA(feat_list, 'Soil_char_pc')

        # Preciep
        print('Reducing Preciep ..')
        feat_list = temp.get('Preciep_pc')
        buildPCA(feat_list, 'Preciep_pc')

        # Topo
        print('Reducing Topo ..')
        feat_list = temp.get('Topo_pc')
        buildPCA(feat_list, 'Topo_pc')

        print("\n ------------- End of dimention reduction ----------- \n")
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
        temp_pc = []
        temp_o = []
        if sample_type == "All":
            in_feat = 'in_features'

            temp = json.load(open('data/model_feature_names.json'))
            model_features = list(set([self.out_feature]+temp.get(in_feat))-set(self.del_features))+self.add_features+temp.get('id_features')#+temp.get('in_features_NWM')+temp.get('in_features_flow_freq')
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
            self.in_features = list(set(self.in_features) - set(temp.get('id_features')) - set([self.out_feature]))
        elif sample_type == "Sub_pca":
            temp_o = json.load(open('model_space/feature_space.json'))
            temp = temp_o.get(sample_type).get(self.out_feature+'_feats')
            temp_pc = temp_o.get(sample_type).get(self.out_feature+'_pc_feats')
            pc_vars = []
            for pc_var in temp_pc:
                matched_vars = [variable for variable in self.add_features if pc_var in variable]
                pc_vars += matched_vars

            model_features = temp + pc_vars
            self.in_features = model_features.copy()
            temp = json.load(open('data/model_feature_names.json'))
            model_features += [self.out_feature]+temp.get('id_features')
        else:
            temp_o = json.load(open('model_space/feature_space.json'))
            temp = temp_o.get(sample_type).get(self.out_feature+'_feats')

            model_features = temp.copy()
            self.in_features = model_features.copy()
            temp = json.load(open('data/model_feature_names.json'))
            model_features += [self.out_feature]+temp.get('id_features')

        del temp, temp_o, temp_pc
        
        # Apply some filtering
        if "TW_" in self.out_feature: 
            # The widest navigable section in the shipping channel of the Mississippi is Lake Pepin, where the channel is approximately 2 miles wide
            # here we consider 3 miles or 15840 ft
            self.data = self.data.loc[self.data[str(self.out_feature )] < 15840]
        else:
            # The deepest river in the U.S. is the Hudson River which reaches a maximum depth of 216 ft.
            self.data = self.data.loc[self.data[str(self.out_feature )] < 216] 
        
        df_mask = self.data[model_features]
        df_mask.to_parquet(self.custom_name+'/metrics/df_mask.parquet')
        df_mask = df_mask.fillna(0) # // to be changed (compensating for EE features in cities that can be set to 0)
        msk = np.random.rand(len(df_mask)) < 0.85
        self.train = df_mask[msk]
        self.train = self.train.reset_index(drop=True)
        self.test = df_mask[~msk]
        self.test = self.test.reset_index(drop=True)
        return
    
    # --------------------------- Plot Transformation --------------------------- #
    def plotDist(self, df_old: pd.DataFrame, df_new: pd.DataFrame, split: str) -> None:
        """
        To show changes in distribution of data after transformation is applied
        
        Parameters:
        ----------
        df_old: pd.DataFrame
            old data to be ploted
        df_new: pd.DataFrame
            transformed data to be ploted
        split: str
            train or test
        
        Parameters:
        ----------
        None

        """
        for feat in df_new.columns:
            print('\n'+feat)
            print(os.path.join(os.getcwd(),self.custom_name+"/img/dist/"+str(self.custom_name)+'_'+feat+'_'+split+'_dist.png'))
            print(os.path.isfile(os.path.join(os.getcwd(),self.custom_name+"/img/dist/"+str(self.custom_name)+'_'+feat+'_'+split+'_dist.png')))

            if os.path.isfile(os.path.join(os.getcwd(),self.custom_name+"/img/dist/"+str(self.custom_name)+'_'+feat+'_'+split+'_dist.png')):
                print(feat+' found')
                continue
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            fig.tight_layout(pad = 6)
            sns.kdeplot(data=df_new[[feat]], x=feat, color='black',  ax=axes[0])
            axes[0].set_xlim((df_new[feat].min(), df_new[feat].max()))
            ax2 = axes[0].twinx()
            sns.histplot(data=df_new[[feat]], x=feat, color='blue', discrete=True, ax=ax2).set(title='After transformation')

            sns.kdeplot(data=df_old[[feat]], x=feat, color='black', ax=axes[1])
            axes[1].set_xlim((df_old[feat].min(), df_old[feat].max()))
            ax2 = axes[1].twinx()
            sns.histplot(data=df_old[[feat]], x=feat, color='orange', discrete=True, ax=ax2).set(title='Before transformation')
            plt.savefig(self.custom_name+'/img/dist/'+str(self.custom_name)+'_'+feat+'_'+split+'_dist.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()
        return


# --------------------------- Transformation --------------------------- #

    def transformData(self, t_type: str = 'power', plot_dist: bool = False) -> tuple[pd.DataFrame,
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
        t_type: str
            t_type of transformation
            Options are:
            - ``power`` for power transformation
            - ``quant`` for quantile transformation
            - ``log`` for log transformation
        
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
        print('transforming and plotting ...')
        dump_list = ['R2', 'siteID']
        if self.x_transform:
            if t_type=='power':
                # t_x = MinMaxScaler(feature_range=(0, 1))
                t_x = PowerTransformer()
            else:
                t_x = QuantileTransformer(
                    n_quantiles=500, output_distribution="normal", 
                    random_state=self.rand_state
                )
            if t_type!='log':
                # scaler_x = StandardScaler()
                train_x = self.train[self.in_features].reset_index(drop=True)
                train_x_cp = train_x.copy()
                train_x_t = t_x.fit_transform(train_x)
                pickle.dump(t_x, open(self.custom_name+'/model/'+'train_x_'+self.out_feature+'_tansformation.pkl', "wb"))
                # train_x_pt = scaler_x.fit_transform(train_x_pt)
                train_x = pd.DataFrame(data=train_x_t,
                        columns=train_x.columns)
                if plot_dist:
                    self.plotDist(train_x_cp, train_x, 'train')
                train_id = self.train[dump_list].reset_index(drop=True)

                test_x = self.test[self.in_features].reset_index(drop=True)
                test_x_cp = test_x.copy()
                test_x_t = t_x.transform(test_x)
                # test_x_pt = scaler_x.transform(test_x_pt)
                test_x = pd.DataFrame(data=test_x_t,
                        columns=test_x.columns)
                if plot_dist:
                    self.plotDist(test_x_cp, test_x, 'test')
                test_id = self.test[dump_list].reset_index(drop=True)
            else:
                train_x = self.train[self.in_features].reset_index(drop=True)
                train_x_cp = train_x.copy()
                # Replace NA and inf
                train_x = np.log(np.abs(train_x)).fillna(0)
                train_x.replace([np.inf, -np.inf], -100, inplace=True)
                if plot_dist:
                    self.plotDist(train_x_cp, train_x, 'train')
                train_id = self.train[dump_list].reset_index(drop=True)
                test_x = self.test[self.in_features].reset_index(drop=True)
                test_x_cp = test_x.copy()
                # Replace NA and inf
                test_x = np.log(np.abs(test_x)).fillna(0)
                test_x.replace([np.inf, -np.inf], -100, inplace=True)
                if plot_dist:
                    self.plotDist(test_x_cp, test_x, 'test')
                test_id = self.test[dump_list].reset_index(drop=True)

        else:
            train_x = self.train[self.in_features].reset_index(drop=True)
            train_id = self.train[dump_list].reset_index(drop=True)
            test_x = self.test[self.in_features].reset_index(drop=True)
            test_id =  self.test[dump_list].reset_index(drop=True)

        if self.y_transform:
            if t_type=='power':
                # t_y = MinMaxScaler(feature_range=(0, 1))
                t_y = PowerTransformer()
            else:    
                t_y = QuantileTransformer(
                    n_quantiles=500, output_distribution="normal", 
                    random_state=self.rand_state
                )
            if t_type!='log':
                # scaler_y = StandardScaler()
                train_y = self.train[[self.out_feature]].reset_index(drop=True)
                train_y_cp = train_y.copy()
                train_y_t = t_y.fit_transform(train_y)
                pickle.dump(t_y, open(self.custom_name+'/model/'+'train_y_'+self.out_feature+'_tansformation.pkl', "wb"))
                # train_y_pt = scaler_x.fit_transform(train_y_pt)
                train_y = train_y_t.ravel()
                if plot_dist:
                    self.plotDist(train_y_cp, pd.DataFrame({self.out_feature: train_y}), 'train')

                test_y = self.test[[self.out_feature]].reset_index(drop=True)
                test_y_cp = test_y.copy()
                test_y_t = t_y.transform(test_y)
                # test_y_pt = scaler_y.transform(test_y_pt)
                test_y = test_y_t.ravel()
                if plot_dist:
                    self.plotDist(test_y_cp, pd.DataFrame({self.out_feature: test_y}), 'test')
            else:
                train_y = self.train[[self.out_feature]].reset_index(drop=True)
                train_y_cp = train_y.copy()
                # Replace NA and inf
                train_y = np.log(np.abs(train_y)).fillna(0)
                train_y.replace([np.inf, -np.inf], -100, inplace=True)
                if plot_dist:
                    self.plotDist(train_y_cp, train_y, 'train')
                train_y = train_y.values.ravel()

                test_y = self.test[[self.out_feature]].reset_index(drop=True)
                test_y_cp = test_y.copy()
                # Replace NA and inf
                test_y = np.log(np.abs(test_y)).fillna(0)
                test_y.replace([np.inf, -np.inf], -100, inplace=True)
                if plot_dist:
                    self.plotDist(train_y_cp, test_y, 'test')
                test_y = test_y.values.ravel()
        else:
            train_y = np.log(self.train[[self.out_feature]].reset_index(drop=True))
            train_y = train_y.values.ravel()
            test_y = np.log(self.test[[self.out_feature]].reset_index(drop=True))
            test_y = test_y.to_numpy().reshape((-1,))

        print('--------------- End of transformation ---------------')
        return train_x, train_y, train_id, test_x, test_y, test_id
