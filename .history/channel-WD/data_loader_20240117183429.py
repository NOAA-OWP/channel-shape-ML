# Libraries
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer
from sklearn.decomposition import PCA, KernelPCA
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
    sample_type : str
        A custom sampling method
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
    train_type : str
        A custom training method
    Example
    --------
    >>> DataLoader(data_path = 'data/test.parquet', out_feature = 'b', rand_state = 115,
        custom_name = 'test', x_transform = False, y_transform = False, R2_thresh = 0.0, count_thresh = 3,
        sample_type = 'All', train_type = 'NWIS')
        
    """
    def __init__(self, data_path: str, target_data_path: str, rand_state: int, out_feature: str, 
                 custom_name: str, sample_type: str, x_transform: bool = False, y_transform: bool = False, 
                 R2_thresh: float = 0.0, count_thresh: int = 3, train_type: str = 'NWIS') -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        self.target_data_path           = target_data_path
        self.data                       = pd.DataFrame([])
        self.data_target                = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        self.in_features                = []
        self.add_features               = []
        self.del_features               = []
        self.out_feature                = out_feature
        self.custom_name                = custom_name
        self.sample_type                = sample_type
        self.x_transform                = x_transform
        self.y_transform                = y_transform
        self.train                      = pd.DataFrame([])
        self.test                       = pd.DataFrame([])
        self.R2_thresh                  = R2_thresh
        self.count_thresh               = count_thresh
        self.train_type                 = train_type
        
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
            # self.data_target = pd.read_parquet(self.target_data_path, engine='pyarrow')
            # self.data_target.astype({'siteID': 'string'})
        except:
            print('Wrong address or data format. Please use parquet file.')   
        
        # ___________________________________________________
        # Merge data and prepare targets
        # self.data_target = self.data_target[set(self.data_target.columns.to_list()) - set(['lat','long','meas_q_va','stream_wdth_va','max_depth_va','bf_ff','in_ff'])] # 'meas_q_va'
        # self.data = pd.merge(self.data_target, self.data, on='siteID', how = 'inner')
        self.data = self.data[set(self.data.columns.to_list()) - set(['geometry','NHDFlowline','full_cats',
                                                                      'gridcode','number_unique_peaks','non_zero_years',
                                                                      'toCOMID','Hydroseq','RPUID','FromNode',
                                                                      'ToNode','VPUID','hy_cats','geometry_poly',
                                                                      'REACHCODE','sourcefc','comid','FEATUREID'])]
        # Data imputation 
        impute = "median"
        if impute == "zero":
            self.data = self.data.fillna(-1) # a temporary brute force way to deal with NAN
        if impute == "median":
            median_values_df = pd.DataFrame(self.data.median(), columns=['Median'])
            median_values_df.to_parquet(self.custom_name+'/metrics/'+'median_imput.parquet')
            self.data = self.data.reset_index(drop=True)
            # self.data = self.data.replace(-1, np.nan)
            number_to_replace = -1
            self.data = self.data.fillna(-1)
            for column_name in self.data.columns:
                if number_to_replace in self.data[column_name].values:
                    vals = pd.Series(self.data[column_name])
                    vals = vals[vals != -1]
                    median_value = vals.median()
                    self.data[column_name] = self.data[column_name].replace(number_to_replace, median_value)
        
        # Find string columns (debug)
        # string_columns = []
        # # Iterate through each column and check if it contains string values
        # for col in self.data.columns:
        #     if self.data[col].dtype == 'O':  # 'O' represents object type (strings) in Pandas
        #         string_columns.append(col)  
        # print(string_columns)
        # Data cleaning based on logical values
        if self.out_feature.startswith("Y"):
            # Hudson River, which reaches 200 feet deep at some points
            self.data = self.data.loc[(self.data[self.out_feature] <= 200) & 
                                      (self.data[self.out_feature] > 0)]
        else:
            # Mississippi River, which reaches 50000 feet width at some points
            self.data = self.data.loc[(self.data[self.out_feature] <= 50000) & 
                                      (self.data[self.out_feature] > 0)]
        # ___________________________________________________
        # Filter bad stations
        #target_df = pd.read_parquet(self.target_data_path, engine='pyarrow')
        #target_df.astype({'siteID': 'string'})
        target_df = self.data[['siteID','R2','Count']]

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
        print("Shape of data after filter: {0}".format(self.data.shape))

       
        return 

 # --------------------------- Dimention Reduction --------------------------- #
    def reduceDim(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """ Reduce the dimention of data some help addressing  multi-colinearity
        """
        print("\n Begin dimention reduction .... \n")
        # Load dimention categories
        temp = json.load(open('model_space/dimention_space.json'))
        train_data_c = train_data.copy()
        test_data_c = test_data.copy()
        # PCA model
        def buildPCA(feat_list, n_components, name):
            """ Builds a PCA and extracts new dimentions
            
            Parameters:
            ----------
            feat_list: list
                A list containing all feature names to be reduced 
            n_components: int, default=None    
                Number of components to keep. if n_components is not set all components are kept
            name: str    
                Reduced feature names

            Returns:
            ----------

            """
            nonlocal train_data, test_data

            pca = PCA(n_components = n_components)
            out_arr = pca.fit_transform(train_data[feat_list])
            test_arr = pca.transform(test_data[feat_list])
            pickle.dump(pca, open(self.custom_name+'/model/'+'train_'+self.out_feature+'_'+name+'_PCA.pkl', "wb"))

            explained_variance = pca.explained_variance_ratio_
            components_matrix = pca.components_
            features_pc = set(self.add_features.copy())
            n_components = out_arr.shape[1]

            # Find optimum number of PCs
            for i in range(0, n_components, 1):
                total_var = np.sum(explained_variance[0:i])
                train_data[str(name)+"_"+str(i)] = out_arr[:, i]
                test_data[str(name)+"_"+str(i)] = test_arr[:, i]
                self.add_features.append(str(name)+"_"+str(i))
                if total_var >= 0.95:
                    # num_pc = i
                    break

            features_pc = set(self.add_features) - features_pc
            # Filter to important ones
            components_matrix = components_matrix[:i+1, :]
            # Save contibutions
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            ax.grid(color='gray', linewidth=0.5)
            cmap = plt.cm.bwr
            median_value = np.median(components_matrix)
            midpoint = 1 - median_value / (components_matrix.max() - components_matrix.min())
            cmap_adjusted = mcolors.TwoSlopeNorm(vmin=components_matrix.min(), vcenter=0, vmax=components_matrix.max())
            plt.pcolor(components_matrix, cmap=cmap, norm=cmap_adjusted, edgecolors='k', linewidths=2)
            # plt.imshow(components_matrix, cmap=cmap, norm=cmap_adjusted, aspect='auto')
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
            all_col = train_data.columns.tolist()
            new_col = list(set(all_col) - set(feat_list))
            
            # look for when dummy drops    
            if "scat_dummy" in feat_list:
                new_col.append('scat_dummy')
            elif "nwm_dummy" in feat_list:
                new_col.append('nwm_dummy')
            elif "vaa_dummy" in feat_list:
                new_col.append('vaa_dummy')
            train_data = train_data[new_col]
            test_data = test_data[new_col]

            # Update varaibles
            self.del_features += feat_list

            return
        if self.sample_type == "Sub_pca":
            # Flood
            print('Reducing Flood ..')
            feat_list = temp.get('Flood_freq_pc')
            buildPCA(feat_list, 5,'Flood_freq_pc')

            # Land_cover
            print('Reducing Land cover ..')
            feat_list = temp.get('Land_cover_pc')
            buildPCA(feat_list, 5,'Land_cover_pc')

            # Lithology
            print('Reducing Lithology ..')
            feat_list = temp.get('Lithology_pc')
            buildPCA(feat_list, 5,'Lithology_pc')

            # Soil
            print('Reducing Soil ..')
            feat_list = temp.get('Soil_pc')
            buildPCA(feat_list, 5,'Soil_pc')

            # Human
            print('Reducing Human ..')
            feat_list = temp.get('Human_pc')
            buildPCA(feat_list, 5,'Human_pc')

            # Dam
            print('Reducing Dam ..')
            feat_list = temp.get('Dam_pc')
            buildPCA(feat_list, 3,'Dam_pc')

            # Topo
            print('Reducing Topo ..')
            feat_list = temp.get('Topo_pc')
            buildPCA(feat_list, 5,'Topo_pc')

            # Watershed
            print('Reducing Watershed ..')
            feat_list = temp.get('Watershed_pc')
            buildPCA(feat_list, 2,'Watershed_pc')

            # Stream
            print('Reducing Stream ..')
            feat_list = temp.get('Stream_pc')
            buildPCA(feat_list, 2,'Stream_pc')
        
        if self.sample_type == "All_pca":
            temp = json.load(open('model_space/feature_space.json'))
            feat_list = temp.get('All').get(self.out_feature+'_feats')
            buildPCA(feat_list, None,'PC')

        # Lookup needed PCs
        pc_columns = [col for col in train_data.columns if '_pc' in col]
        # non_pc_columns = set(train_data.columns) -set(pc_columns)
        temp_o = json.load(open('model_space/feature_space.json'))
        temp_pc = temp_o.get(self.sample_type).get(self.out_feature+'_pc_feats')
        temp = temp_o.get(self.sample_type).get(self.out_feature+'_feats')
        pc_vars = []
        for pc_var in temp_pc:
            matched_vars = [variable for variable in train_data.columns if pc_var in variable]
            pc_vars += matched_vars

        model_features = list(temp) + pc_vars
        # Drop NWM features as input for NWM training only
        if self.train_type == "NWM":
            model_features = set(model_features) - set(["NWM_2","NWM_1.5"])
        train_data = train_data[model_features]
        train_data = train_data[model_features]
        test_data = test_data[model_features]
        
        train_data_c = train_data_c[self.del_features]
        test_data_c = test_data_c[self.del_features]
        train_data_complete = pd.concat([train_data, train_data_c], axis=1)
        test_data_complete = pd.concat([test_data, test_data_c], axis=1)

        train_data = train_data.reset_index(drop=True)
        test_data_c = test_data_c.reset_index(drop=True)
        train_data_complete = train_data_complete.reset_index(drop=True)
        test_data_complete = test_data_complete.reset_index(drop=True)
        print("\n ------------- End of dimention reduction ----------- \n")
        return train_data, test_data, train_data_complete, test_data_complete
        
 # --------------------------- Split train and test --------------------------- #
    
    def splitData(self) -> None:
        """ 
        To split data to train and test, and whether to use all 
        features or few 

        Parameters:
        ----------

        Example
            --------
            >>> splitData("All")
        """
        temp_pc = []
        temp_o = []
        if self.sample_type == "All":
            in_feat = 'in_features'

            temp = json.load(open('data/model_feature_names.json'))
            model_features = list(set([self.out_feature]+temp.get(in_feat)+temp.get('in_NWM')+temp.get('in_flow_freq')+temp.get('in_scat'))
                                  -set(self.del_features))+self.add_features+temp.get('id_features')
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
        
        elif self.sample_type == "Sub_pca":
            temp_o = json.load(open('model_space/feature_space.json'))
            temp = temp_o.get("All").get(self.out_feature+'_feats')
            # temp_pc = temp_o.get(self.sample_type).get(self.out_feature+'_pc_feats')
            # pc_vars = []
            # for pc_var in temp_pc:
            #     matched_vars = [variable for variable in self.add_features if pc_var in variable]
            #     pc_vars += matched_vars

            #model_features = temp + pc_vars
            model_features = temp 
            self.in_features = model_features.copy()
            temp = json.load(open('data/model_feature_names.json'))
            model_features += [self.out_feature]+temp.get('id_features')

        elif self.sample_type == "All_pca":
            temp = json.load(open('data/model_feature_names.json'))
            model_features = [self.out_feature]+self.add_features+temp.get('id_features')#+temp.get('in_features_NWM')+temp.get('in_features_flow_freq')
            self.in_features = self.add_features.copy()

        else:
            temp_o = json.load(open('model_space/feature_space.json'))
            temp = temp_o.get(self.sample_type).get(self.out_feature+'_feats')

            model_features = temp.copy()
            self.in_features = model_features.copy()
            temp = json.load(open('data/model_feature_names.json'))
            model_features += [self.out_feature]+temp.get('id_features')

        del temp, temp_o, temp_pc
        
        # Apply some filtering
        if "TW_" in self.out_feature: 
            # The widest navigable section in the shipping channel of the Mississippi is Lake Pepin, where the channel is approximately 2 miles wide
            # here we consider 3 miles or 15840 ft
            self.data = self.data.loc[self.data[str(self.out_feature)] < 15840]
        else:
            # The deepest river in the U.S. is the Hudson River which reaches a maximum depth of 216 ft.
            self.data = self.data.loc[self.data[str(self.out_feature)] < 216] 
       
       # Drop NWM features as input for NWM training only
        if self.train_type == "NWM":
            self.data = self.data.drop(columns=["NWM_2","NWM_1.5"])
            model_features = set(model_features) - set(["NWM_2","NWM_1.5"])
            self.in_features = set(self.in_features) - set(["NWM_2","NWM_1.5"])
        
        df_mask = self.data[model_features]
        duplicated_columns = df_mask.columns[df_mask.columns.duplicated()]
        print('dupies')
        print(duplicated_columns)
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
    def transformData(self, t_type: str = 'power', sub_trans: bool = True, plot_dist: bool = False) -> tuple[pd.DataFrame,
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
        trans_feats = []
        if sub_trans:
            temp = json.load(open('model_space/dimention_space.json'))
            pca_feats = [string for key in temp for string in temp[key]]
            trans_feats = pca_feats.copy()
        in_feats = set(self.in_features) - set(trans_feats)
        if self.x_transform:
            trans_feats = self.in_features.copy()

        # Lets keep record on input varibales and order
        def preserve_order(item):
            return {"value": item}
        # Convert the list elements into a JSON-serializable format with ordered keys
        serialized_list = [preserve_order(item) for item in trans_feats]
        # Save the list to a JSON file
        with open('model_space/trans_feats'+'_'+self.out_feature+"_"+'.json', 'w') as json_file:
            json.dump(serialized_list, json_file, indent=4)
        min_value = 0
        max_value = 500
        scaler = MinMaxScaler(feature_range=(min_value, max_value))
        if t_type=='power':
            # t_x = MinMaxScaler(feature_range=(0, 1))
            t_x = PowerTransformer()
        if t_type=='quant':
            t_x = QuantileTransformer(
                n_quantiles=500, output_distribution="normal", 
                random_state=self.rand_state
            )
        if t_type!='log':
            # scaler_x = StandardScaler()
            train_x = self.train[trans_feats].reset_index(drop=True)
            train_x = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns)
            train_x_cp = train_x.copy()
            train_x_t = t_x.fit_transform(train_x)
            pickle.dump(t_x, open(self.custom_name+'/model/'+'train_x_'+self.out_feature+'_tansformation.pkl', "wb"))
            pickle.dump(scaler, open(self.custom_name+'/model/'+'train_x_'+self.out_feature+'_scaler_tansformation.pkl', "wb"))

            train_x = pd.DataFrame(data=train_x_t,
                    columns=train_x.columns)
            if plot_dist:
                self.plotDist(train_x_cp, train_x, 'train')
            train_id = self.train[dump_list].reset_index(drop=True)

            test_x = self.test[trans_feats].reset_index(drop=True)
            # test_x.to_parquet('data/tttt.parquet')
            test_x = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns)
            test_x_cp = test_x.copy()
            test_x_t = t_x.transform(test_x)
            test_x = pd.DataFrame(data=test_x_t,
                    columns=test_x.columns)
            if plot_dist:
                self.plotDist(test_x_cp, test_x, 'test')
            test_id = self.test[dump_list].reset_index(drop=True)
        else:
            train_x = self.train[trans_feats].reset_index(drop=True)
            train_x_cp = train_x.copy()
            # Replace NA and inf
            train_x = np.log(np.abs(train_x)).fillna(0)
            train_x.replace([np.inf, -np.inf], -100, inplace=True)
            if plot_dist:
                self.plotDist(train_x_cp, train_x, 'train')
            train_id = self.train[dump_list].reset_index(drop=True)
            test_x = self.test[trans_feats].reset_index(drop=True)
            test_x_cp = test_x.copy()
            # Replace NA and inf
            test_x = np.log(np.abs(test_x)).fillna(0)
            test_x.replace([np.inf, -np.inf], -100, inplace=True)
            if plot_dist:
                self.plotDist(test_x_cp, test_x, 'test')
            test_id = self.test[dump_list].reset_index(drop=True)
        
        if not self.x_transform:
            train_x = pd.concat([train_x, self.train[in_feats]], axis=1)
            test_x = pd.concat([test_x, self.test[in_feats]], axis=1)
        
        # duplicated_columns = train_x.columns[train_x.columns.duplicated()]
        # print(duplicated_columns)
        if self.y_transform:
            if t_type=='power':
                # t_y = MinMaxScaler(feature_range=(0, 1))
                t_y = PowerTransformer()
            if t_type=='quant':  
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
            train_y = self.train[[self.out_feature]].reset_index(drop=True)
            train_y = train_y.values.ravel()
            test_y = self.test[[self.out_feature]].reset_index(drop=True)
            test_y = test_y.to_numpy().reshape((-1,))

        print('--------------- End of transformation ---------------')
        
        # Test data
        is_inf_train_x = train_x.isin([np.inf, -np.inf]).any().any()
        if is_inf_train_x:
            print('---- found inf in train x !!!!' )
        is_inf_test_x = test_x.isin([np.inf, -np.inf]).any().any()
        if is_inf_test_x:
            print('---- found inf in test x !!!!' )
        is_inf_train_y = np.isinf(train_y).any()
        if is_inf_train_y:
            print('---- found inf in train y !!!!' )
        is_inf_test_y = np.isinf(test_y).any()
        if is_inf_test_y:
            print('---- found inf in test y !!!!' )
        has_missing_train_x = train_x.isna().any().any()
        if has_missing_train_x:
            print('---- found nan in train x !!!!' )
        has_missing_test_x= test_x.isna().any().any()
        if has_missing_test_x:
            print('---- found nan in test x !!!!' )
        has_missing_train_y = np.isnan(train_y).any()
        if has_missing_train_y:
            print('---- found nan in train y !!!!' )
        has_missing_test_y = np.isnan(test_y).any()
        if has_missing_test_y:
            print('---- found nan in test y !!!!' )

        train_x.to_parquet(self.custom_name+'/metrics/'+'train_x'+'_'+self.out_feature+'.parquet')
        pd.DataFrame({self.out_feature: train_y}).to_parquet(self.custom_name+'/metrics/'+'train_y'+'_'+self.out_feature+'.parquet')
        train_id.to_parquet(self.custom_name+'/metrics/'+'train_id'+'_'+self.out_feature+'.parquet')
        test_x.to_parquet(self.custom_name+'/metrics/'+'test_x'+'_'+self.out_feature+'.parquet')
        pd.DataFrame({self.out_feature: test_y}).to_parquet(self.custom_name+'/metrics/'+'test_y'+'_'+self.out_feature+'.parquet')
        test_id.to_parquet(self.custom_name+'/metrics/'+'test_id'+'_'+self.out_feature+'.parquet')    
        train_id = train_id.reset_index(drop=True)
        test_id = test_id.reset_index(drop=True)

        return train_x, train_y, train_id, test_x, test_y, test_id
