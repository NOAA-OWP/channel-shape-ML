# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer



# FHG dataset
# --------------------------- Read data files --------------------------- #
class DataVis:
    """ 
    Visualize pre and post processed data

    Paramters:
    ----------
    data_path: str
        path to the data
    plot_choice: int 
        asks for ploting choice
        Options:
        - refrence fabric
        - polaris
        - DEM and NLCD
    """
    def __init__(self, data_path: str) -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        print("Plot choices are: 1-refrence_fabric, 2-polaris, 3-DEM_lancover, 4-streamcat, 5-flow_quantiles")
        self.plot_choice                = input('Pick a feature set and input its number to show:\n') 
        self.data                       = pd.DataFrame([])
        self.plot_map                   = {1: 'refrence_fabric',
                                            2: 'polaris',
                                            3: 'DEM_lancover',
                                            4: 'streamcat',
                                            5: 'flow_quantiles'}
    
    def readFiles(self) -> None:
        """ 
        Load data
        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')
            self.data = self.data[["c","f","a","b","k","m","r",
            "lengthkm","areasqkm","arbolatesu","pathlength","totdasqkm","streamleve","streamorde",
            "slope","slopelenkm","hwnodesqkm","hwnodesqkm_dummy","roughness",
            "clay_mean_0_5", "sand_mean_0_5", "silt_mean_0_5", "soil_texture_dummy",
            "alpha_mean_0_5","bd_mean_0_5","hb_mean_0_5","ksat_mean_0_5","lambda_mean_0_5",
            "n_mean_0_5","om_mean_0_5","ph_mean_0_5","theta_r_mean_0_5","theta_s_mean_0_5",
            "USGS_Seamless_DEM_13","NLCD_encoded"]]
        except:
            print('Wrong address or data format. Please use parquet file.')   
        return
        
 # --------------------------- Transform and classify data files --------------------------- #
   
    def transformData(self) -> None:
        """ 
        Perform data transformation
        """
        pt = PowerTransformer()
        df_plot_pt = pt.fit_transform(self.data)
        df_plot_pt = pd.DataFrame(data=df_plot_pt,
                columns=self.data.columns)
        scaler = StandardScaler()
        data_std = scaler.fit_transform(df_plot_pt)
        self.data_trans_std = pd.DataFrame(data=data_std,
                columns=self.data.columns)
        return

    def classifyFeatures(self) -> None:
        """ 
        Perform data transformation
        """
        quant_f = pd.DataFrame(self.data_trans_std['f'].quantile([0.33, 0.66]))
        quant_b = pd.DataFrame(self.data_trans_std['b'].quantile([0.33, 0.66]))
        quant_m = pd.DataFrame(self.data_trans_std['m'].quantile([0.33, 0.66]))

        def classifyExponents(feat, quant):
            def byRow(row):
                if row[feat] <= quant[feat].iloc[0]:
                    return 0
                elif row[feat] > quant[feat].iloc[0] and row[feat] < quant[feat].iloc[1]:
                    return 1
                else:
                    return 2
            self.data_trans_std[feat+"_Class"] = self.data_trans_std.apply(byRow, axis=1)
        
        classifyExponents('f', quant_f)
        classifyExponents('b', quant_b)
        classifyExponents('m', quant_m)
        return
    
# -------------------------- Plot data --------------------------- #
    def visualizeData(self) -> None:
        """ 
        Visualize the data
        """
        def plotExponents(class_feat, in_features):
            g = sns.pairplot(data=self.data_trans_std[in_features], hue=class_feat)
            new_labels = ['low', 'mid', 'high']
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)
            plt.savefig('img/data_vis/dist_plot_'+str(self.plot_map.get(int(self.plot_choice)))+'_'+class_feat+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()

        if self.plot_choice == '1':
            features = ["a","b","b_Class",
            "lengthkm","areasqkm","arbolatesu","pathlength","totdasqkm","streamleve","streamorde",
            "slope","slopelenkm","hwnodesqkm","roughness"]
            plotExponents("b_Class", features)
            features = ["c","f","f_Class",
            "lengthkm","areasqkm","arbolatesu","pathlength","totdasqkm","streamleve","streamorde",
            "slope","slopelenkm","hwnodesqkm","roughness"]
            plotExponents("f_Class", features)
            features = ["k","m","m_Class",
            "lengthkm","areasqkm","arbolatesu","pathlength","totdasqkm","streamleve","streamorde",
            "slope","slopelenkm","hwnodesqkm","roughness"]
            plotExponents("m_Class", features)
        
        if self.plot_choice == '2':
            features = ["a","b","b_Class",
            "clay_mean_0_5", "sand_mean_0_5", "silt_mean_0_5", 
            "alpha_mean_0_5","bd_mean_0_5","hb_mean_0_5","ksat_mean_0_5","lambda_mean_0_5",
            "n_mean_0_5","om_mean_0_5","ph_mean_0_5","theta_r_mean_0_5","theta_s_mean_0_5"]
            plotExponents("b_Class", features)
            features = ["c","f","f_Class",
            "clay_mean_0_5", "sand_mean_0_5", "silt_mean_0_5", 
            "alpha_mean_0_5","bd_mean_0_5","hb_mean_0_5","ksat_mean_0_5","lambda_mean_0_5",
            "n_mean_0_5","om_mean_0_5","ph_mean_0_5","theta_r_mean_0_5","theta_s_mean_0_5"]
            plotExponents("f_Class", features)
            features = ["k","m","m_Class",
            "clay_mean_0_5", "sand_mean_0_5", "silt_mean_0_5", 
            "alpha_mean_0_5","bd_mean_0_5","hb_mean_0_5","ksat_mean_0_5","lambda_mean_0_5",
            "n_mean_0_5","om_mean_0_5","ph_mean_0_5","theta_r_mean_0_5","theta_s_mean_0_5"]
            plotExponents("m_Class", features)

        if self.plot_choice == '3':
            features = ["a","b","b_Class",
            "USGS_Seamless_DEM_13","NLCD_encoded"]
            plotExponents("b_Class", features)
            features = ["c","f","f_Class",
            "USGS_Seamless_DEM_13","NLCD_encoded"]
            plotExponents("f_Class", features)
            features = ["k","m","m_Class",
            "USGS_Seamless_DEM_13","NLCD_encoded"]
            plotExponents("m_Class", features)
        return


# A driver class
class RunDataVis:
    @staticmethod
    def main(args):
        # Bulid an instance of DataVis object
        data_vis = DataVis('data/merged_fhg.parquet') 

        data_vis.readFiles()
        data_vis.transformData()
        data_vis.classifyFeatures()
        data_vis.visualizeData()


if __name__ == "__main__":
    RunDataVis.main([])