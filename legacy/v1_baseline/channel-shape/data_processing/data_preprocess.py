# Libraries
import wget
import shutil
import pandas as pd
import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import preprocessing
from typing import Tuple


# FHG dataset
# --------------------------- Read data files --------------------------- #
class DataPreProcess:
    def __init__(self) -> None:
        """ 
        Data preprocessing and cleaning
        """
        pd.options.display.max_columns = 60
    
    def  readFiles(self) -> Tuple[pd.DataFrame, pd.DataFrame,
                                pd.DataFrame, pd.DataFrame,
                                pd.DataFrame]:
        """ 
        Load data for processing

        Returns:
        ----------
            Dataframes of diffrent datasets
        """
        fhg = pd.read_parquet('data/training_data.parquet', engine='pyarrow')
        polaris = pd.read_parquet('data/polaris_data.parquet', engine='pyarrow')
        dem = pd.read_parquet('data/3dep_13_data.parquet', engine='pyarrow')
        lc = pd.read_parquet('data/nlcd_2019_data.parquet', engine='pyarrow')
        vaa = pd.read_parquet('data/vaa.parquet', engine='pyarrow')
        streamcat = pd.read_parquet('data/cleaned_streamcat.parquet', engine='pyarrow')
        flow = pd.read_parquet('data/cleaned_summary_flows_AHG.parquet', engine='pyarrow')

        df_list = [polaris, dem, lc]
        for df in df_list:
            df.pop('ID')
            df.pop('siteID')

        print('number of stations {0}\n'.format(dem.comid.unique().size))   
        return fhg, polaris, dem, lc, vaa
# -------------------------- Check some stats --------------------------- #
    def showStats(self, name: str, dataset: pd.DataFrame) -> None:
        """ 
        Show statistical stats of dataset

        Paramters:
        ----------
        name: str
            custom name
        dataset: pd.DataFrame
            input data
                
        """
        print('{0} stats'.format(name))
        print(dataset.describe())
        print('\n')
        return

    

# -------------------------- Look for NaNs --------------------------- #
    def findNan(self, name: str, dataset: pd.DataFrame) -> None:
        """ 
        Check if dataset contains NAN

        Paramters:
        ----------
        name: str
            custom name
        dataset: pd.DataFrame
            input data
                
        """
        print('{0} NaN'.format(name))
        print(dataset.isna().sum())
        print('\n')
        return


    def Process(self, fhg: pd.DataFrame, polaris: pd.DataFrame,
                dem: pd.DataFrame, lc: pd.DataFrame,
                vaa: pd.DataFrame) -> None:
        """ 
        Performing data processing and cleaning

        Paramters:
        ----------
            All data in dataframe format
        """        
        lc = lc.astype({'comid': 'int32'})
        dem = dem.astype({'comid': 'int32'})
        fhg = fhg.astype({'comid': 'int32'})
        vaa = vaa.astype({'comid': 'int32'})
        polaris = polaris.astype({'comid': 'int32'})

        # ___________________________________________________
        # create geo dataframe
        fhg_gdf = gpd.GeoDataFrame(
            fhg, geometry=gpd.points_from_xy(fhg.lng, fhg.lat))

        # ___________________________________________________
        # get state shapes
        path = Path('maps/cb_2018_us_state_20m.shp')
        if not path.is_file():
            wget.download("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
            shutil.unpack_archive('cb_2018_us_state_20m.zip', '/maps')

        states = gpd.read_file('maps/cb_2018_us_state_20m.shp')
        states = states.to_crs("EPSG:4326")
        print(states.crs)
        print('\n')

        st_list = list(states["STUSPS"])
        for i in range(len(st_list)):
            fhg_gdf["STUSPS_"+st_list[i]] = fhg_gdf["geometry"].within(states["geometry"][i]).astype("int")

        def getStates(row):
            for c in fhg_gdf.iloc[:, 24:].columns:
                if row[c]==1:
                    return c[7:]

        fhg_gdf["State"] = fhg_gdf.iloc[:, 24:].apply(getStates, axis=1)

        cols = [c for c in fhg_gdf.columns if c[:6] != 'STUSPS']
        fhg_gdf = fhg_gdf[cols]

        # -------------------------- Merged dataset --------------------------- #
        fhg_gdf_merged = fhg_gdf.merge(vaa, on='comid', how='left')
        
        # ___________________________________________________
        # Deal with NaNs
        fhg_gdf_merged['hwnodesqkm_dummy'] = np.where(fhg_gdf_merged['hwnodesqkm'].isna(),0,1) 
        fhg_gdf_merged['hwnodesqkm'] = fhg_gdf_merged['hwnodesqkm'].replace(np.nan, 0)

        # ___________________________________________________
        # Produce hydrological unit
        fhg_gdf_merged["hydrological_unit"] = fhg_gdf_merged["reachcode"].str[:8]   
        fhg_gdf_merged["hydrological_unit"] = fhg_gdf_merged["hydrological_unit"].astype(int) 

        # ___________________________________________________
        #Number of unique hydrological units
        print('Number of unique hydrological units: {0}\n'.format(len(fhg_gdf_merged["hydrological_unit"].unique())))

        fhg_gdf_merged_copy = fhg_gdf_merged.copy()
        fhg_gdf_merged_copy["ones"] = 1
        temp = fhg_gdf_merged_copy.groupby(['hydrological_unit'])["ones"].agg(['sum']).reset_index()
        temp = temp.rename(columns={"sum": "num_stations_HUC"})
        fhg_gdf_merged = fhg_gdf_merged.merge(temp, on='hydrological_unit', how='left')

        del temp, fhg_gdf_merged_copy

        # ___________________________________________________
        # Merge with Polaris
        temp = fhg_gdf_merged.copy()
        fhg_gdf_merged = temp.merge(polaris, on='comid', how='left')
        fhg_gdf_merged['soil_texture_dummy'] = np.where(fhg_gdf_merged['clay_mean_0_5'].isna(),0,1) 

        fhg_gdf_merged = fhg_gdf_merged.loc[:, (fhg_gdf_merged.columns != 'State') & (fhg_gdf_merged.columns != 'gnis_id')]
        fhg_gdf_merged = fhg_gdf_merged.replace(np.nan, 0)
        fhg_gdf_merged = fhg_gdf_merged.merge(temp[['State','gnis_id','comid']], on='comid', how='left')

        del temp

        # ___________________________________________________
        # DEM and land cover
        fhg_gdf_merged = fhg_gdf_merged.merge(dem, on='comid', how='left')
        fhg_gdf_merged = fhg_gdf_merged.merge(lc, on='comid', how='left')
        print('NaN values in merged dataset')
        print(fhg_gdf_merged.isna().sum())
        print('\n')

        # ___________________________________________________
        # encode lancover
        le = preprocessing.LabelEncoder()
        fhg_gdf_merged['NLCD_encoded'] = le.fit_transform(fhg_gdf_merged['NLCD Land Cover Class'])
        print(fhg_gdf_merged.head())

        # ___________________________________________________
        # save
        fhg_gdf_merged.to_parquet('data/merged_fhg.parquet')
        return

# A driver class
class RunPP:
    @staticmethod
    def main(args):
        # ___________________________________________________
        # Bulid an instance of DataPreProcess object
        data_pp = DataPreProcess() 

        fhg, polaris, dem, lc, vaa = data_pp.readFiles()

        # ___________________________________________________
        # some prints
        data_pp.showStats('fhg', fhg)
        data_pp.showStats('vaa', vaa)
        data_pp.showStats('polaris', polaris)
        data_pp.showStats('dem', dem)
        data_pp.showStats('land cover', lc)

        data_pp.findNan('fhg', fhg)
        data_pp.findNan('vaa', vaa)
        data_pp.findNan('polaris', polaris)
        data_pp.findNan('dem', dem)
        data_pp.findNan('land cover', lc)

        # ___________________________________________________
        # Process data 
        data_pp.Process(fhg, polaris, dem, lc, vaa)

if __name__ == "__main__":
    RunPP.main([])