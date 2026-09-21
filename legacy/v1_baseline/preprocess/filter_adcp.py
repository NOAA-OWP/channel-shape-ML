# Liberaries
import datetime
import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import scipy
import scipy.optimize as optimization

class FilterADCP:
    """     
    Filtering class to read HYDRoSWOT data and perform a filtering on it. The filtering  
    of data (``pandas dataframe``) involves droping rows containg nan i measurments of
    depth and width as well as and limitting observations to those of after 2010
    to be consistent with Sentianl data.    
    """
    def __init__(self, year: str):
        self.year = str(year)
        """ 
        Parameters
        ----------
        year : string
          
        return_lr : empty
            It saves the name and corrdiante of stations.                                                                
        """

    def filterData(self) -> pd.DataFrame:
        """ 
        Returns:
        ----------
        adcp_gdf : pd.DataFrame
            the filtered adcp data                                                               
        """
        # read ADCP and ignor timestamp
        adcp = pd.read_parquet('/mnt/d/Lynker/FEMA_HECRAS/bankfull_W_D/data/adcp.parquet', engine='pyarrow')
        print('Number of unique datums: {0}'.format(adcp['coord_datum_cd'].unique()))
        print('Visual check for timestamps: {0}\n'.format( adcp['site_visit_start_dt'].min()))

        # Remove NAN
        adcp_nona = adcp.dropna(subset=['site_no','dec_lat_va','dec_long_va','stream_wdth_va','max_depth_va'])
        adcp_nona = adcp_nona[['site_no','site_visit_start_dt',
                            'station_nm','dec_lat_va','dec_long_va',
                            'coord_datum_cd','meas_q_va','stream_wdth_va','max_depth_va']]
        print('ADCP shape after NAN removal: {0}\n'.format(adcp_nona.shape))
        
        # Filter date
        adcp_nona['date'] =  pd.to_datetime(adcp_nona['site_visit_start_dt'], format='%Y-%m-%d')
        adcp_nona['date'] = adcp_nona['date'].dt.date
        adcp_nona['date'] =  pd.to_datetime(adcp_nona['date'], format='%Y-%m-%d')
        adcp_date = adcp_nona.loc[(adcp_nona['date']>=self.year+'-01-01')]
                                #   & (adcp_nona['date']<'2011-01-01')]  
        adcp_date = adcp_date[['site_no','date',
                            'station_nm','dec_lat_va','dec_long_va',
                            'meas_q_va','stream_wdth_va','max_depth_va']]
        print('ADCP shape after filttering time: {0}'.format(adcp_date.shape))

        # Groupby 
        agg_adcp = adcp_date.groupby('site_no').max().reset_index().rename(columns={'site_no': 'siteID', 'dec_lat_va': 'lat', 'dec_long_va':'long'})
        agg_adcp = agg_adcp[['siteID','lat','long','station_nm']]
        print(agg_adcp.head())

        # Just CONUS
        adcp_gdf = gpd.GeoDataFrame(
            agg_adcp, geometry=gpd.points_from_xy(agg_adcp.long, agg_adcp.lat))
        lat_point_list = [22.02, 49.84, 47.75, 23.8, 22.02]
        lon_point_list = [-127.56, -128.79, -61.47, -69.64, -127.56]
        boundary_geom = Polygon(zip(lon_point_list, lat_point_list))
        adcp_gdf = adcp_gdf[adcp_gdf.geometry.within(boundary_geom)]
        print('Number of unique stations: {0}\n'.format(len(adcp_gdf.siteID.unique())))
        adcp_gdf = adcp_gdf[['siteID','lat','long','station_nm']]

        # Write 
        adcp_gdf.to_parquet('data/station_ids.parquet')
        print('wrote to ``station_ids.parquet``')
        return adcp_gdf

    def compFitWidth(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Compute R2 and fit 
        Parameters:
        ----------
        df : pd.DataFrame
            the filtered adcp data  

        Returns:
        ----------
        df : pd.DataFrame
            the adcp daset with fits                                                             
        """
        TW_obs = df['stream_wdth_va']
        Q = df['meas_q_va']

        def relationQTW(Q, a, b):
            TW = a * (Q ** b)
            return TW
        
        def calR2(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_pred)
            return r_value**2
        
        try:
            popt, pcov = optimization.curve_fit(f=relationQTW, xdata=Q, ydata=TW_obs, maxfev=5000)
            TW_pred = relationQTW(Q, popt[0], popt[1])
            df['R2'] = calR2(TW_obs, TW_pred)
        except:
            popt = [np.NaN, np.NaN]
            df['R2'] = np.NaN

        df['coe'] = popt[0]
        df['exp'] = popt[1]

        return df

    def findBestWidth(self, df: pd.DataFrame) -> None:
        """ 
        Filter valid width stations
        Parameters:
        ----------
        df : pd.DataFrame
            the filtered adcp data  

        Returns:
        ----------
        adcp_gdf : pd.DataFrame
            the adcp daset with fits                                                             
        """
        df = df.loc[(df['stream_wdth_va'] > 0) & (df['meas_q_va'] > 0)]
        df['Count'] = 1
        df['Count'] = df.groupby('siteID')['siteID'].transform('count')# adcp_gdf_sub = adcp_gdf.loc[adcp_gdf]
        sub_width = df.copy()
        sub_width = sub_width.loc[sub_width['Count'] >= 3]

        agg_adcp_width = sub_width.groupby('siteID').apply(self.compFitWidth).reset_index()
        agg_adcp_width = agg_adcp_width[['siteID','lat','long', 'meas_q_va','stream_wdth_va', 'Count', 'coe', 'exp', 'R2']]
        agg_adcp_width = agg_adcp_width.loc[agg_adcp_width['R2'].notna()]
        st_width = agg_adcp_width.groupby('siteID').agg('max').reset_index()
        # Save
        agg_adcp_width.to_parquet('data/width_adcp.parquet')
        st_width.to_parquet('data/width_stations.parquet')
        
        # Merge with ff
        frame_final = pd.read_parquet('data/discharge_targets.parquet', engine='pyarrow')
        st_width_merged = st_width.merge(frame_final[['siteID', 'bf_ff', 'in_ff']], how='left', left_on='siteID', right_on='siteID')
        st_width_merged = st_width_merged.dropna(subset=['bf_ff','in_ff']).reset_index(drop=True)
        st_width_merged['TW_bf'] = st_width_merged['coe'] * (st_width_merged['bf_ff']**st_width_merged['exp'])
        st_width_merged['TW_in'] = st_width_merged['coe'] * (st_width_merged['in_ff']**st_width_merged['exp'])
        
        # Remove sites that inchannel is greater thand bankfull
        st_width_merged = st_width_merged.loc[st_width_merged['TW_bf'] > st_width_merged['TW_in']].reset_index(drop=True)
        st_width_merged.to_parquet('data/width_target.parquet')
        return

    def compFitDepth(self, df: pd.DataFrame) -> pd.DataFrame:
        D_obs = df['max_depth_va']
        Q = df['meas_q_va']

        def relationQTW(Q, a, b):
            D = a * (Q ** b)
            return D
        
        def calR2(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_pred)
            return r_value**2
        
        try:
            popt, pcov = optimization.curve_fit(f=relationQTW, xdata=Q, ydata=D_obs, maxfev=5000)
            D_pred = relationQTW(Q, popt[0], popt[1])
            df['R2'] = calR2(D_obs, D_pred)
        except:
            popt = [np.NaN, np.NaN]
            df['R2'] = np.NaN
        
        df['coe'] = popt[0]
        df['exp'] = popt[1]

        return df

    def findBestDepth(self, df: pd.DataFrame) -> None:
        """ 
        Filter valid depth stations
        Parameters:
        ----------
        df : pd.DataFrame
            the filtered adcp data  

        Returns:
        ----------
        adcp_gdf : pd.DataFrame
            the adcp daset with fits                                                             
        """
        df = df.loc[(df['max_depth_va'] > 0) & (df['meas_q_va'] > 0)]
        df['Count'] = 1
        df['Count'] = df.groupby('siteID')['siteID'].transform('count')# adcp_gdf_sub = adcp_gdf.loc[adcp_gdf]
        sub_depth = df.copy()
        sub_depth = sub_depth.loc[sub_depth['Count'] >= 3]
        
        agg_adcp_depth = sub_depth.groupby('siteID').apply(self.compFitDepth).reset_index()
        agg_adcp_depth = agg_adcp_depth[['siteID','lat','long', 'meas_q_va','max_depth_va', 'Count', 'coe', 'exp', 'R2']]
        agg_adcp_depth = agg_adcp_depth.loc[agg_adcp_depth['R2'].notna()]
        st_depth = agg_adcp_depth.groupby('siteID')['max_depth_va'].agg('max').reset_index()
        # Save
        agg_adcp_depth.to_parquet('data/depth_adcp.parquet')
        st_depth.to_parquet('data/depth_stations.parquet')

        # Merge with ff
        frame_final = pd.read_parquet('data/discharge_targets.parquet', engine='pyarrow')
        st_depth = pd.read_parquet('data/depth_stations.parquet', engine='pyarrow')
        st_depth_merged = st_depth.merge(frame_final[['siteID', 'bf_ff', 'in_ff']], how='left', left_on='siteID', right_on='siteID')
        st_depth_merged = st_depth_merged.dropna(subset=['bf_ff','in_ff']).reset_index(drop=True)
        st_depth_merged['Y_bf'] = st_depth_merged['coe'] * (st_depth_merged['bf_ff']**st_depth_merged['exp'])
        st_depth_merged['Y_in'] = st_depth_merged['coe'] * (st_depth_merged['in_ff']**st_depth_merged['exp'])
        
        # Remove sites that inchannel is greater thand bankfull
        st_depth_merged = st_depth_merged.loc[st_depth_merged['Y_bf'] > st_depth_merged['Y_in']].reset_index(drop=True)
        st_depth_merged.to_parquet('data/depth_target.parquet')
        return

class RunFiltering:
    @staticmethod
    def main():
        year = str(input("Enter your filtering year date: "))
        filter_obj = FilterADCP(year)
        filtered = filter_obj.filterData()
        # Compute width
        filter_obj.findBestWidth(filtered)
        # Compute depth
        filter_obj.findBestDepth(filtered)

if __name__ == "__main__":
    RunFiltering.main()


