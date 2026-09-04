# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

# --------------------------- Plot fits and metrics --------------------------- #
class PlotComp:
    """ An onject to plot ML model comparisons
    Parameters
    ----------
    b_metric : pd.DataFrame
        Dataframe contining accurecies of best model
    v_metric : pd.DataFrame
        Dataframe contining accurecies of vote model
    m_metric : pd.DataFrame
        Dataframe contining accurecies of meta model
    b_df : pd.DataFrame
        for future
    v_df : np.array
        for future
    m_df : pd.DataFrame
        for future
    out_features : str
        Name of the predicted feature (AHG ceoficients)
    custom_name : str
        A custom name for the modeling instance

    Example
    --------
    >>> PlotComp(b_metric, v_metric, m_metric, b_df, v_df, m_df,
                out_features = 'b', custom_name = 'test)
    """
    def __init__(self, b_metric: str, v_metric: str,
                m_metric: str, b_df: pd.DataFrame,
                v_df: pd.DataFrame, m_df: pd.DataFrame,
                out_features: str, custom_name: str) -> None:
        self.b_metric       = b_metric
        self.v_metric       = v_metric
        self.m_metric       = m_metric
        self.b_df           = b_df
        self.v_df           = v_df
        self.m_df           = m_df
        self.out_features   = out_features
        self.custom_name    = custom_name
        self.all_metrics    = []

    def processData(self) -> None:
        # data transformation ----------------------------------------
        self.all_metrics = []
        for i in range(0, len(self.b_metric)):
            self.all_metrics.append(([self.b_metric[i][0],self.v_metric[i][0],self.m_metric[i][0]], self.b_metric[i][1], 
                                self.b_metric[i][2], [self.b_metric[i][3],self.v_metric[i][3],self.m_metric[i][3]], self.b_metric[i][4]))
        return
    
    def plotComparisons(self) -> None:
        
        def plotCDF(metric: str, xlabel: str, ylabel: str) -> None:
            fig, axs = plt.subplots(1, 1, figsize=(8,4))
            b_metric = metric[0].astype(float)
            v_metric = metric[1].astype(float)
            m_metric = metric[2].astype(float)
            b_ecdf = ECDF(b_metric)
            v_ecdf = ECDF(v_metric)
            m_ecdf = ECDF(m_metric)
            axs.plot(b_ecdf.x, b_ecdf.y, linestyle='--', marker='.', color='black', label='best')
            b_ecdf.x[(b_ecdf.x>1e30) | (b_ecdf.x<-1e30)]=0
            axs.axvline(x=np.median(b_ecdf.x), color='k', ls='--')

            axs.plot(v_ecdf.x, v_ecdf.y, linestyle='--', marker='.', color='blue', label='vote')
            v_ecdf.x[(v_ecdf.x>1e30) | (v_ecdf.x<-1e30)]=0
            axs.axvline(x=np.median(v_ecdf.x), color='b', ls='--')

            axs.plot(m_ecdf.x, m_ecdf.y, linestyle='--', marker='.', color='red', label='meta')
            m_ecdf.x[(m_ecdf.x>1e30) | (m_ecdf.x<-1e30)]=0
            axs.axvline(x=np.median(m_ecdf.x), color='r', ls='--')

            axs.axhline(y=0.5, color='g', ls='--', label='median')
            if ylabel == 'NNSE':
                axs.axvline(x=0.66, color='g', ls='--', label='Good')
            axs.set(xlabel=xlabel, ylabel='CDF('+ylabel+')')
            plt.legend(loc="upper left") 
            my_plot = plt.gcf()
            plt.savefig(self.custom_name+'/img/model/'+str(self.custom_name)+'_'+self.out_features+'_COMP_CDF_'+ylabel+'.png',bbox_inches='tight', dpi = 600, facecolor='white')
            plt.show()

        for met, xl, yl, idx, cmap in tqdm(self.all_metrics):
            plotCDF(met, xl, yl)

        # def plotDist():
            

        