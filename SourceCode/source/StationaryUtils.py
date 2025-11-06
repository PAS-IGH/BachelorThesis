import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import kpss

def getStationary(df_DataSeries, s_regress, n_alpha, s_test_type):

    if s_test_type == "ADF":
        return checkStatADFKPSS(df_DataSeries, s_regress, n_alpha)
    else:
        print("This test is not available atm")
        SystemExit(0)

    

def checkStatADFKPSS (df_DataSeries, s_ARParam, n_alpha):

    if not adfWrapper(df_DataSeries, s_ARParam, n_alpha) and kpssWrapper(df_DataSeries, s_ARParam, n_alpha):
        return 0 #stationary
    elif adfWrapper(df_DataSeries, s_ARParam, n_alpha) and not kpssWrapper(df_DataSeries, s_ARParam, n_alpha):
        return 1 #non-stationary
    elif not adfWrapper(df_DataSeries, s_ARParam, n_alpha) and not kpssWrapper(df_DataSeries, s_ARParam, n_alpha):
        return -1 #inconclusive 
        # adfWrapper = reject(false)/failure to reject(true)
        # kpssWrapper = reject(false)/failure to reject(true)


def adfWrapper(df_DataSeries, s_ARParam, n_alpha):

    t_adf = adf(df_DataSeries, regression =s_ARParam, autolag="AIC") #get the test statistic and critical value
    print(t_adf)
    n_adf_stat= t_adf[0]
    n_adf_crit = t_adf[4][n_alpha]
    bNullHypo = True #set true as the test expects a non stationary time series

    if n_adf_stat <= n_adf_crit: # t-value less than critical reject
        return not bNullHypo
    elif n_adf_stat > n_adf_crit:
        return bNullHypo #t-value more than critical value fails to reject
    
def kpssWrapper(df_DataSeries, s_ARParam, n_alpha):
        
    t_kpss = kpss(df_DataSeries, regression=s_ARParam)
    print(t_kpss)
    n_kpss_stat = t_kpss[0]
    n_kpss_crit = t_kpss[3][n_alpha]
    bNullHypo = True

    if n_kpss_stat >= n_kpss_crit:
        return not bNullHypo #inverted to adf, h_0 rejected if t-value is higher than critical
    elif n_kpss_stat < n_kpss_crit:
        return bNullHypo