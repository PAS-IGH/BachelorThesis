import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import kpss

def getStationary(df_DataSeries, s_regress, s_alpha, s_test_type):
    """
    Tells if a series is stationary or not with a given test and alpha value
    This function performs the following operations:
        1. Takes the received params and runs the specific methods based on the test type
        2. Indicates if stationary or not; depending on the regression type, the interpretation varies
    Args:
        df_DataSeries (pd.Series): A series which is tested on its stationarity
        s_regress (str): A string dictating regression type (ct = constant and trend, c=constant, n = no constant/trend)
        s_alpha (str): A string indicating the percentage for the alpha value
        s_test_type (str): The type desired test 
    Returns:
        A integer indicating the stationary status
        * 0 : stationary
        * 1 : non stationary
        * -1: inconcusive
    """
    if s_test_type == "ADF":
        return checkStatADFKPSS(df_DataSeries, s_regress, s_alpha)
    else:
        raise ValueError("this test is unavailable at the moment")

    

def checkStatADFKPSS (df_DataSeries, s_ARParam, s_alpha):
    """
    Provides the result of the conjoined ADF and KPSS test 
    This function performs the following operations:
        1. Gets the results of ADF and KPSS hypothesis tests
        2. Based on the KPSS paper by Kwiatkowski et al. interprets these 

    Args: 
        df_DataSeries (pd.Series): A series which is tested on its stationarity
        s_regress (str): A string dictating regression type (ct = constant and trend, c=constant, n = no constant/trend)
        s_alpha (str): A string indicating the percentage for the alpha value
    Returns:
        A integer indicating the stationary status
        * 0 : stationary
        * 1 : non stationary
        * -1: inconcusive
    """
    adfRes = adfWrapper(df_DataSeries, s_ARParam, s_alpha)
    kpssResult = kpssWrapper(df_DataSeries, s_ARParam, s_alpha)

    if not adfRes and kpssResult:
        return 0 #stationary
    elif adfRes and not kpssResult:
        return 1 #non-stationary
    elif not adfRes and not kpssResult:
        return -1 #inconclusive 
        # adfWrapper = reject(false)/failure to reject(true)
        # kpssWrapper = reject(false)/failure to reject(true)


def adfWrapper(df_DataSeries, s_ARParam, s_alpha):

    """
    Provides the result of the ADF test
    This function performs the following operations:
        1. Gets the test statistic and critical value based on alpha
        2. Compares these and rejects or fails to reject H_0 

    Args: 
        df_DataSeries (pd.Series): A series which is tested on its stationarity
        s_regress (str): A string dictating regression type (ct = constant and trend, c=constant, n = no constant/trend)
        s_alpha (str): A string indicating the percentage for the alpha value
    Returns:
        A boolean indicating the result of the hypothesis test
        * True : H_0 rejection failure
        * False : H_1 was rejected 
    """
    t_adf = adf(df_DataSeries, regression =s_ARParam, autolag="AIC") #get the test statistic and critical value
    # print(t_adf)
    n_adf_stat= t_adf[0]
    n_adf_crit = t_adf[4][s_alpha]
    bNullHypo = True #set true as the test expects a non stationary time series

    if n_adf_stat <= n_adf_crit: # t-value less than critical reject
        return not bNullHypo
    elif n_adf_stat > n_adf_crit:
        return bNullHypo #t-value more than critical value fails to reject
    
def kpssWrapper(df_DataSeries, s_ARParam, s_alpha):

    """
    Provides the result of the KPSS test
    This function performs the following operations:
        1. Gets the test statistic and critical value based on alpha
        2. Compares these and rejects or fails to reject H_0 

    Args: 
        df_DataSeries (pd.Series): A series which is tested on its stationarity
        s_regress (str): A string dictating regression type (ct = constant and trend, c=constant, n = no constant/trend)
        s_alpha (str): A string indicating the percentage for the alpha value
    Returns:
        A boolean indicating the result of the hypothesis test
        * True : H_0 rejection failure
        * False : H_1 was rejected 
    """
    t_kpss = kpss(df_DataSeries, regression=s_ARParam)
    # print(t_kpss)
    n_kpss_stat = t_kpss[0]
    n_kpss_crit = t_kpss[3][s_alpha]
    bNullHypo = True

    if n_kpss_stat >= n_kpss_crit:
        return not bNullHypo #inverted to adf, h_0 rejected if t-value is higher than critical
    elif n_kpss_stat < n_kpss_crit:
        return bNullHypo