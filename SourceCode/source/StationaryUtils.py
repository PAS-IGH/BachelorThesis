import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import kpss

def getStatInd(df_DataSeries, n_alpha, s_test_type, bTrending, dict_results):
    # n_Stationary =  getStationary(df_DataSeries, s_regress, n_alpha, s_test_type, dict_results)
    dict_stat = {
        "b_Difference": False,
        "b_Detrend": False,
        "b_Stationary": False,  
    }

    if bTrending:
        s_regress = "ct"
    elif not bTrending:
        s_regress = "c"
    
    str_Stationary =  getStationary(df_DataSeries, s_regress, n_alpha, s_test_type, dict_results)

    if str_Stationary == "trend":
        dict_stat["b_Detrend"] = True # detrend 
    elif str_Stationary == "stationary":
        dict_stat["b_Stationary"] = True #stationary
    elif str_Stationary == "difference":
        dict_stat["b_Difference"] = True #difference

    return dict_stat


def getStationary(df_DataSeries, s_regress, n_alpha, s_test_type, dict_results):
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
    trFloatingAlphaToString(n_alpha)

    if s_test_type == "ADF":
        return checkStatADFKPSS(df_DataSeries, s_regress, trFloatingAlphaToString(n_alpha), dict_results)
    else:
        raise ValueError("this test is unavailable at the moment")

    

def checkStatADFKPSS (df_DataSeries, s_ARParam, s_alpha, dict_results):
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
        A string indicating stationary type
    """
    adfRes = adfWrapper(df_DataSeries, s_ARParam, s_alpha, dict_results)
    kpssResult = kpssWrapper(df_DataSeries, s_ARParam, s_alpha, dict_results)

    if not adfRes and kpssResult:
        # === Save results =======================
        dict_results["stationary_status"] = {
            "stationary_status" : "stationary",
            "stat_type": "stationary"
        }
        str_stat_type = "stationary"
        #=========================================
    elif adfRes and not kpssResult:
        # === Save results =======================
        dict_results["stationary_status"] = {
            "stationary_status" : "non-stationary",
            "stat_type":"difference"
        }
        #=========================================
        str_stat_type = "difference"
    
    elif not adfRes and not kpssResult :
        # === Save results =======================
        dict_results["stationary_status"] = {
            "stationary_status" : "non-stationary",
            "stat_type":"difference"
        }
        #=========================================
        str_stat_type = "difference"

    elif adfRes and kpssResult:

        # === Save results =======================
        dict_results["stationary_status"] = {
            "stationary_status" : "non-stationary",
            "stat_type": "trend"
        }
        #=========================================
        str_stat_type = "trend"
    return str_stat_type
         

def adfWrapper(df_DataSeries, s_ARParam, s_alpha, dict_results):

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
    n_adf_stat= t_adf[0]
    n_adf_crit = t_adf[4][s_alpha]
    bNullHypo = True #set true as the test expects a non stationary time 
    
    #=== Save results ===
    dict_results["adf_test_statistic"] = {
        "t_value": n_adf_stat,
        "critical_value": n_adf_crit,
        "null_hypo": False if n_adf_stat <= n_adf_crit else True
    }
    #====================

    if n_adf_stat <= n_adf_crit: # t-value less than critical reject
        return not bNullHypo
    elif n_adf_stat > n_adf_crit:
        return bNullHypo #t-value more than critical value fails to reject
    
def kpssWrapper(df_DataSeries, s_ARParam, s_alpha, dict_results):

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
    n_kpss_stat = t_kpss[0]
    n_kpss_crit = t_kpss[3][s_alpha]
    bNullHypo = True

    #=== Save results ===
    dict_results["kpss_test_statistic"] = {
        "t_value": n_kpss_stat,
        "critical_valu": n_kpss_crit, 
        "null_hypo": False if n_kpss_stat >= n_kpss_crit else True
    }
    #====================

    if n_kpss_stat >= n_kpss_crit:
        return not bNullHypo #inverted to adf, h_0 rejected if t-value is higher than critical
    elif n_kpss_stat < n_kpss_crit:
        return bNullHypo
    
def trFloatingAlphaToString(n_alpha): 

    """
    Turns a numerical alpha into a string in order to use the tests
    This function performs the following operations:
        1. Takes a floating point alpha value and turns it into <number>% string
    Args:
        n_alpha(nr): A floating point alpha value
    Returns
        A alpha value string
    """
    n_percent = int(n_alpha * 100)
    return f"{n_percent}%"