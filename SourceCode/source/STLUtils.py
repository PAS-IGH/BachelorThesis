import statsmodels.tsa.seasonal as STL
import pandas as pd
import matplotlib.pyplot as plt

def getTrendStrength(df_stl_trend, df_stl_residual):
    
    """
    Calculates the trend strength based on the residual and trend component as shown by Hyndman and Athanasopoulos
    This function performs the following operations:
        1. Computes the components and inserts them into the formula
        2. Returns the trend strength 
    Args:
        df_stl_trend (pd.DataSeries): Trend component
        df_stl_residual (pd.DataSeries): Residual component

    Returns:
        (float): A floating point number indicating trend strength
    """
    n_sum_res_trend = df_stl_residual + df_stl_trend 

    var_res = df_stl_residual.var()
    var_res_trend = n_sum_res_trend.var()
    n_trend_strength = 1 - (var_res / var_res_trend)

    return max(0 , n_trend_strength)

def getTrending (df_stl_trend, df_stl_residual, dict_results): 

    """
    Decides if the given trend strength warrants a trending series
    This function performs the following operations:
        1. Obtains the trend strength
        2. Returns the result
    Args:
        df_stl_trend (pd.DataSeries): Trend component
        df_stl_residual (pd.DataSeries): Residual component
        dict_results (dict): A dictionary containing information of previous time series analysis steps
    Returns:
        (bool): A boolean indicating a trending series 
    """

    n_trend_strength = getTrendStrength(df_stl_trend, df_stl_residual)

    # === Save the trend result ============================================
    dict_results["trend_info"] = {
        "trend_strength": n_trend_strength
    }
    #========================================================================
    
    if n_trend_strength >= 0.4:
        return True
    else:
        return False
