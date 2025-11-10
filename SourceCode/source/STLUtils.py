import statsmodels.tsa.seasonal as STL
import pandas as pd
import matplotlib.pyplot as plt

def getTrendStrength(df_stl_trend, df_stl_residual):
    
    """
    Calculates the trend strength based on STL components as shown by Hyndman
    by taking the variance of the residuals and dividing by 
    the variance of the sum of the trend and residuals.
    This function performs the following operations:
        1. Computes the sum of a given residual and trend provided by STL
        2. Gets the variance of the residual and sum of the trend and residuals
        3. Applies the formula with the components
    
    Args:
        df_stl_trend (pd.Series): Smoothed trend series
        df_stl_residual (pd.Series): Smooted residual series

    Returns:
        A factor indicating trend strength
    """
    n_sum_res_trend = df_stl_residual + df_stl_trend 

    var_res = df_stl_residual.var()
    var_res_trend = n_sum_res_trend.var()
    n_trend_strength = 1 - (var_res / var_res_trend)

    return max(0 , n_trend_strength)

def getTrending (df_stl_trend, df_stl_residual, dict_results): 
# Returns a bool indicating a trend
    n_trend_strength = getTrendStrength(df_stl_trend, df_stl_residual)

    # === save the trend result 0============================================
    dict_results["trend_info"] = {
        "trend_strength": n_trend_strength
    }
    #========================================================================
    
    if n_trend_strength >= 0.4:
        return True
    else:
        return False
