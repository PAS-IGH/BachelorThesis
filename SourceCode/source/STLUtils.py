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

# def plotSTL(stlFit, bObserved, bTrend, bSeasonal, bResidual): 

#     if not stlFit:
#         print("no stl fit given")

#     if bObserved and bTrend and bSeasonal and bResidual:
#         fig,axes = plt.subplots(4 , 1 ,figsize=(10,6), sharex=True)

#         axes[0].plot(df_train_3mm_edited.index, stl_fitted.observed, color="black")
#         axes[0].set_xlabel("Torque in nm transformed")
#         axes[0].set_ylabel("Torque[nM]")

#         axes[1].plot(df_train_3mm_edited.index, stl_fitted.trend, color="green")
#         axes[1].set_ylabel("Trend")

#         axes[2].plot(df_train_3mm_edited.index, stl_fitted.seasonal, color="blue")
#         axes[2].set_ylabel("Seasonal")

#         axes[3].plot(df_train_3mm_edited.index, stl_fitted.resid, color="red")
#         axes[3].set_ylabel("Residual")
#         axes[3].set_xlabel("Time in working steps")
    


#     plt.tight_layout()
#     plt.show()