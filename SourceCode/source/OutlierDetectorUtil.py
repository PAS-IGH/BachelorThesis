# This module provides
#       - A method identifying the highest score provided by comparing the forecast of a fitted model to the actual observations of a set of anomalous
#         observations
#       - A method detecting with said z-score the an ongoing or statically provided process to detect anomalies based on that score
from statsmodels.robust.scale import mad
import numpy as np
from . import ARIMAUtils as arimaUtil
import math
import pandas as pd


def getAnomalies(df_baseline_fore, df_anomaly_fore, df_observ, dict_results): 
    # Get the errors from the median of the forecast to the given observation 
    df_base_err = df_observ - np.median(df_baseline_fore)

    # Get the errors for the upper and lower bound by subtracting the median of the anomaly forecast from the observations and its inverted form
    df_anomaly_err_pos = (df_observ - np.median(df_anomaly_fore)).abs() # Idea like the band for ACF and PACF
    df_anomaly_err_neg = np.negative(df_anomaly_err_pos)

    # produces an array containing boolean for anomalies for each index where one happens to be based on being nearer to the error of the forecasted anomalous process
    is_anomaly = (df_anomaly_err_pos <= df_base_err) | (df_anomaly_err_neg >= df_base_err) 
    n_indices_anomalies = np.where(is_anomaly)[0]

    # create a data collection containing the band value for this observation
    n_min = np.abs(df_base_err.min().min())

    n_band_value =  n_min
    n_band_value_fore_upper = n_band_value + np.median(df_baseline_fore)
    n_band_value_fore_lower = np.median(df_baseline_fore) - n_band_value
    
    df_band_values = pd.DataFrame([n_band_value] * len(df_observ))
    df_band_values_base_fore_upper = pd.DataFrame([n_band_value_fore_upper] * len(df_observ))
    df_band_values_base_fore_lower = pd.DataFrame([n_band_value_fore_lower] * len(df_observ))
    # === Save Results =========
    dict_results["anomaly_bool"] = is_anomaly
    dict_results["df_anomalies_indices"] = n_indices_anomalies

    dict_results["df_base_err"] = df_base_err
    dict_results["df_base_band_values"] = df_band_values
    dict_results["df_base_fore_band_values_upper"] = df_band_values_base_fore_upper
    dict_results["df_base_fore_band_values_lower"] = df_band_values_base_fore_lower

    dict_results["df_anomaly_err_pos"] = df_anomaly_err_pos
    dict_results["df_anomaly_err_neg"] = df_anomaly_err_neg

    #===========================




























# def detectOutlier(dict_data_baseline, df_data_compare, n_max_score, n_jarque_bera = None):
#     # gets the mad scors of ongoing
#     #detects the percentage of anomalies of a given observation set
#     df_forecast_comp = arimaUtil.getForecast(dict_data_baseline["fitted_optimal_model"], len(df_data_compare), dict_data_baseline["test_trans_set"]["opt_lamda"])

#     df_scores = getMADScores(df_forecast_comp, df_data_compare)

#     df_filtered = df_scores[(df_scores >= n_max_score).all(axis=1)]
#     n_percentage_anomalies = len(df_filtered) / len(df_data_compare) *100

#     return n_percentage_anomalies 

# def getAnomalyThreshold(dict_data_baseline , dict_data_anomaly, df_compare):
    
#     c = 1.00
#     # df_anomaly_test = dict_data_anomaly["train_set"]
#     # df_baseline_fore = arimaUtil.getForecast(dict_data_baseline["fitted_optimal_model"], len(df_anomaly_test), dict_data_baseline["test_trans_set"]["opt_lamda"])

#     # df_abs_err = np.abs(df_anomaly_test - np.median(df_baseline_fore))
#     # n_mad = mad(df_abs_err, c = c)
#     # #score calc
#     # n_mad_scores = df_abs_err / n_mad
#     # n_max_score = np.quantile(n_mad_scores , 0.75)

#     df_baseline_fore = arimaUtil.getForecast(dict_data_baseline["fitted_optimal_model"], len(df_compare), dict_data_baseline["test_trans_set"]["opt_lamda"])
#     df_anomaly_fore = arimaUtil.getForecast(dict_data_anomaly["fitted_optimal_model"], len(df_compare), dict_data_anomaly["test_trans_set"]["opt_lamda"])

#     df_baseline_err = df_compare - df_baseline_fore
#     df_anomaly_err = df_compare - df_anomaly_fore

#     is_anomaly = df_anomaly_err < df_baseline_err
#     return n_max_score    

# def getMADScores(df_baseline_fore,df_to_compare, n_jarque_bera = None): 
#     c = 1.00
#     df_abs_err = np.abs(df_to_compare - np.median(df_baseline_fore))
#     n_mad = mad(df_abs_err, c = 1.00)

#     #score per observation
#     df_scores = df_abs_err / n_mad

#     return df_scores



