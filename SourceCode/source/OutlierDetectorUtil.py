# This module provides
#       - A method identifying the highest score provided by comparing the forecast of a fitted model to the actual observations of a set of anomalous
#         observations
#       - A method detecting with said z-score the an ongoing or statically provided process to detect anomalies based on that score
from statsmodels.robust.scale import mad
import numpy as np
from . import ARIMAUtils as arimaUtil
def getAnomalyThreshold(dict_data_baseline , dict_data_anomaly, n_Seasons):
    
    df_baseline_fore = arimaUtil.getForecast(dict_data_baseline["fitted_optimal_model"], n_Seasons, dict_data_baseline["test_trans_set"]["opt_lamda"])
    df_anomaly_fore = arimaUtil.getForecast(dict_data_anomaly["fitted_optimal_model"], n_Seasons, dict_data_anomaly["test_trans_set"]["opt_lamda"])

    n_mad_score = getMADScores(df_anomaly_fore, df_baseline_fore).max()

    return n_mad_score

def getMADScores(df_anomaly_fore, df_baseline_fore, n_jarque_bera = None): 

    if not n_jarque_bera:

        # df_err = df_anomaly_fore - df_baseline_fore
        # n_med = np.median(df_err)

        n_med_baseline = np.median(df_baseline_fore)
        
        n_mad = mad(df_err, c=1.0)
        df_score = np.abs(df_err - n_med) / n_mad
        return df_score
    else: 
        df_err = df_anomaly_fore - df_baseline_fore
        n_med = np.median(df_err)
        n_mad = mad(df_err, c=1.4826)
        df_score = np.abs(df_err - n_med) / n_mad
        return df_score

def detectOutlier(df_forecast, df_ong_obs, n_max_score, n_jarque_bera = None):
    # gets the mad scors of ongoing
    #detects the percentage of anomalies of a given observation set
    df_mad_scores = getMADScores(df_ong_obs, df_forecast, n_jarque_bera)
    print(df_mad_scores) 
    df_obs_filtered = df_mad_scores[df_mad_scores >= n_max_score]
    n_percentage_anomalies = len(df_obs_filtered) / len(df_ong_obs) * 100 # amount of anomalies per observation set detected maybe math floor would make sense
    return n_percentage_anomalies

