"""
Provides functions for evaluating model and outlier detector performance.
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import numpy as np
from . import ARIMAUtils as arimaUtil
from . import OutlierDetectorUtil as outDetUtil

def getEvaluationResults(df_test_base_results, df_test_ano_results):
    """
    A function for determining forecasting precision and determining outlier detector performance based on a confusion matrix.
    This function performs the following operations:
        1. Gets the base and anomalous forecast based on the length of their respective test sets
        2. Gets the base and anomalous forecast based on the length of their non-respective test sets
        3. Computes out of sample MAE values with these forecasts to evaluate forecasting performance and validate outlier detection by forecasting error
        4. Constructs a ground truth label set with a concatenated set of base and anomalous test sets
        5. Obtains the predicted labels via application of the detector on the test sets and concatenates the produced labels
        6. Constructs the confusion matrix based on ground truth and predicted labels
        7. Saves the result in a dict object and returns it along MAE values
    Args:
        df_test_base_results (dict): A dictionary containing information from the previous time series analysis steps of the base process
        df_test_ano_results (dict): A dictionary containing information from the previous time series analysis steps of the anomalous process
    Returns:
        dict_results (dict): A dictionary containing the MAE values and confusion matrix, with the computed recall and precision
    """

    # base
    df_test_base_obs = df_test_base_results["test_set"]
    a_forecast_base = arimaUtil.getForecast(df_test_base_results["fitted_optimal_model"], len(df_test_base_obs), df_test_base_results["train_trans_set"]["opt_lambda"])
    # anomaly 
    df_test_anom_obs = df_test_ano_results["test_set"]
    a_forecast_ano = arimaUtil.getForecast(df_test_ano_results["fitted_optimal_model"], len(df_test_anom_obs), df_test_ano_results["train_trans_set"]["opt_lambda"])
    # additionally forecast to the length of the other procesh in case of differently sized sets
    a_forecast_base_for_ano = arimaUtil.getForecast(df_test_base_results["fitted_optimal_model"], len(df_test_anom_obs), df_test_base_results["train_trans_set"]["opt_lambda"])
    a_forecast_ano_for_base = arimaUtil.getForecast(df_test_ano_results["fitted_optimal_model"], len(df_test_base_obs), df_test_ano_results["train_trans_set"]["opt_lambda"])
    # mae base to base
    n_MAE_base_base = mean_absolute_error(df_test_base_obs, a_forecast_base) *100
    # mae anomaly to anomaly
    n_MAE_ano_ano = mean_absolute_error(df_test_anom_obs, a_forecast_ano) * 100
    # mae base to anomaly
    n_MAE_base_ano = mean_absolute_error(df_test_base_obs, a_forecast_ano_for_base) * 100
    # mae anomaly to base
    n_MAE_ano_base = mean_absolute_error(df_test_anom_obs, a_forecast_base_for_ano) *100


    # === Confusion matrix
    # labels
    observ_to_base = {}
    observ_to_anom = {}

    y_true_base = np.zeros(len(df_test_base_obs)) # 0, ergo false means that no anomaly is detected
    y_true_anom = np.ones(len(df_test_anom_obs)) # 1 , ergo true an anomaly was detected

    y_true = np.concatenate([y_true_base, y_true_anom])

    # get base for matrix
    a_base_forecast_anomalies = outDetUtil.getAnomalies(a_forecast_base, a_forecast_ano_for_base, df_test_base_obs, observ_to_base) 
    anomaly_indices_base = observ_to_base["anomaly_bool"]
   
    # get ano for matrix
    a_ano_forecast_anomalies = outDetUtil.getAnomalies(a_forecast_base_for_ano, a_forecast_ano, df_test_anom_obs, observ_to_anom) 
    anomaly_indices_ano = observ_to_anom["anomaly_bool"]
    # concatenate
    y_pred = np.concatenate([anomaly_indices_base, anomaly_indices_ano])


    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100

    dict_results = {
        "base_to_base": f"{n_MAE_base_base:.6f}%",
        "ano_to_ano":f"{n_MAE_ano_ano:.6f}%",
        "base_to_ano":f"{n_MAE_base_ano:.6f}%",
        "ano_to_base": f"{n_MAE_ano_base:.6f}%",
        "cm": {
            "TN": TN,
            "FP": FP,
            "FN": FN ,
            "TP": TP,
            "precision":f"{precision:.6f}%",
            "recall": f"{recall:.6f}%"
        }
    }

    return dict_results
