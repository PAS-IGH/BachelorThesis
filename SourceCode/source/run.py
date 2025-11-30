from . import UtilsDataFrame as dfUtils
import pandas as pd
import matplotlib.pyplot as plt
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda
from . import STLUtils as stlUtils
from . import StationaryUtils as statUtil
from . import ACF_PACFUtils as corrUtil
from . import ARIMAUtils as arimaUtil
from . import OutlierDetectorUtil as outDetUtil
from . import OutPut as out
from . import ModelEvaluation as modEval
import statsmodels.tsa.seasonal as STL
from sklearn.metrics import mean_absolute_error
import numpy as np

def run(str_path_undamaged, str_path_damaged, sDepVar, sRenameVar, n_Seasons, n_alpha, s_test_type, script_dir, nSplit=0.8, bAbs=False, str_FolderName=None):

    """
    A function for running the time series analysis and outlier detection pipeline.
    This function performs the following operations:
        1. Preprocesses the data and provides the results as dataframes
        2. Turns the values with in the dataframes to their absolute values if needed
        3. Uses the dataframes for time series analysis
        4. Evaluates the models and the outlier detector based on it
        5. Constructs sets for the outlier simulation
        6. Implements the output function


    Args:
        str_path_undamaged (str): A string containing the file path of the base process
        str_path_damaged (str): A string containing the file path of the anomalous process
        sDepVar (str): A string containing the dependent variable
        sRenameVar (str): A string for renaming the dependent variable
        n_Seasons (int): An integer indicating the number of seasons/periods
        n_alpha (float): A floating point value for tests invloving an alpha value
        s_test_type (str): A string containing the stationary test type
        script_dir (str): A string pointing to the right project directory
        nSplit (float): A floating point number indicating the test split 
        bAbs (bool): A boolean for indicating the need to turn the values to their absolute counterparts
        str_FolderName (str): A string for naming a folder where the output can be stored
    """

    df_undamaged_train, df_undamaged_test = dfUtils.getTrainAndTestSet(pd.read_csv(str_path_undamaged), n_Seasons, sDepVar, sRenameVar, True, nSplit)
    df_damaged_train, df_damaged_test = dfUtils.getTrainAndTestSet(pd.read_csv(str_path_damaged), n_Seasons, sDepVar, sRenameVar, True, nSplit)

    if bAbs:
        df_undamaged_train = df_undamaged_train.abs()
        df_undamaged_test = df_undamaged_test.abs()
        df_damaged_train = df_damaged_train.abs()
        df_damaged_test = df_damaged_test.abs()
    
    tsa_undmg_results = doTimeSeriesAnalysis(df_undamaged_train, df_undamaged_test, n_Seasons, n_alpha, s_test_type) #produces fitted model for a given set and plotted graphs for analysing
    tsa_dmg_results = doTimeSeriesAnalysis(df_damaged_train, df_damaged_test, n_Seasons, n_alpha, s_test_type)


    t_model_detector_eval = modEval.getEvaluationResults(tsa_undmg_results, tsa_dmg_results)
    # ========================= Outlier Detection Simulation =============================
    # Get three cases for the three types of maintenance alerts

    df_damaged_test_sched_maint = df_damaged_test.iloc[-8:]
    df_damaged_test_sched_maint_asap = df_damaged_test.iloc[-31:]
    df_damaged_test_maint_imm = df_damaged_test.iloc[-60:]
    df_damaged_test_critical = df_damaged_test.iloc[-100:]

    concat_series_sched_maint = pd.concat([df_undamaged_test, df_damaged_test_sched_maint]).sample(frac=1).reset_index(drop=True)
    concat_series_sched_maint_asap = pd.concat([df_undamaged_test, df_damaged_test_sched_maint_asap]).sample(frac=1).reset_index(drop=True)
    concat_series_sched_maint_imme = pd.concat([df_undamaged_test, df_damaged_test_maint_imm]).sample(frac=1).reset_index(drop=True)
    concat_series_sched_crit = pd.concat([df_undamaged_test, df_damaged_test_critical]).sample(frac=1).reset_index(drop=True)

    outDetect_result_sched_main = simulateOutlierDetection(tsa_undmg_results, tsa_dmg_results, concat_series_sched_maint)
    outDetect_result_sched_maint_asap = simulateOutlierDetection(tsa_undmg_results, tsa_dmg_results, concat_series_sched_maint_asap)
    outDetect_result_sched_maint_imme  = simulateOutlierDetection(tsa_undmg_results, tsa_dmg_results, concat_series_sched_maint_imme)
    outDetect_result_sched_crit  = simulateOutlierDetection(tsa_undmg_results, tsa_dmg_results, concat_series_sched_crit)

    # reinstate later
    # out.output(tsa_undmg_results, tsa_dmg_results,t_model_detector_eval, [outDetect_result_sched_main, outDetect_result_sched_maint_asap, outDetect_result_sched_maint_imme, outDetect_result_sched_crit],script_dir, str_FolderName)

def doTimeSeriesAnalysis(df_train, df_test, n_Seasons, n_alpha, s_test_type):

    """
    A function for the implementation of the time series pipeline.
    This function performs the following operations:
        1. Transforms the given training set via a Box-Cox transformation with the optimal lambda estimated via the method by Guerrero
        2. Performs a STL decomposition
        3. Checks the stationary type of the series
        4. Gathers an estimation of ARIMA parameters based on the determined stationary type following the method by Tran and Reed
        5. Selects the optimal ARIMA model
        6. Checks variance and normality of residuals for possible further improvements
        7. Forecasts one season and returns the information gathered during this process
    Args:
        df_train (pandas.DataFrame): A dataframe containing the training set
        df_test (pandas.DataFrame): A dataframe containing the test set
        n_alpha (float): A floating point value for tests invloving an alpha value
        s_test_type (str): A string containing the stationary test type
    Returns:
        dict_results (dict): A dictionary object containing information gathered during the time series analysis
    """

    dict_results = {
    "train_set": df_train,
    "test_set" : df_test
    }
    # === 1. Transform data =================================================================== 
    opt_lambda = boxcox_lambda(df_train, method="guerrero", season_length=n_Seasons)
    df_train_trans = pd.DataFrame(
        boxcox(df_train, opt_lambda),
        index = df_train.index,
        columns = df_train.columns
    )
    dict_results["train_trans_set"] = {
        "df_set": df_train_trans,
        "opt_lambda": opt_lambda
        }

    # === 2. STL Decomposition ================================================================= 
    stl_train_set = STL.STL(df_train_trans, period=n_Seasons) # Set up the STL object and its params
    stl_fitted = stl_train_set.fit()                          # Decomposing via STL
    dict_results["stl_train"] = stl_fitted

    # === 3. Stationary Check ==================================================================
    b_Trending = stlUtils.getTrending(stl_fitted.trend, stl_fitted.resid, dict_results)
    dict_stat_ind = statUtil.getStatInd(df_train_trans, n_alpha, s_test_type, b_Trending, dict_results)

    # === 4. ARIMA Params Gathering ============================================================

    if dict_results['stationary_status']["stat_type"] == "trend":
        p, d, q = corrUtil.getARIMA_Params(df_train_trans, n_Seasons*2 ,n_alpha, dict_stat_ind, dict_results,df_trend_series =stl_fitted.trend)
    else:
        p, d, q = corrUtil.getARIMA_Params(df_train_trans, n_Seasons*2 ,n_alpha, dict_stat_ind, dict_results)
    # === 5. Get Optimal ARIMA Model ===========================================================
    #need to add one for detrend

    fitted_model = arimaUtil.getOptimalModel(df_train_trans, p, d, q, dict_results)
    dict_results["fitted_optimal_model"] = fitted_model 


    #diagnostic with hetereoscedesticity. plot it maybe and jarque bera

    # === 6. Residual Diagnostics for Model Validation via Hetero/Homoskedatsicity, Jarque Bera and MAE
    # show these stats or rather save them as well?


    a_forecast_for_mae = arimaUtil.getForecast(fitted_model, len(df_test),opt_lambda)

    n_MAE = mean_absolute_error(df_test, a_forecast_for_mae)
    dict_results["ARIMA"] = {
        "summary": fitted_model.summary().as_text(),
        "mae" : n_MAE
    } 

    # === 7. Forecast the next Season==========================================================================

    dict_results["forecast_next_season"] = arimaUtil.getForecast(fitted_model, n_Seasons, opt_lambda)

    return dict_results

def simulateOutlierDetection(m_TimeSeries_Baseline, m_TimeSeries_Anomalous, df_observ):

    """
    A function for simulating the outlier detector.
    This function performs the following operations:
        1. Gets the base and anomalous forecast
        2. Detects anomalies
        3. Saves the median forecasts for plotting and returns the results of the simulations

    Args:
        m_TimeSeries_Baseline (dict):  A dictionary containing information from the time series analysis of the base process
        m_TimeSeries_Anomalous (dict): A dictionary containing information from the time series analysis of the anomalous process
        df_observ (pandas.DataSeries): A series containing observation for the simulation
    Returns:
        dict_results (dict): A dictionary containing thre results of the simulation
    """

    dict_results = {}
    df_baseline_fore = arimaUtil.getForecast(m_TimeSeries_Baseline["fitted_optimal_model"], len(df_observ), m_TimeSeries_Baseline["train_trans_set"]["opt_lambda"])
    df_anomaly_fore = arimaUtil.getForecast(m_TimeSeries_Anomalous["fitted_optimal_model"], len(df_observ), m_TimeSeries_Anomalous["train_trans_set"]["opt_lambda"])
    dict_results["df_observ_outDet"] = df_observ

    # === 1. Get detected anomalies and its indices based on the given observations set
    anomalies_detected = outDetUtil.getAnomalies(df_baseline_fore, df_anomaly_fore, df_observ, dict_results)
    anomaly_percentage = outDetUtil.getRecommendation(anomalies_detected, df_observ, dict_results)

    # === 2. Plot the detected anomalies with the given indices

    # Save the observation, median base forecast df, median anomalous forecast df
    #Base forecast
    n_median_baseline = np.median(df_baseline_fore)
    dict_results["n_baseline_fore_median"] = n_median_baseline
    dict_results["df_baseline_fore_median"] = pd.DataFrame([n_median_baseline] * len(df_observ))

    # Anomalous forecast
    n_median_anomaly_pos = np.median(df_anomaly_fore)
    dict_results["n_median_anomaly_fore_pos"] = n_median_anomaly_pos
    dict_results["df_anomal_fore_median_pos"]  = pd.DataFrame([n_median_anomaly_pos] * len(df_observ))
    return dict_results