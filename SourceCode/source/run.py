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
import statsmodels.tsa.seasonal as STL
from sklearn.metrics import mean_absolute_error
import numpy as np

def run(str_path_undamaged, str_path_damaged, sDepVar, sRenameVar, n_Seasons, n_alpha, s_test_type, script_dir, nSplit=0.8, bAbs=False, str_FolderName=None):
    
    df_undamaged_train, df_undamaged_test = dfUtils.getTrainAndTestSet(pd.read_csv(str_path_undamaged), n_Seasons, sDepVar, sRenameVar, True, nSplit)
    df_damaged_train, df_damaged_test = dfUtils.getTrainAndTestSet(pd.read_csv(str_path_damaged), n_Seasons, sDepVar, sRenameVar, True, nSplit)

    if bAbs:
        df_undamaged_train = df_undamaged_train.abs()
        df_undamaged_test = df_undamaged_test.abs()
        df_damaged_train = df_damaged_train.abs()
        df_damaged_test = df_damaged_test.abs()
    
    tsa_undmg_results = doTimeSeriesAnalysis(df_undamaged_train, df_undamaged_test, n_Seasons, n_alpha, s_test_type) #produces fitted model for a given set and plotted graphs for analysing
    tsa_dmg_results = doTimeSeriesAnalysis(df_damaged_train, df_damaged_test, n_Seasons, n_alpha, s_test_type)

    # ========================= Outlier Detection Simulation
    df_damaged_train_cut = df_damaged_train.iloc[:10]

    concat_series = pd.concat([df_damaged_train_cut, df_undamaged_test])
    scrambled_series = concat_series.sample(frac=1).reset_index(drop=True)

    outDetect_result = simulateOutlierDetection(tsa_undmg_results, tsa_dmg_results, scrambled_series)
    # outDetect_result1 = simulateOutlierDetection(tsa_undmg_results, tsa_dmg_results, df_damaged_train)

    out.output(tsa_undmg_results, tsa_dmg_results, [outDetect_result],script_dir, str_FolderName)

def doTimeSeriesAnalysis(df_train, df_test, n_Seasons, n_alpha, s_test_type):

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
    stl_fitted = stl_train_set.fit()                                   # Decomposing via STL
    dict_results["stl_train"] = stl_fitted

    # === 3. Stationary Check ==================================================================
    b_Trending = stlUtils.getTrending(stl_fitted.trend, stl_fitted.resid, dict_results)
    dict_stat_ind = statUtil.getStatInd(df_train_trans, n_alpha, s_test_type, b_Trending, dict_results)

    # === 4. ARIMA Params Gathering ============================================================
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
        "mae" : n_MAE,
        "mae_in_sample": (n_MAE / df_train_trans.mean()) * 100,
        "mae_out_sample": (n_MAE / df_test.mean()) * 100
    } 

    # === 7. Forecast the next Season==========================================================================

    dict_results["forecast_next_season"] = arimaUtil.getForecast(fitted_model, n_Seasons, opt_lambda)

    return dict_results

def simulateOutlierDetection(m_TimeSeries_Baseline, m_TimeSeries_Anomalous, df_observ):

    dict_results = {}
    df_baseline_fore = arimaUtil.getForecast(m_TimeSeries_Baseline["fitted_optimal_model"], len(df_observ), m_TimeSeries_Baseline["train_trans_set"]["opt_lambda"])
    df_anomaly_fore = arimaUtil.getForecast(m_TimeSeries_Anomalous["fitted_optimal_model"], len(df_observ), m_TimeSeries_Anomalous["train_trans_set"]["opt_lambda"])
    dict_results["df_observ_outDet"] = df_observ

    # === 1. Get detected anomalies and its indices based on the given observations set
    anomalies_detected = outDetUtil.getAnomalies(df_baseline_fore, df_anomaly_fore, df_observ, dict_results)


    # === 2. Plot the detected anomalies with the given indices

    # Save the observation, median base forecast df, median anomalous forecast df
    #Base forecast
    n_median_baseline = np.median(df_baseline_fore)
    dict_results["df_baseline_fore_median"] = pd.DataFrame([n_median_baseline] * len(df_observ))

    # Anomalous forecast
    n_median_anomaly_pos = np.median(df_anomaly_fore)
    dict_results["df_anomal_fore_median_pos"]  = pd.DataFrame([n_median_anomaly_pos] * len(df_observ))

    return dict_results
    



















# def doSomething():
#     fig,axes = plt.subplots(4 , 1 ,figsize=(10,6), sharex=True)
#     axes[0].plot(df_train_3mm_edited.index, stl_fitted.observed, color="black")
#     axes[0].set_xlabel("Torque in nm transformed")
#     axes[0].set_ylabel("Torque[nM]")

#     axes[1].plot(df_train_3mm_edited.index, stl_fitted.trend, color="green")
#     axes[1].set_ylabel("Trend")

#     axes[2].plot(df_train_3mm_edited.index, stl_fitted.seasonal, color="blue")
#     axes[2].set_ylabel("Seasonal")

#     axes[3].plot(df_train_3mm_edited.index, stl_fitted.resid, color="red")
#     axes[3].set_ylabel("Residual")
#     axes[3].set_xlabel("Time in working steps")

#     plt.tight_layout()