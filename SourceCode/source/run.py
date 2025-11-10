from . import UtilsDataFrame as dfUtils
import pandas as pd
import matplotlib.pyplot as plt
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda
from . import STLUtils as stlUtils
from . import StationaryUtils as statUtil
from . import ACF_PACFUtils as corrUtil
from . import ARIMAUtils as arimaUtil
import statsmodels.tsa.seasonal as STL

def run(str_path_undamaged, tr_path_damaged, sDepVar, sRenameVar, n_Seasons, n_alpha, s_test_type, nSplit=0.8, bAbs=False,):

    # df_train_3mm_edited = udf.getTrainSet(pd.read_csv(str_path_undamaged), "Zaehler", "Torque_ax8", "Torque", True).abs()
    # df_test_3mm_edited = udf.getTestSet(pd.read_csv(file_path_3mm_noDMG_edited),"Zaehler", "Torque_ax8", "Torque", True).abs()
    # df_train_dmg_3mm_edited = udf.getTrainSet(pd.read_csv(file_path_3mm_DMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()
    # df_test_dmg_3mm_edited = udf.getTestSet(pd.read_csv(file_path_3mm_DMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()
    
    dict_results = {
        "train_set": None,
        "test_set" : None,
        "test_trans_set": None,
        "stl_train" : None
    }

    df_undamaged_train, df_undamaged_test = dfUtils.getTrainAndTestSet(pd.read_csv(str_path_undamaged), n_Seasons, sDepVar, sRenameVar, True, nSplit)
    if bAbs:
        df_undamaged_train = df_undamaged_train.abs()
        df_undamaged_test = df_undamaged_test.abs()
        
    dict_results["train_set"] = df_undamaged_train
    dict_results["test_set"] = df_undamaged_test

    TimeSeriesAnalysis(dict_results, df_undamaged_train, df_undamaged_test, n_Seasons, n_alpha, s_test_type) #produces fitted model for a given set and plotted graphs for analysing
    # outlierdetection() # based on time series analysis detects outliers

def TimeSeriesAnalysis(dict_results, df_undamaged_train, df_undamaged_test, n_Seasons, n_alpha, s_test_type):

    # === 1. Transform data =================================================================== 
    opt_lambda = boxcox_lambda(df_undamaged_train, method="guerrero", season_length=n_Seasons)
    df_undamaged_train_trans = pd.DataFrame(
        boxcox(df_undamaged_train, opt_lambda),
        index = df_undamaged_train.index,
        columns = df_undamaged_train.columns
    )
    dict_results["test_trans_set"] = df_undamaged_train_trans

    # === 2. STL Decomposition ================================================================= 
    stl_train_set = STL.STL(df_undamaged_train_trans, period=n_Seasons) # Set up the STL object and its params
    stl_fitted = stl_train_set.fit()                                   # Decomposing via STL
    dict_results["stl_train"] = stl_fitted

    # === 3. Stationary Check ==================================================================
    b_Trending = stlUtils.getTrending(stl_fitted.trend, stl_fitted.resid, dict_results)
    dict_stat_ind = statUtil.getStatInd(df_undamaged_train_trans, n_alpha, s_test_type, b_Trending, dict_results)

    # === 4. ARIMA Params Gathering ============================================================
    p, d, q = corrUtil.getARIMA_Params(df_undamaged_train_trans, n_Seasons*2 ,n_alpha, dict_stat_ind, dict_results)

    # === 5. Get Optimal ARIMA Model ===========================================================

    fitted_model = arimaUtil.getOptimalModel(df_undamaged_train_trans, p, d, q, dict_results)
    dict_results["fitted_optimal_model"] = fitted_model 
    print(fitted_model.summary()) #diagnostic with hetereoscedesticity. plot it maybe and jarque bera
    #forecast into leng(testSet), transform with inverse boxcox before with the lambda computed above

    # === 6. Model Goodness via Hetero/Homoskedatsicity, Jarque Bera and MAE
    # show these stats or rather save them as well?
    

    # === 7. Forecast ==========================================================================
    pred_forecast = arimaUtil.getForecast(model_fit, len(df_test_3mm_edited["Torque"]), opt_lambda_3mm_NoDmg)
    pred_forecast_for_dmg_train = arimaUtil.getForecast(model_fit, len(df_train_dmg_3mm_edited["Torque"]), opt_lambda_3mm_NoDmg)
    pred_forecast_for_dmg_test = arimaUtil.getForecast(model_fit, len(df_test_dmg_3mm_edited["Torque"]), opt_lambda_3mm_NoDmg)

    print(p, d, q)


    



















def doSomething():
    fig,axes = plt.subplots(4 , 1 ,figsize=(10,6), sharex=True)
    axes[0].plot(df_train_3mm_edited.index, stl_fitted.observed, color="black")
    axes[0].set_xlabel("Torque in nm transformed")
    axes[0].set_ylabel("Torque[nM]")

    axes[1].plot(df_train_3mm_edited.index, stl_fitted.trend, color="green")
    axes[1].set_ylabel("Trend")

    axes[2].plot(df_train_3mm_edited.index, stl_fitted.seasonal, color="blue")
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(df_train_3mm_edited.index, stl_fitted.resid, color="red")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Time in working steps")

    plt.tight_layout()