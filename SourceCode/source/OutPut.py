"""
Provides functions for generating the outputs for time series analysis and outlier detection simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def output(l_TimeSeries_Result_Base,l_TimeSeries_Result_Anomaly,l_Evaluation, l_Outlier_Results,script_dir, str_FolderName =None):

    """
    A wrapper function containing output methods for the results of the time series analysis of the base and anomalous process, as well as of the outlier detector.

    This function performs the following operations:
        1. Executes the plotting of the results of the base and anomalous time series analysis
        2. Executes the writing functions for the results of the base and anomalous time series analysis
        3. Executes the plotting of the results of the outlier simulations
        4. Executes the writing process of the outlier detection evaluation
    Args:
        l_TimeSeries_Result_Base (dict): A dictionary containing information from the time series analysis of the base process
        l_TimeSeries_Result_Anomaly (dict): A dictionary containing information from the time series analysis of the anomalous process
        l_Evaluation (dict): A dictionary containing the results of the outlier detector evaluation
        l_Outlier_Results (dict): A dictionary containing the results of the outlier detector simulation
        script_dir (str): A string pointing to the right project directory
        str_FolderName (str): A string for naming a folder where the output can be stored
    """


    #=== Time Series Result=====================================

    plotOutputsTSA(l_TimeSeries_Result_Base,l_TimeSeries_Result_Anomaly, script_dir, str_FolderName)
    # === Time Series Analysis output

    writeOutputsTSA(l_TimeSeries_Result_Base, script_dir,str_FolderName, "Base")
    writeOutputsTSA(l_TimeSeries_Result_Anomaly, script_dir,str_FolderName, "Anomaly")

    # === Outlier Results =============================================================
    plotOutlierSim(l_Outlier_Results,script_dir, str_FolderName)

    writeOutputOutlierSim(l_Outlier_Results,script_dir, str_FolderName)
    writeEval(l_Evaluation, script_dir, str_FolderName )


def writeEval(l_Evaluation, script_dir, str_FolderName):

    """
    A function for applying the markdown template in order to write out the outlier detection evaluation.
    This function performs the following operations:
        1. Obtains the appropriate template
        2. Constructs a model for the fields needed in the template and inserts the results
        3. Saves the markdown file 

    Args:
        l_Evaluation (dict): A dictionary containing the results of the outlier detector evaluation        
        script_dir (str): A string pointing to the right project directory
        str_FolderName (str): A string for naming a folder where the output can be stored
    """

    with open(f"{script_dir.parent}/output/reportEvaluationTemplate.md", "r", encoding="utf-8") as f:
        template_text = f.read()

        dict_context = {
            "base_to_base": l_Evaluation["base_to_base"],
            "ano_to_ano": l_Evaluation["ano_to_ano"],
            "base_to_ano": l_Evaluation["base_to_ano"],
            "ano_to_base": l_Evaluation["ano_to_base"],
            "TN": l_Evaluation["cm"]["TN"],
            "FP": l_Evaluation["cm"]["FP"],
            "FN": l_Evaluation["cm"]["FN"] ,
            "TP": l_Evaluation["cm"]["TP"],
            "precision": l_Evaluation["cm"]["precision"],
            "recall": l_Evaluation["cm"]["recall"]
        }

    try:
        os.makedirs(f"{script_dir.parent}/output/{str_FolderName}")
    except:
        print("Folder already exists. Saving it there")

    gen_text = template_text.format(**dict_context)
    with open(f"{script_dir.parent}/output/{str_FolderName}/model_eval.md", "w", encoding="utf-8") as f:
        f.write(gen_text)

def writeOutputsTSA(l_TimeSeries_Results, script_dir, str_FolderName, str_tsa_name):

    """
    A function for applying the markdown template in order to write out the time series analysis results.
    This function performs the following operations:
        1. Obtains the appropriate template
        2. Constructs a model for the fields needed in the template and inserts the results
        3. Saves the markdown file 

    Args:
        l_TimeSeries_Results (dict): A dictionary containing information from the time series analysis of a process    
        script_dir (str): A string pointing to the right project directory
        str_FolderName (str): A string for naming a folder where the output can be stored
        str_tsa_name (str): A string containing the string to identify the underlying process
    """

    with open(f"{script_dir.parent}/output/reportTSATemplate.md", "r", encoding="utf-8") as f:
        template_text = f.read()

    # construct strings and a proper table for displaying the results of the unit root tests
    dict_adf = l_TimeSeries_Results['adf_test_statistic']
    str_hypo_adf = "Failed to reject" if dict_adf["null_hypo"] else "Rejected" 
    adf_table_string = f"|ADF| {round(dict_adf['t_value'], 4)} | {round(dict_adf['critical_value'],6)} | {str_hypo_adf}|"

    dict_kpss = l_TimeSeries_Results['kpss_test_statistic']
    str_hypo_kpss = "Failed to reject" if dict_kpss["null_hypo"] else "Rejected"
    kpss_table_string = f"|KPSS| {round(dict_kpss['t_value'],4)} | {round(dict_kpss['critical_value'],6)} | {str_hypo_kpss}|"

    dict_context = {
        "var_name":l_TimeSeries_Results["train_set"].columns[0],
        "lambda": round(l_TimeSeries_Results["train_trans_set"]["opt_lambda"], 6),

        "trend_strength": round(l_TimeSeries_Results['trend_info']["trend_strength"], 6),
        "test_table_adf":adf_table_string,
        "test_table_kpss":kpss_table_string,
        "stationary_status":l_TimeSeries_Results['stationary_status']['stationary_status'],
        "stationary_type": l_TimeSeries_Results['stationary_status']['stat_type'],

        "threshold_table": l_TimeSeries_Results['cutoff_thresholds'].to_markdown(index=False),
        "Min_Lag": l_TimeSeries_Results['cutoff_thresholds_minimum']["Lag"],
        "Minimum_value": round(l_TimeSeries_Results['cutoff_thresholds_minimum']["Cut_T_Value"],6),
        "Plot_Type": l_TimeSeries_Results['cutoff_thresholds_minimum']["Plot_Type"],
        "sugg_Model": "",

        "candidate_table":"",
        "chosen_model": f"ARIMA{l_TimeSeries_Results['fitted_optimal_model'].model.order}",

        "n_obs":l_TimeSeries_Results['fitted_optimal_model'].nobs,
        "optimal_model_aic":round(l_TimeSeries_Results['fitted_optimal_model'].aic, 6),

        "coefficient_table":l_TimeSeries_Results["fitted_optimal_model"].params.to_markdown()
    }
    # === Get the estimated params
    p  = l_TimeSeries_Results['ARIMA_Params_estimated']["p"]
    d  = l_TimeSeries_Results['ARIMA_Params_estimated']["d"]
    q  = l_TimeSeries_Results['ARIMA_Params_estimated']["q"]
    arima_model_string = f"ARIMA({p}, {d}, {q})"
    dict_context["sugg_Model"] = arima_model_string

    # === Get the candidates models
    str_candidate_table = ""
    for i in range(len(l_TimeSeries_Results['models'])):
        
        order = l_TimeSeries_Results['models'][i]["order"]
        aic = round(l_TimeSeries_Results['models'][i]["aic"],4) 
        ljung_box_pValue = round(l_TimeSeries_Results['models'][i]["ljung_box_pValue"], 6)

        str_model = f"|{order}|{aic}|{ljung_box_pValue}|\n"
        str_candidate_table +=str_model

    dict_context["candidate_table"] = str_candidate_table
    # coefficient_table impl

    # === Write the object into the md file
    gen_text = template_text.format(**dict_context)

    try:
        os.makedirs(f"{script_dir.parent}/output/{str_FolderName}")
    except:
        print("Folder already exists. Saving it there")

    with open(f"{script_dir.parent}/output/{str_FolderName}/TSA_{str_tsa_name}.md", "w", encoding="utf-8") as f:
        f.write(gen_text)
    


def writeOutputOutlierSim(l_Outlier_Results,script_dir, str_FolderName):

    """
    A function for applying the markdown template in order to write out the outlier simulation output.
    This function performs the following operations:
        1. Obtains the appropriate template
        2. Constructs a model for the fields needed in the template and inserts the results
        3. Saves the markdown file 

    Args:
        l_Outlier_Results (dict): A dictionary containing information of the outlier detection simulation  
        script_dir (str): A string pointing to the right project directory
        str_FolderName (str): A string for naming a folder where the output can be stored
    """

    with open(f"{script_dir.parent}/output/reportOutlierSimTemplate.md", "r", encoding="utf-8") as f:
        template_text = f.read()


    try:
        os.makedirs(f"{script_dir.parent}/output/{str_FolderName}")
    except:
        print("Folder already exists. Saving it there")

    for i in range(len(l_Outlier_Results)):
        outl_res = l_Outlier_Results[i]

        dict_context = {
        "median_base_fore" :f"{outl_res['n_baseline_fore_median']:.6f}",
        "median_anomaly_fore": f"{outl_res['n_median_anomaly_fore_pos']:.6f}",
        "lowest_anomaly":f"{outl_res['lowest_anomaly']:.6f}",

        "anomalies": outl_res["df_anomalies"].to_markdown(),
        "failure_percentage": outl_res["failure_percentage"],
        "str_recommendation": outl_res["str_recommendation"] if outl_res["str_recommendation"] else "Operations can go on"
        }

        gen_text = template_text.format(**dict_context)
        with open(f"{script_dir.parent}/output/{str_FolderName}/OutlierSim_{i}.md", "w", encoding="utf-8") as f:
            f.write(gen_text)

def plotOutlierSim(l_Outlier_Results,script_dir, str_FolderName =None):

    """
    A function for plotting the outlier detector simulations.
    This function performs the following operations:
        1. Creates a figure and plots the medians of base and anomaly forecast
        2. Plots the found anomalies
        3. Creates and plots a no-anomaly area
        4. Saves the output

    Args:
        l_Outlier_Results (dict): A dictionary containing information of the outlier detection simulation  
        script_dir (str): A string pointing to the right project directory
        str_FolderName (str): A string for naming a folder where the output can be stored
    """
    for i in range(len(l_Outlier_Results)):
    # outlier simulation results
        test_outlier = l_Outlier_Results[i]

        plt.figure(figsize=(6,4)) 
        plt.title("Anomaly Detection Simulation", fontsize=16)
        # === Create a blue "confidence band" based on the lowest found anomaly of baseline to observation
        
        # === Plot the forecasted base and anomalous value  
        plt.plot(test_outlier["df_baseline_fore_median"], label="Forecast: Baseline", linestyle="--", color ="blue")
        plt.plot(test_outlier["df_anomal_fore_median_pos"], label="Forecast: Anomaly Positive", linestyle="--", color ="orange")
        plt.plot(test_outlier["df_observ_outDet"], label="Original Forecast", color="black")

        # As we only have an anomaly set for a damaged mill head where the toque is higher than the base, only the uppe region is considered
        plt.fill_between(test_outlier["df_baseline_fore_median"].index, test_outlier["df_baseline_fore_median"].squeeze(), test_outlier["df_base_fore_band_values_upper"].squeeze(), color="#0072B2", alpha=0.2, label="Limit for Baseline")
        # === Plot the anomalies
        plt.scatter(test_outlier["df_anomalies_indices"], test_outlier["df_observ_outDet"].iloc[test_outlier["df_anomalies_indices"]], color="red", marker="x", s=50, label="Anomaly Detected")

        # === Create the plot with a legend and x an y-axis descriptions
        plt.legend(loc="lower right")
        plt.xlabel("Milling Steps(cumulated)")
        plt.ylabel("Torque in Nm")
        plt.grid(True, linestyle=":", alpha=0.6)
        try: 
            os.makedirs(f"{script_dir.parent}/output/{str_FolderName}/plots")
        except:
            print("Directory already existing. Using existing instead")
        plt.savefig(f"{script_dir.parent}/output/{str_FolderName}/plots/OutlierSimulation_{i}.pdf", bbox_inches="tight")
        plt.clf()
        plt.close()

def plotOutputsTSA(l_TimeSeries_Result_Base,l_TimeSeries_Result_Anomaly,script_dir, str_FolderName =None):
    """
    A function for plotting the results of the time series analysis pipeline.
    This function performs the following operations:
        1. Creates a figure, plots the base and anomalous observations and saves the figure
        2. Creates a figure, plots the observation and trend component of the STL decomposition of both processes and saves the figure
        3. Creates a figure, plots the one season forecast and saves the figure
        4. Plots the results of heteroscedasticity, to be used for further improvements

    Args:
        l_TimeSeries_Result_Base (dict): A dictionary containing information from the time series analysis of the base process
        l_TimeSeries_Result_Anomaly (dict): A dictionary containing information from the time series analysis of the anomalous process
        script_dir (str): A string pointing to the right project directory
        str_FolderName (str): A string for naming a folder where the output can be stored
    """


    # === plot===
        # ====Base and Anomaly as well as their respective transformations ====
    tsa_base_original = pd.concat([l_TimeSeries_Result_Base["train_set"], l_TimeSeries_Result_Base["test_set"]]).reset_index(drop=True)
    tsa_anomaly_original = pd.concat([l_TimeSeries_Result_Anomaly["train_set"], l_TimeSeries_Result_Anomaly["test_set"]]).reset_index(drop=True)
    plt.figure(figsize=(6,4)) 
    plt.plot(tsa_base_original, label="Base Observation", color="blue", alpha=0.7)
    plt.plot(tsa_anomaly_original, label="Anomaly Observation", color="red", alpha=0.7)

    plt.title("Base and Anomaly Observations")
    plt.legend()
    plt.xlabel("Milling Steps (cumulated)")
    plt.ylabel("Torque in Nm")
    plt.grid(True, linestyle=":", alpha=0.6)

    #== Save Figure
    try: 
        os.makedirs(f"{script_dir.parent}/output/{str_FolderName}/plots")
    except:
        print("Directory already existing. Using existing instead")
    plt.savefig(f"{script_dir.parent}/output/{str_FolderName}/plots/Observations.pdf", bbox_inches="tight")

        # === Plot the respective STL graphs. Maybe only transformed observation and trend ===
    tsa_base_train_trans = l_TimeSeries_Result_Base["train_trans_set"]["df_set"]
    tsa_anomaly_train_trans = l_TimeSeries_Result_Anomaly["train_trans_set"]["df_set"]

    stl_fitted_base =l_TimeSeries_Result_Base["stl_train"]
    stl_fitted_anomaly = l_TimeSeries_Result_Anomaly["stl_train"]

            # === Plot Base STL ========
    fig, axes_base = plt.subplots(2, 1, figsize=(6,4), sharex=True)
    axes_base[0].plot(tsa_base_train_trans.index, stl_fitted_base.observed, color="black")
    axes_base[0].set_ylabel("Torque[Nm]")
    axes_base[0].set_title("STL of Transformed Base Training Set")

    axes_base[1].plot(tsa_base_train_trans.index, stl_fitted_base.trend, color="green")
    axes_base[1].set_ylabel("Trend")
    axes_base[1].set_xlabel("Milling Steps (cumulated)")

    # axes_base[2].plot(tsa_base_train_trans.index, stl_fitted_base.seasonal, color="blue")
    # axes_base[2].set_ylabel("Seasonal")

    # axes_base[3].plot(tsa_base_train_trans.index, stl_fitted_base.resid, color="red")
    # axes_base[3].set_ylabel("Residual")
    # axes_base[3].set_xlabel("Milling Steps (cumulated)")

    plt.tight_layout()
            #== Save Figure
    plt.savefig(f"{script_dir.parent}/output/{str_FolderName}/plots/STL_Base.pdf", bbox_inches="tight")

            # === Plot Anomaly STL ======
    fig, axes_base = plt.subplots(2, 1, figsize=(6,4), sharex=True)
    axes_base[0].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.observed, color="black")
    axes_base[0].set_ylabel("Torque[Nm]")
    axes_base[0].set_title("STL of Transformed Anomaly Training Set")

    axes_base[1].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.trend, color="green")
    axes_base[1].set_ylabel("Trend")
    axes_base[1].set_xlabel("Milling Steps (cumulated)")

    # axes_base[2].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.seasonal, color="blue")
    # axes_base[2].set_ylabel("Seasonal")

    # axes_base[3].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.resid, color="red")
    # axes_base[3].set_ylabel("Residual")
    # axes_base[3].set_xlabel("Milling Steps (cumulated)")

            #== Save Figure
    plt.savefig(f"{script_dir.parent}/output/{str_FolderName}/plots/STL_Anomaly.pdf", bbox_inches="tight")

    # Plot the forecast for the next season (in this case 39 steps)
    base_forecast = l_TimeSeries_Result_Base["forecast_next_season"]
    anomaly_forecast = l_TimeSeries_Result_Anomaly["forecast_next_season"]

    plt.figure(figsize=(6,4)) 
    plt.plot(anomaly_forecast, label="Anomaly Forecast", color="red", alpha=0.7)
    plt.plot(base_forecast, label="Base Forecast", color="blue", alpha=0.7)

    plt.title("Base and Anomaly One Season Forecast")
    plt.legend()
    plt.xlabel("Milling Steps (cumulated)")
    plt.ylabel("Torque in Nm")
    plt.grid(True, linestyle=":", alpha=0.6)

    #== Save Figure
    plt.savefig(f"{script_dir.parent}/output/{str_FolderName}/plots/BaseAndAnomalyForecast.pdf", bbox_inches="tight")

    # Maybe plot hetero/homoscedasticity
    fitted_model_base = l_TimeSeries_Result_Base["fitted_optimal_model"]
    fitted_model_anomaly = l_TimeSeries_Result_Anomaly["fitted_optimal_model"]
        # Base Residuals vs Fitted 
    plt.figure(figsize=(6,4)) 
    plt.scatter(fitted_model_base.fittedvalues, fitted_model_base.resid, alpha=0.6)
    plt.axhline(0, color="black")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Base Variance of Residuals")

        #== Save Figure
    plt.savefig(f"{script_dir.parent}/output/{str_FolderName}/plots/BaseResidVariance.pdf", bbox_inches="tight")
        # Anomaly Residuals vs Fitted 
    plt.figure(figsize=(6,4)) 
    plt.scatter(fitted_model_anomaly.fittedvalues, fitted_model_anomaly.resid, alpha=0.6)
    plt.axhline(0, color="black")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Anomaly Variance of Residuals")

        #== Save Figure
    plt.savefig(f"{script_dir.parent}/output/{str_FolderName}/plots/AnomalyResidVariance.pdf", bbox_inches="tight")