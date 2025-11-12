import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def output(l_TimeSeries_Result_Base,l_TimeSeries_Result_Anomaly, l_Outlier_Results):
    #=== Time Series Result=====================================

    # === plot===
        # ====Base and Anomaly as well as their respective transformations ====
    tsa_base_original = pd.concat([l_TimeSeries_Result_Base["train_set"], l_TimeSeries_Result_Base["test_set"]]).reset_index(drop=True)
    tsa_anomaly_original = pd.concat([l_TimeSeries_Result_Anomaly["train_set"], l_TimeSeries_Result_Anomaly["test_set"]]).reset_index(drop=True)
    plt.figure(figsize=(6,4)) 
    plt.plot(tsa_base_original, label="Base Observation", color="blue", alpha=0.7)
    plt.plot(tsa_anomaly_original, label="Anomaly Observation", color="red", alpha=0.7)

    plt.title("Base and Anomaly Observations")
    plt.legend()
    plt.xlabel("Milling Steps (cumuluated)")
    plt.ylabel("Torque in nM")
    plt.grid(True, linestyle=":", alpha=0.6)

        # === Plot the respective STL graphs. Maybe only transformed observation and trend ===
    tsa_base_train_trans = l_TimeSeries_Result_Base["train_trans_set"]["df_set"]
    tsa_anomaly_train_trans = l_TimeSeries_Result_Anomaly["train_trans_set"]["df_set"]

    stl_fitted_base =l_TimeSeries_Result_Base["stl_train"]
    stl_fitted_anomaly = l_TimeSeries_Result_Anomaly["stl_train"]

            # === Plot Base STL ========
    fig, axes_base = plt.subplots(4, 1, figsize=(10,6), sharex=True)
    axes_base[0].plot(tsa_base_train_trans.index, stl_fitted_base.observed, color="black")
    axes_base[0].set_ylabel("Torque[nM]")
    axes_base[0].set_title("STL of Transformed Base Training Set")

    axes_base[1].plot(tsa_base_train_trans.index, stl_fitted_base.trend, color="green")
    axes_base[1].set_ylabel("Trend")

    axes_base[2].plot(tsa_base_train_trans.index, stl_fitted_base.seasonal, color="blue")
    axes_base[2].set_ylabel("Seasonal")

    axes_base[3].plot(tsa_base_train_trans.index, stl_fitted_base.resid, color="red")
    axes_base[3].set_ylabel("Residual")
    axes_base[3].set_xlabel("Milling Steps (cumuluated)")

    plt.tight_layout()
            
            # === Plot Anomaly STL ======
    fig, axes_base = plt.subplots(4, 1, figsize=(10,6), sharex=True)
    axes_base[0].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.observed, color="black")
    axes_base[0].set_ylabel("Torque[nM]")
    axes_base[0].set_title("STL of Transformed Anomaly Training Set")

    axes_base[1].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.trend, color="green")
    axes_base[1].set_ylabel("Trend")

    axes_base[2].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.seasonal, color="blue")
    axes_base[2].set_ylabel("Seasonal")

    axes_base[3].plot(tsa_anomaly_train_trans.index, stl_fitted_anomaly.resid, color="red")
    axes_base[3].set_ylabel("Residual")
    axes_base[3].set_xlabel("Milling Steps (cumuluated)")

    # Plot the forecast for the next season (in this case 39 steps)
    base_forecast = l_TimeSeries_Result_Base["forecast_next_season"]
    anomaly_forecast = l_TimeSeries_Result_Anomaly["forecast_next_season"]

    plt.figure(figsize=(6,4)) 
    plt.plot(base_forecast, label="Base Forecast", color="blue", alpha=0.7)
    plt.plot(anomaly_forecast, label="Anomaly Forecast", color="red", alpha=0.7)

    plt.title("Base and Anomaly One Season ahead Forecast")
    plt.legend()
    plt.xlabel("Milling Steps (cumuluated)")
    plt.ylabel("Torque in nM")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Maybe plot hetero/homoscedasticity
    # === 
        # For the transformation write down the optimal lambda and season length used 
        # Get the trending strength, and the adf and kpss results as well as the result of their combination
        # Plot ACF and PACF after the series was differenced
        # Get Minimum Cutoff Thresholds and the associated threshholds
        # Get estimated vs actual "optimal" model
        #     Get ljung box and AIC especially
        # For the actual model show the constant, coefficients, Hetero and Jarue bera p values, sample size, mae_out_of_sample
        # Forecasted mean value
    # === Outlier Results =============================================================
    for i in range(len(l_Outlier_Results)):
    # outlier simulation results
        test_outlier = l_Outlier_Results[i]

        plt.figure(figsize=(6,4)) 
        plt.title("Anomaly Detection Simulation", fontsize=16)
        # === Create a blue "confidence band" based on the lowest found anomaly of baseline to observation
        # As we only have an anomaly set for a damaged mill head where the toque is higher than the base, only the uppe region is considered
        plt.fill_between(test_outlier["df_baseline_fore_median"].index, test_outlier["df_baseline_fore_median"].squeeze(), test_outlier["df_base_fore_band_values_upper"].squeeze(), color="#0072B2", alpha=0.2, label="Limit for Baseline")
        # === Plot the forecasted base and anomalous value  
        plt.plot(test_outlier["df_baseline_fore_median"], label="Forecast: Baseline", linestyle="--", color ="blue")
        plt.plot(test_outlier["df_anomal_fore_median_pos"], label="Forecast: Anomaly Positive", linestyle="--", color ="orange")
        plt.plot(test_outlier["df_observ_outDet"], label="Original Forecast", color="black")
        
        # === Plot the anomalies
        plt.scatter(test_outlier["df_anomalies_indices"], test_outlier["df_observ_outDet"].iloc[test_outlier["df_anomalies_indices"]], color="red", marker="x", s=50, label="Anomaly Detected")

        # === Create the plot with a legend and x an y-axis descriptions
        plt.legend()
        plt.xlabel("Production Step (cumuluated)")
        plt.ylabel("Value")
        plt.grid(True, linestyle=":", alpha=0.6)
    





    plt.show()