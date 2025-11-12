import matplotlib.pyplot as plt
import numpy as np

def output(l_TimeSeries_Result_Base,l_TimeSeries_Result_Anomaly, l_Outlier_Results):
    #=== Time Series Result=====================================

    # === plot
        # Base and Anomaly as well as their respective transformations
        # Plot the respective STL graphs. Maybe only transformed observation and trend
        # Plot the forecast for the next season (in this case 39 steps)

    plt.plot()



    # === 
        # For the transformation write down the optimal lambda and season lenght used 
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