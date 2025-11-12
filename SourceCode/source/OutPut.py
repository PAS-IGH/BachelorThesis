import matplotlib.pyplot as plt
import numpy as np

def output(l_TimeSeries_Results, l_Outlier_Results):




    # outlier simulation results
    test_outlier = l_Outlier_Results[0]

    plt.figure(figsize=(6,4)) 
    plt.title("Anomaly Detection Simulation", fontsize=16)
    # === Create a blue "confidence band based on the max error of baseline to observation"

    plt.fill_between(test_outlier["df_baseline_fore_median"].index, test_outlier["df_base_fore_band_values_lower"].squeeze(), test_outlier["df_base_fore_band_values_upper"].squeeze(), color="#0072B2", alpha=0.2, label="Limits for Baseline")
    # === Plot the forecasted base and anomalous value  
    plt.plot(test_outlier["df_baseline_fore_median"], label="Forecast: Baseline", linestyle="--", color ="blue")
    # plt.plot(test_outlier["df_anomal_fore_median_pos"], label="Forecast: Anomaly Positive", linestyle="--", color ="orange")
    # plt.plot(test_outlier["df_anomal_fore_median_neg"], label="Forecast: Anomaly Negative", linestyle="--", color ="orange")

    plt.plot(test_outlier["df_observ_outDet"], label="Original Forecast", color="black")
    
    # === Plot the anomalies
    plt.scatter(test_outlier["df_anomalies_indices"], test_outlier["df_observ_outDet"].iloc[test_outlier["df_anomalies_indices"]], color="red", marker="x", s=50, label="Anomaly Detected")

    # === Create the plot with a legend and x an y-axis descriptions
    plt.legend()
    plt.xlabel("Production Step (cumuluated)")
    plt.ylabel("Value")
    plt.grid(True, linestyle=":", alpha=0.6)




    plt.show()