import matplotlib.pyplot as plt

def output(l_TimeSeries_Results, l_Outlier_Results):




    # outlier simulation results
    test_outlier = l_Outlier_Results[0]

    plt.figure(figsize=(6,4)) 
    plt.title("Anomaly Detection Simulation", fontsize=16)

    plt.plot(test_outlier["df_baseline_fore_median"], label="Forecast: Baseline", linestyle="--", color ="blue")
    plt.plot(test_outlier["df_anomal_fore_median_pos"], label="Forecast: Anomaly Positive", linestyle="--", color ="orange")
    plt.plot(test_outlier["df_anomal_fore_median_neg"], label="Forecast: Anomaly Negative", linestyle="--", color ="orange")

    plt.plot(test_outlier["df_observ_outDet"], label="Original Forecast", color="black")
    

    plt.scatter(test_outlier["df_anomalies"], test_outlier["df_observ_outDet"].iloc[test_outlier["df_anomalies"]], color="red", marker="x", s=50, label="Anomaly Detected")

    plt.legend()
    plt.xlabel("Production Step (cumuluated)")
    plt.ylabel("Value")
    plt.grid(True, linestyle=":", alpha=0.6)




    plt.show()