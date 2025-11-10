#put a method in here starting it all?
from . import OutlierDetectorUtil as detUtil
from . import run as run
from pathlib import Path

# read the given data sets this is part of the main
script_dir = Path(__file__).parent

file_path_3mm_noDMG_edited = script_dir.parent / "testData" /"3mm"/"3mm_NoDMG_20092025_edited.csv" #that should also be in createTimeSeriesDataFrame, the read csv file should always be turned into a dataframe 
file_path_3mm_DMG_edited =  script_dir.parent / "testData" / "3mm" / "3mm_DMG_20092025_edited.csv" 

# Everything at this point needs to go into its seperate thing
#torque data needs to be put into absolute values due to the machine giving inverted data
run.run(file_path_3mm_noDMG_edited, file_path_3mm_DMG_edited, "Torque_ax8", "Torque", 39, 0.05, "ADF", bAbs= True) #nSeasons how many seasons with obs there are, to get observ per season

#Outlier Detector, MAD and training it inv_boxcox opt_lambda_3mm_NoDmg
#Outlier detector impl
    # Get the model and the ratio for outlier with MAD_model MAD_damaged
    # Simulate with datasets

# n_error = df_train_dmg_3mm_edited["Torque"] - pred_forecast_for_dmg
# med = np.median(n_error)
# mad = mad(n_error, c=1.0)
# n_z_score = np.abs(n_error - med) / mad
# print(n_z_score.max())
# print(n_z_score.idxmax())
# n_max_score = detUtil.getAnomalyThreshold(df_train_dmg_3mm_edited["Torque"], pred_forecast_for_dmg_train)
print(f"this {n_max_score} max")

# Simulate to show how the anomaly detection works
# print(pred_forecast_for_dmg_test)
# print(df_test_dmg_3mm_edited["Torque"])
# n_outlier_percent = detUtil.detectOutlier(pred_forecast_for_dmg_test, df_test_dmg_3mm_edited["Torque"], n_max_score)
n_outlier_percent = detUtil.detectOutlier(pred_forecast, df_test_3mm_edited["Torque"], n_max_score)

print(f"{n_outlier_percent} % detected")  # give back the percentage of anomalies detected in the given data based on the forecast of a fitted model and the computed anomaly threshhold 