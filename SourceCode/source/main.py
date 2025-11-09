#put a method in here starting it all?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import UtilsDataFrame as udf
from . import STLUtils as stlu
from . import StationaryUtils as statutil
from . import ACF_PACFUtils as corrUtil
from . import ARIMAUtils as arimaUtil
from . import OutlierDetectorUtil as detUtil
from pathlib import Path
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda
import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.robust.scale import mad
from sklearn.metrics import mean_absolute_error

# read the given data sets this is part of the main
script_dir = Path(__file__).parent

file_path_3mm_noDMG_edited = script_dir.parent / "testData" /"3mm"/"3mm_NoDMG_20092025_edited.csv" #that should also be in createTimeSeriesDataFrame, the read csv file should always be turned into a dataframe 
file_path_3mm_DMG_edited =  script_dir.parent / "testData" / "3mm" / "3mm_DMG_20092025_edited.csv" 

# Everything at this point needs to go into its seperate thing
#torque data needs to be put into absolute values due to the machine giving inverted data
df_train_3mm_edited = udf.getTrainSet(pd.read_csv(file_path_3mm_noDMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()
df_test_3mm_edited = udf.getTestSet(pd.read_csv(file_path_3mm_noDMG_edited),"Zaehler", "Torque_ax8", "Torque", True).abs()
df_train_dmg_3mm_edited = udf.getTrainSet(pd.read_csv(file_path_3mm_DMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()
df_test_dmg_3mm_edited = udf.getTestSet(pd.read_csv(file_path_3mm_DMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()

# print(df_train_3mm_edited) #sanity checks
# print(df_test_3mm_edited) #sanity checks

#now onto detrending, remember get the lambda first then and save it as a global var for later detransforming after ARIMA modeling
# Lambda by guerrero, transform by box cox
opt_lambda_3mm_NoDmg = boxcox_lambda(df_train_3mm_edited["Torque"], method="guerrero", season_length=38)
df_train_3mm_edited["Transformed"] = boxcox(df_train_3mm_edited["Torque"], opt_lambda_3mm_NoDmg)


# print(opt_lambda_3mm_NoDmg)
# print(df_train_3mm_edited["Transformed"])
plt.figure(figsize=(10,6))
plt.plot(df_train_3mm_edited["Torque"], alpha=0.7)
plt.plot(df_train_3mm_edited["Transformed"], alpha=0.7)

# plt.show()

#STL implement. Get Graph and especially trend

stl_3mm_edited = STL.STL(df_train_3mm_edited["Transformed"], period=38, seasonal=7)
stl_fitted = stl_3mm_edited.fit()

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
# plt.show()

# implment hyndmans formula for trend detection
# print(stl_fitted.trend) #sanity check
# print(stlu.getTrendStrength(stl_fitted.trend, stl_fitted.resid))

# A trend strength more than 40 percent should be significant enough in deciding if 
# there is a trend there, important for ADF and making time series stationary
if stlu.getTrendStrength(stl_fitted.trend, stl_fitted.resid) >= 0.4:
    bTrending = True
else:
    bTrending = False
# print(bTrending)
# Stationary checks; if trend then constant and trend, otherwise constant
b_Difference = False
b_Detrend = False
b_Stationary = False

if bTrending:
    nStationary = statutil.getStationary(df_train_3mm_edited["Transformed"], "ct", 0.05, "ADF")
    # as in the trend case, the series is already expected to be stationary, adf here should just decide upon if it is RWWD (H_0) or DT (H_0 rejected)
    # ergo if the series should be detrended or differenced; this might be a bit confusing when looking at the implementation and the var names
    if nStationary == 0:
        b_Detrend = True
        print("detrend") # detrend before going further
    elif nStationary == 1:
        #if differenced just do so when implementing ARIMA
        b_Difference = True
        print("difference")
    elif nStationary == -1:
        print(nStationary) #no idea yet

else:
    # as there is no trend a rejeted H_0 means that a series is already stationary
    nStationary = statutil.getStationary(df_train_3mm_edited["Transformed"], "c", 0.05, "ADF")
    if nStationary == 0:
        b_Stationary = True
        print("do nothing") # its stationary
    elif nStationary == 1:
        #if differenced just do so when implementing ARIMA
        b_Difference = True
        print("difference")
    elif nStationary == -1:
        print(nStationary) #no idea yet


# plt.show()
# print(df_train_3mm_edited["Transformed"].diff())
# ok take what has been learned about the stationarity and build a ACF/PACF module to determine which lag is optimal; return a tupel for p and q, d is already set thrugh bDifference
# plot in the run code, get the results with the module, 
plot_acf(df_train_3mm_edited["Transformed"].diff().dropna(), lags=76)
# plot_acf(df_train_3mm_edited["Transformed"], lags=76)
# plot_pacf(df_train_3mm_edited["Transformed"], lags=76)
# plot_pacf(df_train_3mm_edited["Transformed"].diff().dropna(), lags=76)
# plt.show()

p, d, q = corrUtil.getARIMA_Params(df_train_3mm_edited["Transformed"], 76, 0.05, b_Detrend, b_Stationary)
# print(p)
# print(d)
# print(q)

# model = ARIMA(df_train_3mm_edited["Transformed"], order=(p , d , 2))

model_fit = arimaUtil.getOptimalModel(df_train_3mm_edited["Transformed"], p, d, q) 
print(model_fit.summary()) #diagnostic with hetereoscedesticity. plot it maybe and jarque bera
#forecast into leng(testSet), transform with inverse boxcox before with the lambda computed above

pred_forecast = arimaUtil.getForecast(model_fit, len(df_test_3mm_edited["Torque"]), opt_lambda_3mm_NoDmg)
pred_forecast_for_dmg_train = arimaUtil.getForecast(model_fit, len(df_train_dmg_3mm_edited["Torque"]), opt_lambda_3mm_NoDmg)
pred_forecast_for_dmg_test = arimaUtil.getForecast(model_fit, len(df_test_dmg_3mm_edited["Torque"]), opt_lambda_3mm_NoDmg)
#MAE: to show how good the model performs
# print(pred_forecast)
# print(pred_forecast_for_dmg_train)
# mae = mean_absolute_error(df_test_3mm_edited["Torque"], pred_forecast)
# print(mae)
# mae_2 = mean_absolute_error(df_train_dmg_3mm_edited["Torque"], pred_forecast_for_dmg)
# print(mae_2)




















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
n_max_score = detUtil.getAnomalyThreshold(df_train_dmg_3mm_edited["Torque"], pred_forecast_for_dmg_train)
print(f"this {n_max_score} max")

# Simulate to show how the anomaly detection works
# print(pred_forecast_for_dmg_test)
# print(df_test_dmg_3mm_edited["Torque"])
# n_outlier_percent = detUtil.detectOutlier(pred_forecast_for_dmg_test, df_test_dmg_3mm_edited["Torque"], n_max_score)
n_outlier_percent = detUtil.detectOutlier(pred_forecast, df_test_3mm_edited["Torque"], n_max_score)

print(f"{n_outlier_percent} % detected")  # give back the percentage of anomalies detected in the given data based on the forecast of a fitted model and the computed anomaly threshhold 