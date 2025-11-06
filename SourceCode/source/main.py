#put a method in here starting it all?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import UtilsDataFrame as udf
from . import STLUtils as stlu
from . import StationaryUtils as statutil
from pathlib import Path
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda
import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as STL

# read the given data sets
script_dir = Path(__file__).parent

file_path_3mm_noDMG_edited = script_dir.parent / "testData" /"3mm"/"3mm_NoDMG_20092025_edited.csv" #that should also be in createTimeSeriesDataFrame, the read csv file should always be turned into a dataframe 
#torque data needs to be put into absolute values due to the machine giving inverted data
df_train_3mm_edited = udf.getTrainSet(pd.read_csv(file_path_3mm_noDMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()
df_test_3mm_edited = udf.getTestSet(pd.read_csv(file_path_3mm_noDMG_edited),"Zaehler", "Torque_ax8", "Torque", True).abs()

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

stl_3mm_edited = STL.STL(df_train_3mm_edited["Transformed"], period=6, seasonal=39)
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
# Stationary checks; if trend then constant and trend, otherwise constant, schwart for lags and implment AIC for making it more precise

if bTrending:
    nStationary = stlu.getStationary(df_train_3mm_edited["Transformed"], "ct", "5%", "ADF")
    # as in the trend case, the series is already expected to be stationary, adf here should just decide upon if it is RWWD (H_0) or DT (H_0 rejected)
    # ergo if the series should be detrended or differenced; this might be a bit confusing when looking at the implementation and the var names
    if nStationary == 0:
        print("detrend") # detrend before going further
    elif nStationary == 1:
        #if differenced just do so when implementing ARIMA
        print("difference")
    elif nStationary == -1:
        print(nStationary) #no idea yet


# if stlu.getStationary(bTrending, df_train_3mm_edited["Transformed"]): put in ct otherwise c or in extreme cases with zero mean n, type of test
#     bStationary = True

# trend, difference, stationary
# else:
#     bStationay = False
# print(bStationay)