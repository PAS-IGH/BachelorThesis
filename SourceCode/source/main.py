#put a method in here starting it all?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import UtilsDataFrame as udf
from pathlib import Path

# read the given data sets
script_dir = Path(__file__).parent

file_path_300mm_noDMG_edited = script_dir.parent / "testData" /"300mm"/"300mm_NoDMG_20092025_edited.csv" #that should also be in createTimeSeriesDataFrame, the read csv file should always be turned into a dataframe 
# df_300mm_edited = udf.getTrainTestSet(pd.read_csv(file_path_300mm_noDMG_edited), "Zaehler")
df_train_300mm_edited = udf.getTrainSet(pd.read_csv(file_path_300mm_noDMG_edited), "Zaehler", "Torque_ax8", "Torque", True)
df_test_300mm_edited = udf.getTestSet(pd.read_csv(file_path_300mm_noDMG_edited),"Zaehler", "Torque_ax8", "Torque", True)

print(df_train_300mm_edited) #sanity checks
print(df_test_300mm_edited) #sanit checks

#now onto detrending, remember get the lambda first then and save it as a global var for later detransforming after ARIMA modeling