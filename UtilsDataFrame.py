import pandas as pd
import numpy as np

def createTimeSeriesDataFrame(data, depVar,renameDepVar, isGermany = True):
    # The reason why timestamp is "uninteristing" as the index is due to drill length already dictating the equidistance rather then the timestamp
    dataFrame = data[[depVar]].copy() # I like not mutating the original dataset; leave it as it were and create a new one with the wanted independent variable
    if renameDepVar:
        dataFrame.rename(columns={depVar:renameDepVar}, inplace=True)
    
    if isGermany == True: 
        dataFrame[renameDepVar] = dataFrame[renameDepVar].str.replace(",", ".", regex=False) # German floating point numbers are written with a "," instead of "."
                                                                                                # this fact tends to crash methods
    
    dataFrame[renameDepVar] = pd.to_numeric(dataFrame[renameDepVar], errors="raise") # turns all values in the renamDepVar coloumn numeric and enforeces NaN for 
                                                                                     # errors
    return dataFrame

# testData_01_power_df = createTimeSeriesDataFrame(testData_01, "Strom", "Power", True)
# print(testData_01_power_df)