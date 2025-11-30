"""
Provides functions around creating the dataframe needed for the time series analysis pipeline.
"""

import pandas as pd
import numpy as np
import math as math

def createTimeSeriesDataFrame(dfData, sDepVar, sRenameDepVar ="y", bGerman = True):
     """ 
     Processes a given dataframe and gives back a series with dependent variable for time series analysis.
     This function performs the following operations:
          1. Copies the given dataframe and renames the dependent variable coloumn
          2. Replaces any ',' with '.' in order to avoid crashing of panda and related methods to the inability of handling German type floating point numbers
          3. Ensure that the dependent variable contains numeric values only
          4. Returns a dataframe for time series analysis
     Args:
          dfData (pd.DataFrame): The data given as a frame 
          sDepVar (str): The name of the desired dependent variable
          sRenamDepVar (str): String for renaming the desired dependent variable
          bGerman (bool): Boolean value in case of German floating point numbers

     Returns:
          dataFrame (pandas.DataFrame): A dataframe with a standardized index as the independent and the coloumn as the dependent variable
     """
     dataFrame = dfData[[sDepVar]].copy() 
     if sRenameDepVar:
          dataFrame.rename(columns={sDepVar:sRenameDepVar}, inplace=True)

     if bGerman == True: 
          dataFrame[sRenameDepVar] = dataFrame[sRenameDepVar].str.replace(",", ".", regex=False) # Handling German floating point numbers

     dataFrame[sRenameDepVar] = pd.to_numeric(dataFrame[sRenameDepVar], errors="raise") # turns all values in the renamDepVar coloumn numeric and enforces NaN for 
                                                                                     # errors
     return dataFrame


def getTrainAndTestSet(dfData, nObsPerSeason, depVar, sRenameDepVar, bGerman, n_Split): 

     """ 
     A wrapper function for returning a training and test set.
     This function performs the following operations:
          1. Computes the season length based on given seasons and amount of observations
          2. Applies createTimeSeriesDataFrame() to get both sets
          3. Returns the set as a tuple

     Args:
          dfData (pd.DataFrame): The data given as a frame 
          sDepVar (str): The name of the desired dependent variable
          sRenamDepVar (str): String for renaming the desired dependent variable
          bGerman (bool): Boolean value in case of German floating point numbers
          n_Split (float): A number indicating the training/test split
     Returns:
          tuple (df_train_set, df_test_set): 
               df_train_set (pandas.DataFrame): The training set
               df_test_set (pandas.DataFrame): The test set
     """
     n_seasons = int(len(dfData) / nObsPerSeason) # get the amount of seasons
     n_seasons_train = int(math.floor(n_seasons * n_Split)) #get the amount of seasons for training set
     n_observ_train = n_seasons_train * nObsPerSeason #get the amount of observations for train set

     #max index for test set is n_observ_train - 1
     df_train_set = createTimeSeriesDataFrame(dfData.iloc[:n_observ_train], depVar, sRenameDepVar, bGerman)
     df_test_set = createTimeSeriesDataFrame(dfData.iloc[n_observ_train:], depVar, sRenameDepVar, bGerman).reset_index(drop=True)
     return df_train_set, df_test_set