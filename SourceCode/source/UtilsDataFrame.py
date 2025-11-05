import pandas as pd
import numpy as np
import math as math

def createTimeSeriesDataFrame(dfData, sDepVar, sRenameDepVar ="y", bGerman = True):
     """ 
     Process a given data frame and gives back one with a single column and panda indexing (0, ... , N)
     This function performs the following operations:
          1. Copies the given data frame and renames the dependant variable coloumn
          2. Replaces any ',' with '.' in order to avoid crashing of panda and related methods to the inability of handling German type floating point numbers
          3. Ensure that the dependent variable contains numeric values only
 
     Args:
          dfData (pd.DataFrame): The data given as a frame 
          sDepVar (str): The name of the desired dependent variable
          sRenamDepVar (str): String for renaming the desired dependent variable
          bGerman (bool): Expecting German floating point numbers

     Returns:
          A dataframe with the standardized index as independent and the coloumn as the dependent variable
     """
    # The reason why timestamp is "uninteristing" as the index is due to drill length already dictating the equidistance rather then the timestamp
     dataFrame = dfData[[sDepVar]].copy() 
     if sRenameDepVar:
          dataFrame.rename(columns={sDepVar:sRenameDepVar}, inplace=True)

     if bGerman == True: 
          dataFrame[sRenameDepVar] = dataFrame[sRenameDepVar].str.replace(",", ".", regex=False) # Handling German floating point numbers

     dataFrame[sRenameDepVar] = pd.to_numeric(dataFrame[sRenameDepVar], errors="raise") # turns all values in the renamDepVar coloumn numeric and enforces NaN for 
                                                                                     # errors
     return dataFrame


def getTrainSet (dfData, sGroupVar, depVar, sRenameDepVar, bGerman):
     """ 
     Splits the data into a training set based on the 80/20 split (training/test)
     This function operates as follows:
          1. Computes the 80 percent based on the zaehler coloumn which groups singular working processes or rather a period
          2. Decides to get the location upon which the panda vector method cuts off the dataframe
          3. Creates a data frame with the specified dependent value

     Args:
          dfData (pd.DataFrame): The data given as a frame
          sGroupVar (str): Period variable for grouping 
          sDepVar (str): The name of the desired dependent variable
          sRenamDepVar (str): String for renaming the desired dependent variable
          bGerman (bool): Expecting German floating point numbers

     Returns:
          A data frame with approximately 80 percent of the data set
     """
    
     try:
          iAmount= math.floor(int(dfData[sGroupVar].unique()[-1]) * 0.8) # get all unique zaehler 
          iStop = (dfData[sGroupVar] > iAmount).idxmax() #get to the point before the next row has a higher zaehler
          iPos = dfData.index.get_loc(iStop) #get location of that row
          # print(dfData.iloc[:iPos]) #sanity check
          return createTimeSeriesDataFrame(dfData.iloc[:iPos], depVar, sRenameDepVar, bGerman) #get a dataframe with the wanted dependent variable
     except:
        print("oh sHoet")

def getTestSet (dfData, sGroupVar, depVar, sRenameDepVar, bGerman) :

     """ 
     Splits the data into a test set based on the 80/20 split (training/test)
     This function operates as follows:
          1. Computes the 80 percent based on the zaehler coloumn which groups singular working processes or rather a period
          2. Decides to get the location from where the panda vector method starts the new dataframe
          3. Creates a data frame with the specified dependent value
     
     Args:
          dfData (pd.DataFrame): The data given as a frame
          sGroupVar (str): Period variable for grouping 
          sDepVar (str): The name of the desired dependent variable
          sRenamDepVar (str): String for renaming the desired dependent variable
          bGerman (bool): Expecting German floating point numbers

     Returns:
          A data frame with the rest of the data set
     """
     # more specific comments are not needed as this is basically the inverse operation of getTrainSet()
     try:
          iAmount= math.floor(int(dfData[sGroupVar].unique()[-1]) * 0.8)
          iStop = (dfData[sGroupVar] > iAmount).idxmax()
          iPos = dfData.index.get_loc(iStop)
          # print(dfData.iloc[iPos:]) #sanity check
          return createTimeSeriesDataFrame(dfData.iloc[iPos:].reset_index(), depVar, sRenameDepVar, bGerman)
     except:
        print("oh sHoet")  