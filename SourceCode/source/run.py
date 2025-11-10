from . import UtilsDataFrame as udf
import pandas as pd
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda

def run(str_path_undamaged, tr_path_damaged, sDepVar, sRenameVar, nSeasons, nSplit=0.8, bAbs=False):

    # df_train_3mm_edited = udf.getTrainSet(pd.read_csv(str_path_undamaged), "Zaehler", "Torque_ax8", "Torque", True).abs()
    # df_test_3mm_edited = udf.getTestSet(pd.read_csv(file_path_3mm_noDMG_edited),"Zaehler", "Torque_ax8", "Torque", True).abs()
    # df_train_dmg_3mm_edited = udf.getTrainSet(pd.read_csv(file_path_3mm_DMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()
    # df_test_dmg_3mm_edited = udf.getTestSet(pd.read_csv(file_path_3mm_DMG_edited), "Zaehler", "Torque_ax8", "Torque", True).abs()
    
    dict_results = {
        "train_set": None,
        "test_set" : None,
        "transformed_set": None
    }

    df_undamaged_train, df_undamaged_test = udf.getTrainAndTestSet(pd.read_csv(str_path_undamaged), nSeasons, "Torque_ax8", "Torque", True, nSplit)
    if bAbs:
        df_undamaged_train = df_undamaged_train.abs()
        df_undamaged_test = df_undamaged_test.abs()
        
    dict_results["train_set"] = df_undamaged_train
    dict_results["test_set"] = df_undamaged_test

    TimeSeriesAnalysis(dict_results, df_undamaged_train, df_undamaged_test, nSeasons) #produces fitted model for a given set and plotted graphs for analysing
    # outlierdetection() # based on time series analysis detects outliers

def TimeSeriesAnalysis(dict_results, df_undamaged_train, df_undamaged_test, nSeasons):
    opt_lambda = boxcox_lambda(df_undamaged_train, method="guerrero", season_length=nSeasons)
    df_undamaged_train_trans = pd.DataFrame(
        boxcox(df_undamaged_train, opt_lambda),
        index = df_undamaged_train.index,
        columns = df_undamaged_train.columns
    )
    dict_results["transformed_set"] = df_undamaged_train_trans
    print()