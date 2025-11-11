#put a method in here starting it all?
from . import OutlierDetectorUtil as detUtil
from . import run as run
from pathlib import Path

# read the given data sets this is part of the main
script_dir = Path(__file__).parent

file_path_3mm_noDMG_edited = script_dir.parent / "testData" /"3mm"/"3mm_NoDMG_20092025_edited.csv" #that should also be in createTimeSeriesDataFrame, the read csv file should always be turned into a dataframe 
# file_path_3mm_DMG_edited =  script_dir.parent / "testData" / "3mm" / "3mm_DMG_20092025_edited.csv"
file_path_3mm_DMG_edited =  script_dir.parent / "testData" / "3mm" / "3mm_DMG_05_11_2025_edited.csv"  

# Everything at this point needs to go into its seperate thing
#torque data needs to be put into absolute values due to the machine giving inverted data
run.run(file_path_3mm_noDMG_edited, file_path_3mm_DMG_edited, "Torque_ax8", "Torque", 39, 0.05, "ADF", bAbs= True) #nSeasons how many seasons with obs there are, to get observ per season