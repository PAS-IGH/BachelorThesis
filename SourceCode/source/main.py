"""
A wrapper for the run functions. 
This module performs the following operations:
    1. Extracts the sets
    2. Hands them over to run functions
"""
from . import OutlierDetectorUtil as detUtil
from . import run as run
from pathlib import Path

script_dir = Path(__file__).parent

file_path_3mm_noDMG_edited = script_dir.parent / "testData" /"3mm"/"3mm_NoDMG_20092025_edited.csv" 
file_path_3mm_DMG_edited =  script_dir.parent / "testData" / "3mm" / "3mm_DMG_05_11_2025_edited.csv"

file_path_5mm_noDMG_edited = script_dir.parent / "testData" /"5mm"/"5mm_noDMG_05_11_2025_edited.csv" 
file_path_5mm_DMG_edited =  script_dir.parent / "testData" / "5mm" / "5mm_DMG_05_11_2025_edited.csv" 

run.run(file_path_3mm_noDMG_edited, file_path_3mm_DMG_edited, "Torque_ax8", "Torque", 39, 0.05, "ADF",script_dir, bAbs= True, str_FolderName="3mm_Analysis") 
run.run(file_path_5mm_noDMG_edited, file_path_5mm_DMG_edited, "Torque_ax8", "Torque", 39, 0.05, "ADF",script_dir, bAbs= True, str_FolderName="5mm_Analysis") 