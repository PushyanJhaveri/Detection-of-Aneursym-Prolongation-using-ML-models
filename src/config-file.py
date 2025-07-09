"""
Configuration parameters for the velocity prediction project.
"""

# Data configuration
TRAINING_FILES = [
    'APV2013A1-1-03750_systole.csv',
    'APV2013A2-1-03750_systole.csv',
    'APV2013C1-1-03750_systole.csv',
    'APV2013C2-1-37500_systole.csv',
    'APV2013C3-1-03750_systole.csv',
    'APV2014AA-1-03750_systole.csv',
    'APV2014A1-1-03750_systole.csv',
    'APV2014A2-1-03750_systole.csv',
    'APV2014A3-1-03750_systole.csv',
    'APV2014B-1-03750_systole.csv',
    'APV2014C-1-03750_systole.csv',
    'APV2014D-1-03750_systole.csv',
    'APV2015A1-1-03750_systole.csv',
    'APV2015A2_1-1-03750_systole.csv',
    'APV2015A2-1-03750_systole.csv',
    'APV2015A3-1-03750_systole.csv',
    'APV2015A4_1-1-03750_systole.csv',
    'APV2015A4_2-1-03750_systole.csv',
    'APV2015A4-1-03750_systole.csv',
    'AAD2017A1-1-03750_systole.csv',
    'AAD2017A2-1-03750_systole.csv',
    'AAD2017A3-1-03750_systole.csv',
    'AAD2017A4-1-03750_systole.csv',
    'AAD2017A5-1-03750_systole.csv',
    'AD_2018B_O-1-03750_systole.csv',
    'Draft2_ASCII_AD_2017A_O-1-03750_systole.csv',
    'AAD2018A2-1-03750_systole.csv',
    'AMW2011A-1-03750_systole.csv',
    'AMW2011B-1-03750_systole.csv',
    'AMW2011C-1-03750_systole.csv',
    'AMW2011D-1-03750_systole.csv',
    'AMW2011E-1-03750_systole.csv',
    'AMW2012A-1-03750_systole.csv',
    'AMW2012B-1-03750_systole.csv',
    'AMW2012C-1-03750_systole.csv',
    'AMW2012D-1-03750_systole.csv',
    'AJT2011A1-1-03750_systole.csv',
    'AJT2011A2-1-03750_systole.csv',
    'AJT2011A3-1-03750_systole.csv',
    'AJT2011A4-1-03750_systole.csv',
    'AJT2011A5-1-03750_systole.csv',
    'JT2011OriginalSTL-1-03750_systole.csv',
    'OPv2013_FinalDraft-1-03750_systole.csv',
    'OPV2013C-1-03750_systole.csv',
    'OPV2014-1-03750_systole.csv',
    'OPv2015FinalDraft_Hopefully_Final_2-1-03750_systole.csv'
]

TEST_FILES = [
    'AAAD2018A1-1-03750_systole.csv',
    'AAAMW2012E-1-03750_systole.csv'
]

# Feature and target columns
FEATURE_COLUMNS = slice(1, 4)  # Second, third, and fourth column (x, y, z coordinates)
TARGET_COLUMN = 6            # Velocity column

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model saving paths
MODEL_DIR = 'models'
DATA_DIR = 'data'
