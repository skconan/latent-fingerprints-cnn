"""
    File name: constants.py
    Author: skconan
    Date created: 2019/10/12
    Python Version: 3.7
"""

import os

PATH_DATABASE = r"E:\KSIP\Database"
PATH_NIST14 = PATH_DATABASE + r"\NIST14"
PATH_NIST14_BINARY = PATH_DATABASE + r"\NIST14_Binary"

PATH_WS = os.path.dirname(os.path.abspath(__file__)).replace("source", "")[:-1] 
PATH_DATASET = PATH_WS + r"\dataset"
PATH_INPUT = PATH_DATASET + r"\input"
PATH_TARGET = PATH_DATASET + r"\target"

PATH_INPUT_SEG = PATH_INPUT + r"\segmented"
PATH_TARGET_SEG = PATH_TARGET + r"\segmented"

PATH_TARGET_SEG_STFT = PATH_TARGET + r"\segmented_stft"

PATH_INPUT_TRAIN_STFT = PATH_INPUT + r"\train_stft"
PATH_INPUT_VALID_STFT = PATH_INPUT + r"\valid_stft"
PATH_TARGET_TRAIN_STFT = PATH_TARGET + r"\train_stft"
PATH_TARGET_VALID_STFT = PATH_TARGET + r"\valid_stft"
