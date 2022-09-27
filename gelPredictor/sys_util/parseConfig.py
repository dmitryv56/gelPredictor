#! /usr/bin/env python3

import configparser
from pathlib import Path

# Paths and e.t.c

LOG_FOLDER_NAME="log_electricity_5"
MAIN_SYSTEM_LOG="main_electricity_5"
SERVER_LOG="server"
CHART_LOG="chart_electricity_5"

PATH_ROOT_FOLDER = Path(Path(__file__).parent.parent.absolute())
PATH_LOG_FOLDER = Path(PATH_ROOT_FOLDER/LOG_FOLDER_NAME)
PATH_LOG_FOLDER.mkdir(parents=True, exist_ok=True)
PATH_MAIN_LOG = Path(PATH_LOG_FOLDER/MAIN_SYSTEM_LOG).with_suffix(".log")
PATH_SERVER_LOG = Path(PATH_LOG_FOLDER/SERVER_LOG).with_suffix(".log")
PATH_CHART_LOG = Path(PATH_LOG_FOLDER/CHART_LOG).with_suffix(".log")

PATH_REPOSITORY = Path(PATH_ROOT_FOLDER / "model_Repository")
PATH_REPOSITORY.mkdir(parents=True, exist_ok=True)
PATH_DATASET_REPOSITORY = Path(PATH_ROOT_FOLDER / "dataset_Repository")
PATH_DATASET_REPOSITORY.mkdir(parents=True, exist_ok=True)
PATH_DESCRIPTOR_REPOSITORY = Path(PATH_ROOT_FOLDER / "descriptor_Repository")
PATH_DESCRIPTOR_REPOSITORY.mkdir(parents=True, exist_ok=True)

MAX_LOG_SIZE_BYTES=5 * 1024 * 1024
BACKUP_COUNT=2

# PATH_TO_DATASET='~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero2018_2022.csv'
# PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "ElHiero_24092020_27102020.csv")
PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "Electricity_generation_in_Germany_2020_2022").with_suffix(".csv")

# CNN Hyperparams
BATCH_SIZE = 64
EPOCHS = 10
NUM_CLASSES = 5
ALFA_RELU =0.1
DROPOUT = 0.25
DENSE_INPUT =128
NUM_KERNELS = 32

# TS
TS_NAME = "Wind_offshore_50Hertz"
TS_TIMESTAMP_LABEL = "Date Time"

# TS hypeparameters
SAMPLING = 15 * 60
SEGMENT_SIZE = 98              # size of segments over ts. The segment are being transformed to scalograms.
OVERLAP = 0                    # sigments overlap. If 0 then the sigments are adjacent and number of segments over TS
                               # is [n/n_step].
                               # If 0 < overlap < n_step then  number of segment is the following sum
                               #  while (n-k*overlap<=n_step):
                               #         nn=nn+[n-k*overlap)/n_step] , where k=0,1,..
N_STEPS = 48                   # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
                               # neural nets.
NUM_SCALES = 16                # scale for wavelet
DATA_NORMALIZATION = 'norm'    # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
CONTINUOUS_WAVELET = 'mexh'    # see in imported 'pywt'-package




def printInfo()->str:
    msg = f"""
    
         COMMON SET    
Log Folder Name            : {LOG_FOLDER_NAME}
Main System Log            : {MAIN_SYSTEM_LOG}
Server Log                 : {SERVER_LOG}
Chart Log                  : {CHART_LOG}
Path Root Folder           : {PATH_ROOT_FOLDER} 
Path Log Folder            : {PATH_LOG_FOLDER}
Path Main Log              : {PATH_MAIN_LOG}
Path Server Log            : {PATH_SERVER_LOG}
Path Chart Log             : {PATH_CHART_LOG}
Path Repository            : {PATH_REPOSITORY}
Path Dataset Repository    : {PATH_DATASET_REPOSITORY}
Path Descriptor Repository : {PATH_DESCRIPTOR_REPOSITORY}
Max. Log Size(Bytes)       : {MAX_LOG_SIZE_BYTES}
Backup Count               : {BACKUP_COUNT}

Path to Dataset            : {PATH_TO_DATASET}
Time Series Name           : {TS_NAME}
Timestamp Label            : {TS_TIMESTAMP_LABEL}

Time Series hyper-parameters
Sampling                   : {SAMPLING} sec
Segment Size(n_step)       : {N_STEPS}
Segment Size for Scalogram : {SEGMENT_SIZE}
Overlap of Segments        : {OVERLAP}
Data Normalization         : {DATA_NORMALIZATION}
Continous Wavelet          : {CONTINUOUS_WAVELET}

        Convolution Neural Net Hyper-parameters Set 
Batch Size                 : {BATCH_SIZE}
Epochs                     : {EPOCHS}
Number Output classes      : {NUM_CLASSES}
Alfa Relu                  : {ALFA_RELU}
Dropout                    : {DROPOUT}
Output (Dense) Layer Size  : {DENSE_INPUT}
Number Kernels             : {NUM_KERNELS}

          Other

{funny_header()}
     
    """

    return msg

def funny_header():
    msg=f"""
    
    *****************************************************************
    *                                                               *
    * Our goals are clear, our tasks are define. To work, comrades! *
    *                                                               *
    *****************************************************************

"""
    return msg


