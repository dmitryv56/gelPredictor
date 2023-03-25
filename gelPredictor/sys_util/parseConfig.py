#! /usr/bin/env python3

import sys
from os import getcwd
from os.path import basename
import configparser
from pathlib import Path

PROGRAM_NAME_GELPREDICTOR   = 'gelPredictor.py'
PROGRAM_NAME_STATEEXTRACTOR = 'stateExtract.py'
PROGRAM_NAME_GELANALYZER    = 'gelAnalyzer.py'
PROGRAM_NAME_GELDEMAND      = 'gelDemand.py'

if basename(sys.argv[0]) == PROGRAM_NAME_GELPREDICTOR:

    FORECASTING_OBJECT = "Power_Solar_5_20"
    FORECASTING_OBJECT_TITLE = "Power Solar Sunshine 5-20"
elif basename(sys.argv[0]) == PROGRAM_NAME_STATEEXTRACTOR :
    FORECASTING_OBJECT = "ElHierro2012_2014"
    FORECASTING_OBJECT_TITLE = "El-Hierro (2012-2014)"
elif basename(sys.argv[0]) == PROGRAM_NAME_GELANALYZER :
    FORECASTING_OBJECT = "Demand"
    FORECASTING_OBJECT_TITLE = "Demand El-Hierro (2012-2014)"
elif basename(sys.argv[0]) == PROGRAM_NAME_GELDEMAND :
    # FORECASTING_OBJECT = "Demand_3state"
    # FORECASTING_OBJECT_TITLE = "Demand El-Hierro (2012-2014)"
    FORECASTING_OBJECT = "PowerSolar_detrend_3state"
    FORECASTING_OBJECT_TITLE = "Detrend Power Solar(2021) "
else:
    FORECASTING_OBJECT = "Forecating_object"
    FORECASTING_OBJECT_TITLE = "Forecating_object"

# Paths and e.t.c
#German Electricity
# LOG_FOLDER_NAME="log_electricity_5"
# MAIN_SYSTEM_LOG="main_electricity_5"
# SERVER_LOG="server"
# CHART_LOG="chart_electricity_5"

# Power Solar
LOG_FOLDER_NAME = "log_{}".format(FORECASTING_OBJECT)
MAIN_SYSTEM_LOG = "main_{}".format(FORECASTING_OBJECT)
SERVER_LOG      = "server_{}".format(FORECASTING_OBJECT)
CHART_LOG       = "chart_{}".format(FORECASTING_OBJECT)
SHRT_CHART_LOG  = "shrt_chart_{}".format(FORECASTING_OBJECT)
HMM_CHART_LOG   = "hmm_chart_{}".format(FORECASTING_OBJECT)

if sys.platform == 'win32':
    PATH_ROOT_FOLDER = Path(Path(Path(getcwd()).drive))
elif sys.platform == 'linux':
    PATH_ROOT_FOLDER = Path(Path(__file__).parent.parent.absolute())
else:
    print("Unknown platform. Exit...")
    sys.exit(1)

PATH_LOG_FOLDER            = Path(PATH_ROOT_FOLDER / LOG_FOLDER_NAME)
PATH_MAIN_LOG              = Path(PATH_LOG_FOLDER / MAIN_SYSTEM_LOG).with_suffix(".log")
PATH_SERVER_LOG            = Path(PATH_LOG_FOLDER / SERVER_LOG).with_suffix(".log")
PATH_CHART_LOG             = Path(PATH_LOG_FOLDER / CHART_LOG).with_suffix(".log")

PATH_REPOSITORY            = Path(PATH_ROOT_FOLDER / "model_Repository")
PATH_DATASET_REPOSITORY    = Path(PATH_ROOT_FOLDER / "dataset_Repository")
PATH_DESCRIPTOR_REPOSITORY = Path(PATH_ROOT_FOLDER / "descriptor_Repository")
PATH_CHARTS                = Path(PATH_LOG_FOLDER / CHART_LOG)
PATH_SHRT_DATASETS         = Path(PATH_LOG_FOLDER / "short_term_Repository")
PATH_SHRT_MODELS           = Path(PATH_LOG_FOLDER / "short_term_model_Repository")
PATH_SHRT_CHARTS           = Path(PATH_LOG_FOLDER / SHRT_CHART_LOG)
PATH_HMM_CHARTS            = Path(PATH_LOG_FOLDER / HMM_CHART_LOG)

PATH_LOG_FOLDER.mkdir(            parents = True, exist_ok = True)
PATH_REPOSITORY.mkdir(            parents = True, exist_ok = True)
PATH_DATASET_REPOSITORY.mkdir(    parents = True, exist_ok = True)
PATH_DESCRIPTOR_REPOSITORY.mkdir( parents = True, exist_ok = True)
PATH_CHARTS.mkdir(                parents = True, exist_ok = True)
PATH_SHRT_DATASETS.mkdir(         parents = True, exist_ok = True)
PATH_SHRT_MODELS.mkdir(           parents = True, exist_ok = True)
PATH_SHRT_CHARTS.mkdir(           parents = True, exist_ok = True)
PATH_HMM_CHARTS.mkdir(            parents = True, exist_ok = True)

MAX_LOG_SIZE_BYTES=5 * 1024 * 1024
BACKUP_COUNT=2

# PATH_TO_DATASET='~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero2018_2022.csv'
# PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "ElHiero_24092020_27102020.csv")
# German Electricity
# PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "Electricity_generation_in_Germany_2020_2022").with_suffix(".csv")

# Power Solar
if basename(sys.argv[0]) == PROGRAM_NAME_GELPREDICTOR :
    PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "PowerSolar_2021_5_20").with_suffix(".csv")
    TS_NAME            = "Power_Solar"
    TS_GENERATION_NAME = ""
    TS_DEMAND_NAME     = ""
    # TS hypeparameters
    SAMPLING = 60 * 60  # German Electricity15 * 60
    SEGMENT_SIZE = 16  # 96  # size of segments over ts. The segment are being transformed to scalograms.
    OVERLAP = 0  # sigments overlap. If 0 then the sigments are adjacent and number of segments over TS
    # is [n/n_step].
    # If 0 < overlap < n_step then  number of segment is the following sum
    #  while (n-k*overlap<=n_step):
    #         nn=nn+[n-k*overlap)/n_step] , where k=0,1,..
    N_STEPS = 12  # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
    # neural nets.
    NUM_SCALES = 12  # scale for wavelet
    DATA_NORMALIZATION = 'norm'  # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
    CONTINUOUS_WAVELET = 'mexh'  # see in imported 'pywt'-package
    # CNN Hyperparams
    NUM_CLASSES = 11
elif basename(sys.argv[0]) == PROGRAM_NAME_STATEEXTRACTOR :
    PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "ElHierro_2012_2014").with_suffix(".csv")
    TS_NAME            = "Imbalance"
    TS_GENERATION_NAME = "Diesel"
    TS_DEMAND_NAME     = "Demand"
    # TS hypeparameters
    SAMPLING = 10 * 60  #
    SEGMENT_SIZE = 144  #
    OVERLAP = 0  # sigments overlap. If 0 then the sigments are adjacent and number of segments over TS
    # is [n/n_step].
    # If 0 < overlap < n_step then  number of segment is the following sum
    #  while (n-k*overlap<=n_step):
    #         nn=nn+[n-k*overlap)/n_step] , where k=0,1,..
    N_STEPS = 32  # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
    # neural nets.
    NUM_SCALES = 12  # scale for wavelet
    DATA_NORMALIZATION = 'norm'  # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
    CONTINUOUS_WAVELET = 'mexh'  # see in imported 'pywt'-package
    # CNN Hyperparams
    NUM_CLASSES = 11
elif basename(sys.argv[0]) == PROGRAM_NAME_GELANALYZER :
    PATH_TO_DATASET = Path(PATH_DATASET_REPOSITORY / "ElHierro_2012_2014").with_suffix(".csv")
    TS_NAME = "Demand"
    TS_GENERATION_NAME = "Diesel"
    TS_DEMAND_NAME = "Day_In_Week"
    # TS hypeparameters
    SAMPLING = 10 * 60  #
    SEGMENT_SIZE = 144  #
    OVERLAP = 0  # sigments overlap. If 0 then the sigments are adjacent and number of segments over TS
    # is [n/n_step].
    # If 0 < overlap < n_step then  number of segment is the following sum
    #  while (n-k*overlap<=n_step):
    #         nn=nn+[n-k*overlap)/n_step] , where k=0,1,..
    N_STEPS = 32  # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
    # neural nets.
    NUM_SCALES = 12  # scale for wavelet
    DATA_NORMALIZATION = 'norm'  # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
    CONTINUOUS_WAVELET = 'mexh'  # see in imported 'pywt'-package
    # CNN Hyperparams
    NUM_CLASSES = 11
elif basename(sys.argv[0]) == PROGRAM_NAME_GELDEMAND :
    # PATH_TO_DATASET = Path(PATH_DATASET_REPOSITORY / "ElHierro_2012_2014").with_suffix(".csv")
    PATH_TO_DATASET = Path(PATH_DATASET_REPOSITORY / "SolarPlantPowerGen_21012020_Light").with_suffix(".csv")
    # TS_NAME = "Demand"
    TS_NAME = "Power_Solar"
    TS_GENERATION_NAME = "Diesel"
    TS_DEMAND_NAME = "Day_In_Week"
    # TS hypeparameters
    SAMPLING = 10 * 60  # 10 * 60  #
    SEGMENT_SIZE = 72  # 144  #
    OVERLAP = 0
    N_STEPS = 36  # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
    # neural nets.
    NUM_SCALES = 16  # scale for wavelet
    DATA_NORMALIZATION = 'other'  # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
    CONTINUOUS_WAVELET = 'mexh'  # see in imported 'pywt'-package
    # CNN Hyperparams
    NUM_CLASSES = 3
else:
    PATH_TO_DATASET = Path(PATH_DATASET_REPOSITORY / "ElHierro_2012_2014").with_suffix(".csv")
    TS_NAME = "Imbalance"
    TS_GENERATION_NAME = "Diesel"
    TS_DEMAND_NAME = "Demand"
    # TS hypeparameters
    SAMPLING = 10 * 60  #
    SEGMENT_SIZE = 144  #
    OVERLAP = 0  # sigments overlap. If 0 then the sigments are adjacent and number of segments over TS
    # is [n/n_step].
    # If 0 < overlap < n_step then  number of segment is the following sum
    #  while (n-k*overlap<=n_step):
    #         nn=nn+[n-k*overlap)/n_step] , where k=0,1,..
    N_STEPS = 32  # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
    # neural nets.
    NUM_SCALES = 12  # scale for wavelet
    DATA_NORMALIZATION = 'norm'  # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
    CONTINUOUS_WAVELET = 'mexh'  # see in imported 'pywt'-package
    # CNN Hyperparams
    NUM_CLASSES = 10



TS_TIMESTAMP_LABEL = "Date Time"

# CNN Hyperparams

BATCH_SIZE   = 64
EPOCHS       = 20 #10
# NUM_CLASSES  = 11
ALFA_RELU    = 0.1
DROPOUT      = 0.25
DENSE_INPUT  = 128
NUM_KERNELS  = 32
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.28
COMPRESS     = 'pca'   # or 'no'
N_COMPONENTS = 2

# TS
# German Electricity
#TS_NAME = "Wind_offshore_50Hertz"


# Short-Term forecasting ANN
SHRT_ANN_MODEL_TYPES     = ['MLP', 'CNN', 'LSTM']
SHRT_ANN_MODEL_DICT      = {'MLP':  [(0,'mlp_1'), (1, 'mlp_2')], \
                            'CNN':  [(2, 'univar_cnn')], \
                            'LSTM': [(3, 'vanilla_lstm'), (4, 'stacked_lstm'), (5, 'bidir_lstm')]}
SHRT_EPOCHS              = 10
SHRT_HIDDEN_LAYERS       = 128
SHRT_DROPOUT             = 0.25
SHRT_FEATURES            = 1
SHRT_UNITS               = 64
SHRT_MIN_TRAIN_DATA_SIZE = 2 * N_STEPS
SHRT_TRAIN_PART          = 0.7
SHRT_VAL_PART            = 1.0 - SHRT_TRAIN_PART

# Imbalance state rules
STATE_0 = 0
STATE_DEMAND = 1
STATE_GENERATION = 2
STATE_0_MARGE = 0.1



def printInfo()->str:
    msg = f"""
    
         Artifical Neural Net (ANN) Medium Term And Very Short-Term Forescting
                  {FORECASTING_OBJECT_TITLE }
                  
    MEDIUM TERM FORECASTING GORIZONT   
Log Folder Name            : {LOG_FOLDER_NAME}
Main System Log            : {MAIN_SYSTEM_LOG}
Server Log                 : {SERVER_LOG}
Chart Log                  : {CHART_LOG}
Path Root Folder           : {PATH_ROOT_FOLDER} 
Path Log Folder            : {PATH_LOG_FOLDER}
Path Main Log              : {PATH_MAIN_LOG}
Path Server Log            : {PATH_SERVER_LOG}
Path Chart Log             : {PATH_CHART_LOG}
Path Med Term ANN Models 
Repository                 : {PATH_REPOSITORY}
Path Dataset Repository    : {PATH_DATASET_REPOSITORY}
Path Descriptor Repository : {PATH_DESCRIPTOR_REPOSITORY}
Path Med Term Charts       : {PATH_CHARTS}
Max. Log Size(Bytes)       : {MAX_LOG_SIZE_BYTES}
Backup Count               : {BACKUP_COUNT}

Path to Dataset            : {PATH_TO_DATASET}
Time Series Name           : {TS_NAME}
Time Series Name(X1)       : {TS_GENERATION_NAME}
Time Series Name(X2)       : {TS_DEMAND_NAME}
Timestamp Label            : {TS_TIMESTAMP_LABEL}

Time Series hyper-parameters
Sampling                   : {SAMPLING} sec
Segment Size(n_step)       : {N_STEPS}
Segment Size for Scalogram : {SEGMENT_SIZE}
Overlap of Segments        : {OVERLAP}
Data Normalization         : {DATA_NORMALIZATION}
Continous Wavelet          : {CONTINUOUS_WAVELET}

        Classification by KMeans method
Num Classes                : {NUM_CLASSES}
Transformation Data        : {COMPRESS}
Num PCA Components         : {N_COMPONENTS}

        Medium Term Prediction Convolution ANN Hyper-parameters Set 
Batch Size                 : {BATCH_SIZE}
Epochs                     : {EPOCHS}
Number Output classes      : {NUM_CLASSES}
Alfa Relu                  : {ALFA_RELU}
Dropout                    : {DROPOUT}
Output (Dense) Layer Size  : {DENSE_INPUT}
Number Kernels             : {NUM_KERNELS}
Ratio (Tran:Valid:Test)    : {TRAIN_RATIO} : {VAL_RATIO} : {round(1.- TRAIN_RATIO - VAL_RATIO, 3)}


     Very Short-Term Forecasting Gorizont
Path Very Short-Term Datasets   : {PATH_SHRT_DATASETS}
Path Very Short-Term ANN Models
Repository                 : {PATH_SHRT_MODELS}
Path Very Short-Term Charts     : {PATH_SHRT_CHARTS}
Artifical Neural Net Types : {SHRT_ANN_MODEL_TYPES}
Used ANN models dict       : {SHRT_ANN_MODEL_DICT}

    Very Short-Term Prediction ANN Hyper-parameters Set
Epochs                     : {SHRT_EPOCHS}
Hidden Layers              : {SHRT_HIDDEN_LAYERS}
Dropout                    : {SHRT_DROPOUT}
Features                   : {SHRT_FEATURES}
Units                      : {SHRT_UNITS}

     Others
Path Hidden Markov Models 
(HMM) Charts               : {PATH_HMM_CHARTS}
Min Train data size        : {SHRT_MIN_TRAIN_DATA_SIZE}
Train part                 : {SHRT_TRAIN_PART}
Validation part            : {SHRT_VAL_PART}
   
                
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


