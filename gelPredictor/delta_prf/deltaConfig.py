#! /usr/bin/env python3

import sys
from os import getcwd
from os.path import basename
import configparser
from pathlib import Path

PROGRAM_NAME_DELTAPRF   = 'deltaProfiles.py'

if basename(sys.argv[0]) == PROGRAM_NAME_DELTAPRF:

    FORECASTING_OBJECT = "Delta_1"
    FORECASTING_OBJECT_TITLE = "Delta_1"
else:
    FORECASTING_OBJECT = "Forecating_object"
    FORECASTING_OBJECT_TITLE = "Forecating_object"

LOG_FOLDER_NAME = "log_{}".format(FORECASTING_OBJECT)
MAIN_SYSTEM_LOG = "main_{}".format(FORECASTING_OBJECT)
SERVER_LOG      = "server_{}".format(FORECASTING_OBJECT)
CHART_LOG       = "chart_{}".format(FORECASTING_OBJECT)
SHRT_CHART_LOG  = "shrt_chart_{}".format(FORECASTING_OBJECT)
HMM_CHART_LOG   = "hmm_chart_{}".format(FORECASTING_OBJECT)
WV_IMAGE_LOG    = "Wavelet_image {}".format(FORECASTING_OBJECT)

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
PATH_WV_IMAGES             = Path(PATH_LOG_FOLDER / WV_IMAGE_LOG)
TRAIN_FOLDER               = Path(LOG_FOLDER_NAME / Path("TrainPath"))
AUX_TRAIN_FOLDER           = Path(LOG_FOLDER_NAME / Path("AuxTrain"))
TEST_FOLDER                = Path(LOG_FOLDER_NAME / Path("TestPath"))


PATH_LOG_FOLDER.mkdir(            parents = True, exist_ok = True)
PATH_REPOSITORY.mkdir(            parents = True, exist_ok = True)
PATH_DATASET_REPOSITORY.mkdir(    parents = True, exist_ok = True)
PATH_DESCRIPTOR_REPOSITORY.mkdir( parents = True, exist_ok = True)
PATH_CHARTS.mkdir(                parents = True, exist_ok = True)
PATH_SHRT_DATASETS.mkdir(         parents = True, exist_ok = True)
PATH_SHRT_MODELS.mkdir(           parents = True, exist_ok = True)
PATH_SHRT_CHARTS.mkdir(           parents = True, exist_ok = True)
PATH_HMM_CHARTS.mkdir(            parents = True, exist_ok = True)
PATH_WV_IMAGES.mkdir(             parents = True, exist_ok = True)
TRAIN_FOLDER.mkdir(               parents = True, exist_ok = True)
AUX_TRAIN_FOLDER.mkdir(           parents = True, exist_ok = True)
TEST_FOLDER.mkdir(                parents = True, exist_ok = True)

MAX_LOG_SIZE_BYTES=5 * 1024 * 1024
BACKUP_COUNT=2

TS_TIMESTAMP_LABEL = "Date Time"

# common init
PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "DeltaForProfiling.csv").with_suffix(".csv")
TS_NAME            = "Delta_1"
TS_GENERATION_NAME = ""
TS_DEMAND_NAME     = ""
# TS hypeparameters
SAMPLING = 5 * 60  #
SEGMENT_SIZE = 288  # 96  # size of segments over ts. The segments are being transformed to scalograms.
OVERLAP = 0  # sigments overlap. If 0 then the sigments are adjacent and number of segments over TS
N_STEPS = 12  # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
NUM_SCALES = 12  # scale for wavelet
DATA_NORMALIZATION = 'norm'  # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
CONTINUOUS_WAVELET = 'mexh'  # see in imported 'pywt'-package
# CNN Hyperparams
NUM_CLASSES = 11
DETREND = False

if basename(sys.argv[0]) == PROGRAM_NAME_DELTAPRF :
    PATH_TO_DATASET= Path(PATH_DATASET_REPOSITORY / "DeltaForProfiling.csv").with_suffix(".csv")
    TS_NAME            = "Delta_1"
    TS_GENERATION_NAME = ""
    TS_DEMAND_NAME     = ""
    # TS hypeparameters
    SAMPLING = 5 * 60  #
    SEGMENT_SIZE = 288  # 96  # size of segments over ts. The segments are being transformed to scalograms.
    OVERLAP = 0  # sigments overlap. If 0 then the sigments are adjacent and number of segments over TS
    N_STEPS = 12  # size of sliding block over ts. They form learning-data inputs for multilayer and LSTM
    NUM_SCALES = 32  # scale for wavelet
    DATA_NORMALIZATION = 'norm'  # 'norm' -normalazation 0.0-1.0; 'stat'-statistical nomalization; 'other' -no normalization
    CONTINUOUS_WAVELET = 'mexh'  # see in imported 'pywt'-package
    # CNN Hyperparams
    NUM_CLASSES = 11
    DETREND = False

else:
    pass


ORDINAL_PROFILE = 0
TYPICAL_PROFILE = 1
if __name__ == "__main__":
    pass
