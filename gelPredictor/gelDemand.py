#!/usr/bin/env python3

"""       Short-Term and Very Short-Term forecasting for renewable electricity loading.

    Discretization of the studied and predicted processes in green renewable power generation is performed with
a period of a few minutes (T).  Accordingly, the very short-term (VST) forecast horizon for this discretization T is
1*T, 2*T, 3*T, 4*T, i.e. tens of minutes. On the contrary, the short-term (ST) forecasting horizon is hours or even
days, and medium-term (MT) horizon a month or a quarter.
    For each forecast horizon VST, ST, MT different models are used and possibly aggregation of historical observation
time series (TS). G(reen)E(lectricity)L(oading) Predictor is a tool for VST and ST forecasting based on using
A(rtifical)N(eural)N(et) , H(idden)M(arkov)M(odels), statistical classification  and  image  distingush methods.

 """

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.dau import Dataset
from src.cnn import  CNN
from src.vshtrm import VeryShortTerm
from sys_util.parseConfig import LOG_FOLDER_NAME, MAIN_SYSTEM_LOG, SERVER_LOG, CHART_LOG, \
PATH_ROOT_FOLDER, PATH_LOG_FOLDER , PATH_MAIN_LOG , PATH_SERVER_LOG, PATH_CHART_LOG , \
PATH_REPOSITORY , PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY, \
MAX_LOG_SIZE_BYTES, BACKUP_COUNT, PATH_TO_DATASET, DATA_NORMALIZATION, OVERLAP, N_STEPS, CONTINUOUS_WAVELET, \
NUM_CLASSES, NUM_SCALES, SAMPLING, TS_NAME, TS_TIMESTAMP_LABEL, SEGMENT_SIZE, PATH_SHRT_DATASETS,printInfo
from sys_util.utils import setOutput, plotClusters
from src.drive import drive_all_classes
from src.flow_control import cntrPath,medTrainPath, shrtTrainPath, hmmTrainPath, cntrPath_for_Demand
from src.DatasetDemand import DatasetDemand
# ------------------------------------------------------------------------------------------------------------------

size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)

logger.info(printInfo())
# ------------------------------------------------------------------------------------------------------------------

_, log_folder, chart_log = setOutput(model_repository=PATH_REPOSITORY,
                                     log_folder=PATH_LOG_FOLDER, chart_log=PATH_CHART_LOG)
def main(argc,argv)->int:
    """ Main function """

    ds, X,Y = cntrPath_for_Demand()


    return 0

if __name__ == '__main__':
   pass

   nret = main(None, None)

   logger.info('\n\n==============================================================================================')
   logger.info('\n==============================================================================================')
   logger.info('\n\n\n\n\n\n')