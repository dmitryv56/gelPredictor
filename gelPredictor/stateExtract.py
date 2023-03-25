#!/usr/bin/env python3

"""       State extraction from Imbabalance time series (TS) based on Diesel (Power Generation) and Demand (Consumer
demand) Ttime series.
   TS-s acqured in ElHierro_2012_2014 dataset. The discretization is performed witha a periood of 10 minutes (600
   seconds).
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
from sys_util.utils import setOutput
from src.drive import drive_all_classes
from src.flow_control import cntrPath_no_segm,medTrainPath, shrtTrainPath, hmmTrainPath
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

def main(argc,argv)->int:
    """ Main function """

    _, log_folder, chart_log = setOutput(model_repository=PATH_REPOSITORY,
                                         log_folder=PATH_LOG_FOLDER, chart_log=PATH_CHART_LOG)


    """ Dataset is Control path """
    ds, X, Y = cntrPath_no_segm()

    logger.info('\n\n==============================================================================================')
    logger.info('\n==============================================================================================')
    logger.info('\n\n\n\n\n\n')

    return 0

if __name__ == '__main__':
   pass

   nret = main(None, None)
