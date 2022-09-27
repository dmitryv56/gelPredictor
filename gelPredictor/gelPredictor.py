# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.dau import Dataset
from src.cnn import  CNN
from sys_util.parseConfig import LOG_FOLDER_NAME, MAIN_SYSTEM_LOG, SERVER_LOG, CHART_LOG, \
PATH_ROOT_FOLDER, PATH_LOG_FOLDER , PATH_MAIN_LOG , PATH_SERVER_LOG, PATH_CHART_LOG , \
PATH_REPOSITORY , PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY, \
MAX_LOG_SIZE_BYTES, BACKUP_COUNT, PATH_TO_DATASET, DATA_NORMALIZATION, OVERLAP, N_STEPS, CONTINUOUS_WAVELET, \
NUM_CLASSES, NUM_SCALES, SAMPLING, TS_NAME, TS_TIMESTAMP_LABEL, SEGMENT_SIZE, printInfo
from sys_util.utils import setOutput
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


if __name__ == '__main__':
   pass

   _, log_folder, chart_log = setOutput(model_repository=PATH_REPOSITORY,
                                                       log_folder=PATH_LOG_FOLDER, chart_log=PATH_CHART_LOG)
   #


   ds=Dataset(pathTo = PATH_TO_DATASET, ts = TS_NAME, dt = TS_TIMESTAMP_LABEL, sampling= SAMPLING,
              n_steps = N_STEPS, overlap = OVERLAP, continuous_wavelet = CONTINUOUS_WAVELET, norm = DATA_NORMALIZATION,
              num_classes = NUM_CLASSES, model_repository = PATH_REPOSITORY, log_folder = log_folder,
              chart_log = chart_log)

   ds.readDataset()
   ds.data_normalization()
   ds.setTrainValTest()
   ds.__str__()
   ds.createSegmentLst()
   ds.ExtStatesExtraction()
   ds.initHMM_logClasses()
   ds.hmm.fit(ds.df[ds.ts_name], ds.segment_size)
   ds.scalogramEstimation()

   # prepare raw data for Convolution Neural Net
   X,Y =ds.Data4CNN()
   # n1=ds.n_train_blocks + ds.n_val_blocks
   # m1=ds.n_steps
   # X=ds.y[:n1*m1].reshape((n1,m1))


   Cnn=CNN()
   Cnn.chart_folder = chart_log
   Cnn.prepare_train_data( train_X=X[:-4,:,:], train_Y=Y[:-4], test_X=X[-4:,:,:], test_Y=Y[-4:])
   Cnn.model()
   Cnn.fit_cnn()

   Cnn.AccuracyChart()

   logger.info('\n\n==============================================================================================')
   logger.info('\n==============================================================================================')
   logger.info('\n\n\n\n\n\n')
