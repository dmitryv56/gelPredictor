# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.dau import Dataset
from src.cnn import  CNN

LOG_FOLDER_NAME="log"
MAIN_SYSTEM_LOG="main"
SERVER_LOG="server"
CHART_LOG="chart"

# -----------------------------------------------------------------------------------------------------------------
# Logging & Charting
PATH_ROOT_FOLDER = Path(Path(__file__).parent.absolute())
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
PATH_TO_DATASET='~/LaLaguna/stgelpDL/dataLaLaguna/ElHiero_24092020_27102020.csv'
# ------------------------------------------------------------------------------------------------------------------

size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))

size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)
logger.info('\nLogs: {} \nMain log: {} \nRepository: {} \nDatasets:{} \nDescriptors{}'.format(PATH_LOG_FOLDER,
            PATH_MAIN_LOG, PATH_REPOSITORY, PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY))
# ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
   pass

   # cnn.drive()

   ds=Dataset(pathTo = PATH_TO_DATASET, ts = "WindTurbine_Power", dt = "Date Time", sampling= 10 * 60, n_steps=36)
   ds.readDataset()
   ds.statnorm()
   ds.setTrainValTest()
   ds.__str__()
   ds.crtListBlocks()
   ds.StatesExtraction()
   X,Y =ds.Data4CNN()
   # n1=ds.n_train_blocks + ds.n_val_blocks
   # m1=ds.n_steps
   # X=ds.y[:n1*m1].reshape((n1,m1))

   Cnn=CNN()


   Cnn.prepare_train_data( train_X=X[:-4,:,:], train_Y=Y[:-4], test_X=X[-4:,:,:], test_Y=Y[-4:])
   Cnn.model()
   Cnn.fit_cnn()

   Cnn.AccuracyChart()

   logger.info('\n\n==============================================================================================')
   logger.info('\n==============================================================================================')
   logger.info('\n\n\n\n\n\n')
