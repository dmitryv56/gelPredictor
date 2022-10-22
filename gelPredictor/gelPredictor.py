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
from sys_util.utils import setOutput
from src.drive import drive_all_classes
from src.flow_control import cntrPath,medTrainPath, shrtTrainPath, hmmTrainPath
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
    ds, X, Y = cntrPath()

    # ds = Dataset(pathTo=PATH_TO_DATASET, ts=TS_NAME, dt=TS_TIMESTAMP_LABEL, sampling=SAMPLING,
    #              segment_size=SEGMENT_SIZE, n_steps=N_STEPS, overlap=OVERLAP,
    #              continuous_wavelet=CONTINUOUS_WAVELET, norm=DATA_NORMALIZATION, num_classes=NUM_CLASSES,
    #              model_repository=PATH_REPOSITORY, log_folder=log_folder, chart_log=chart_log)
    #
    # ds.readDataset()
    # ds.data_normalization()
    # ds.setTrainValTest()
    # ds.__str__()
    # ds.createSegmentLst()
    # ds.ExtStatesExtraction()
    # ds.initHMM_logClasses()
    # ds.createExtendedDataset()
    # ds.hmm.fit(ds.df[ds.ts_name], ds.segment_size)
    # ds.scalogramEstimation()
    #
    # # prepare raw data for Convolution Neural Net
    # X, Y = ds.Data4CNN()
    # # n1=ds.n_train_blocks + ds.n_val_blocks
    # # m1=ds.n_steps
    # # X=ds.y[:n1*m1].reshape((n1,m1))

    """ Train path for med term forecasting"""

    Cnn = medTrainPath(ds = ds, X = X, Y = Y)
    # Cnn = CNN()
    # Cnn.chart_folder = chart_log
    # Cnn.prepare_train_data(train_X=X[:-4, :, :], train_Y=Y[:-4], test_X=X[-4:, :, :], test_Y=Y[-4:])
    # Cnn.model()
    # Cnn.fit_cnn()
    #
    # Cnn.AccuracyChart()

    """ Train for hmm-model """
    hmmTrainPath(ds=ds)

    """ Train path for Very Short-Term forecasting """

    shrtTerm = shrtTrainPath(ds=ds)

    #
    # shrtTerm = VeryShortTerm(num_classes=ds.num_classes, segment_size=ds.segment_size, n_steps=N_STEPS, df=ds.df,
    #                      dt_name=ds.dt_name, ts_name=ds.ts_name, exogen_list=[ds.dt_name, ds.ts_name],
    #                      list_block=ds.lstBlocks, repository_path=PATH_SHRT_DATASETS)
    # for i in range(ds.num_classes):
    #     shrtTerm.createDS(i)
    #
    # drive_all_classes(shrt_data=shrtTerm)

    logger.info('\n\n==============================================================================================')
    logger.info('\n==============================================================================================')
    logger.info('\n\n\n\n\n\n')

    return 0

if __name__ == '__main__':
   pass

   nret = main(None, None)
