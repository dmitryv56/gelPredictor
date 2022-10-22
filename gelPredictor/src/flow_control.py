#!/usr/bin/env ptrhon3

""" While trainpath and predictpath are the paths on which model estimation and prediction computations occur, the
control path is the path of initialization of objects that control code flow over train- and predictpath.
The control path is the logic to manage the flow of data through design blocks. Control paths include such as followinf:
 - acquisition data source,
 - data normalization,
 - segment extraction from TS,
 - correlation and spectral analysis,
 - supervised learning data for ANN models,
 - HMM initialization.

 """

import logging
from pathlib import Path
import numpy as np

from src.dau import Dataset
from src.cnn import  CNN
from src.vshtrm import VeryShortTerm
from sys_util.parseConfig import LOG_FOLDER_NAME, MAIN_SYSTEM_LOG, SERVER_LOG, CHART_LOG, \
PATH_ROOT_FOLDER, PATH_LOG_FOLDER , PATH_MAIN_LOG , PATH_SERVER_LOG, PATH_CHART_LOG , \
PATH_REPOSITORY , PATH_DATASET_REPOSITORY, PATH_DESCRIPTOR_REPOSITORY, \
MAX_LOG_SIZE_BYTES, BACKUP_COUNT, PATH_TO_DATASET, DATA_NORMALIZATION, OVERLAP, N_STEPS, CONTINUOUS_WAVELET, \
NUM_CLASSES, NUM_SCALES, SAMPLING, TS_NAME, TS_TIMESTAMP_LABEL, SEGMENT_SIZE, PATH_SHRT_DATASETS,printInfo
from sys_util.utils import setOutput, exec_time, drive_HMM
from src.drive import drive_all_classes

logger = logging.getLogger(__name__)

_, log_folder, chart_log = setOutput(model_repository=PATH_REPOSITORY,
                                     log_folder=PATH_LOG_FOLDER, chart_log=PATH_CHART_LOG)
@exec_time
def cntrPath()->(Dataset, np.array, np.array):
    """ Control paths"""



    """ Data Source acquisition """
    ds = Dataset(pathTo=PATH_TO_DATASET, ts=TS_NAME, dt=TS_TIMESTAMP_LABEL, sampling=SAMPLING,
                 segment_size=SEGMENT_SIZE, n_steps=N_STEPS, overlap=OVERLAP,
                 continuous_wavelet=CONTINUOUS_WAVELET, norm=DATA_NORMALIZATION, num_classes=NUM_CLASSES,
                 model_repository=PATH_REPOSITORY, log_folder=log_folder, chart_log=chart_log)

    ds.readDataset()
    """ Normalization"""
    ds.data_normalization()
    ds.setTrainValTest()
    ds.__str__()
    ds.createSegmentLst()
    ds.ExtStatesExtraction()
    ds.initHMM_logClasses()
    ds.createExtendedDataset()
    ds.hmm.fit(ds.df[ds.ts_name], ds.segment_size)
    ds.scalogramEstimation()

    # prepare raw data for Convolution Neural Net
    X, Y = ds.Data4CNN()

    return ds, X,Y

@exec_time
def medTrainPath(ds: Dataset = None, X: np.array = None, Y: np.array = None)->CNN:
    """ Median train path"""

    Cnn = CNN()
    Cnn.chart_folder = chart_log
    Cnn.prepare_train_data(train_X=X[:-4, :, :], train_Y=Y[:-4], test_X=X[-4:, :, :], test_Y=Y[-4:])
    Cnn.model()
    Cnn.fit_cnn()

    Cnn.AccuracyChart()

    return Cnn

@exec_time
def shrtTrainPath(ds: Dataset = None)->VeryShortTerm:
    pass

    shrtTerm = VeryShortTerm(num_classes=ds.num_classes, segment_size=ds.segment_size, n_steps=N_STEPS, df=ds.df,
                         dt_name=ds.dt_name, ts_name=ds.ts_name, exogen_list=[ds.dt_name, ds.ts_name],
                         list_block=ds.lstBlocks, repository_path=PATH_SHRT_DATASETS)
    for i in range(ds.num_classes):
        shrtTerm.createDS(i)

    drive_all_classes(shrt_data=shrtTerm)

    return shrtTerm

@exec_time
def hmmTrainPath(ds: Dataset = None):
    """ HMM """
    observation_labels =[]
    observations = []
    for item in ds.lstBlocks:
        observation_labels.append(item.timestamp)
        segment_index = item.index
        val =0.0
        for i in range(segment_index, segment_index + ds.segment_size):
            val = val + ds.df[ds.ts_name].values[i]
        observations.append(val/ds.segment_size)

    viterbi_path, post_marg_name_, post_marg_logits_ = drive_HMM(folder_predict_log = ds.chart_log, ts_name = ds.ts_name,
                                                               pai = ds.hmm.pi, transitDist = ds.hmm.A,
                                                               emisDist=ds.hmm.B, observations=np.array(observations),
                                                               observation_labels=np.array(observation_labels),
                                                               states_set = ds.hmm.state_sequence)

    pred_states, pred_states_probability = hmmPredpath(ds = ds, viterbi_path = viterbi_path, num_predictions = 4 )

    return

def hmmPredpath(ds: Dataset = None, viterbi_path:np.array = None, num_predictions:int =2)->(list, list):
    """ H(idden) M(arkov) M(odel) """

    pred_states = []
    pred_states_probability = []
    new_pred_step = viterbi_path[-1]
    prob_pred_state = 1.0
    for i in range(num_predictions):
        new_pred_step, prob_new_pred_state = ds.hmm.one_step_predict(new_pred_step)
        pred_states.append(new_pred_step)
        prob_pred_state = prob_pred_state * prob_new_pred_state
        pred_states_probability.append(prob_pred_state)
    return pred_states, pred_states_probability

if __name__ == "__main__":
    pass