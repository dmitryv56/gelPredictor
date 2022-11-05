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

import matplotlib.pyplot as plt
import numpy as np

from src.dau import Dataset
from src.cnn import CNN
from src.vshtrm import VeryShortTerm
from sys_util.parseConfig import LOG_FOLDER_NAME, MAIN_SYSTEM_LOG, SERVER_LOG, CHART_LOG, PATH_ROOT_FOLDER, \
    PATH_LOG_FOLDER, PATH_MAIN_LOG, PATH_SERVER_LOG, PATH_CHART_LOG, PATH_REPOSITORY, PATH_DATASET_REPOSITORY, \
    PATH_DESCRIPTOR_REPOSITORY, MAX_LOG_SIZE_BYTES, BACKUP_COUNT, PATH_TO_DATASET, DATA_NORMALIZATION, OVERLAP, \
    N_STEPS, CONTINUOUS_WAVELET, NUM_CLASSES, NUM_SCALES, SAMPLING, TS_NAME, TS_TIMESTAMP_LABEL, SEGMENT_SIZE, \
    PATH_SHRT_DATASETS, TEST_FOLDER, COMPRESS, printInfo
from sys_util.utils import setOutput, exec_time, drive_HMM
from src.drive import drive_all_classes

logger = logging.getLogger(__name__)

_, log_folder, chart_log = setOutput(model_repository=PATH_REPOSITORY,
                                     log_folder=PATH_LOG_FOLDER, chart_log=PATH_CHART_LOG)


@exec_time
def cntrPath() -> (Dataset, np.array, np.array):
    """ Control paths"""

    """ Data Source acquisition """
    ds = Dataset(pathTo = PATH_TO_DATASET, ts = TS_NAME, dt = TS_TIMESTAMP_LABEL, sampling = SAMPLING,
                 segment_size = SEGMENT_SIZE, n_steps = N_STEPS, overlap = OVERLAP,
                 continuous_wavelet = CONTINUOUS_WAVELET, norm = DATA_NORMALIZATION, compress = COMPRESS,
                 num_classes=NUM_CLASSES, model_repository = PATH_REPOSITORY, log_folder = log_folder,
                 chart_log=chart_log)

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

    return ds, X, Y


@exec_time
def shrtTermTrainPath(ds: Dataset = None, X: np.array = None, Y: np.array = None) -> CNN:
    """ Median train path"""

    Cnn = CNN()
    Cnn.chart_folder = chart_log
    Cnn.prepare_train_data(train_X=X[:-4, :, :], train_Y=Y[:-4], test_X=X[-4:, :, :], test_Y=Y[-4:])
    Cnn.model()
    Cnn.fit_cnn()

    Cnn.AccuracyChart()

    return Cnn


@exec_time
def veryShrtTrainPath(ds: Dataset = None) -> VeryShortTerm:
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
    """ HMM Observation for segment is mean value per segment.
    HMM operates with segment(blocks), i.e. state per block. The observation are aggregated by mean of observations
    per segment(block). The each segment signed by label that is a timestamp for 1 sampling item in segment.
    The matrix mean4state is auxiliary matrix contain mean vector of obserbations  are belongs  to state.
    Those mean vectors shall use for compare predict and dedicated test data.
    For example, the predicted state for next segment  is state=S. The S-th mean vector items is compared with test
    segment values.
    """
    observation_labels = []
    observations = []     # aggregated observations

    mean4state = np.zeros(shape=(ds.num_classes, ds.segment_size), dtype=float)  # auxiliary.
    cntr_states = np.zeros(shape=ds.num_classes, dtype=int)

    for item in ds.lstBlocks:
        observation_labels.append(item.timestamp)  # add first timestamp for segment
        segment_index = item.index                 # offset segment of the beginning of the time series
        val = 0.0
        cntr_states[item.desire] = cntr_states[item.desire] + 1  # counter of the states
        j = 0
        for i in range(segment_index, segment_index + ds.segment_size):
            # accumulate values along segment
            val = val + ds.df[ds.ts_name].values[i]
            # accumulate sampling values along states
            mean4state[item.desire, j] = mean4state[item.desire, j] + ds.df[ds.ts_name].values[i]
            j = j + 1

        observations.append(val/ds.segment_size)

    (cl, m) = mean4state.shape
    for i in range(cl):
        for k in range(m):
            mean4state[i, k] = mean4state[i, k]/float(cntr_states[i])

    viterbi_path, post_marg_name_, post_marg_logits_ = drive_HMM(folder_predict_log=ds.chart_log, ts_name=ds.ts_name,
                                                                 pai=ds.hmm.pi, transitDist=ds.hmm.A,
                                                                 emisDist=ds.hmm.B, observations=np.array(observations),
                                                                 observation_labels=np.array(observation_labels),
                                                                 states_set=ds.hmm.state_sequence)

    pred_states, pred_states_probability = hmmPredpath(ds=ds, viterbi_path=viterbi_path, num_predictions=4)

    shorttermForecastChart(ds=ds, predicted_states=pred_states)  # TODO

    return


def hmmPredpath(ds:Dataset=None, viterbi_path:np.array=None, num_predictions:int=2)->(list,list):
    """ H(idden) M(arkov) M(odel)  """

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

def shorttermForecastChart(ds:Dataset = None, predicted_states:list = None ):   # TODO
    pass


    predict_ll =[ ds.blck_class_centers[i].x for i in predicted_states]  # list of lists [[..],..,[..]]
    predict = [item for sublist in predict_ll for item in sublist ]  # flatten list [..]
    test_start= (ds.n_train_blocks + ds.n_val_blocks) * ds.segment_size
    tail_ts = ds.df[ds.ts_name].values[test_start:]

    predict_timestamp = ds.df[ds.dt_name].values[test_start:]

    predict_size = len(predict) if len(predict) <len( tail_ts) else len( tail_ts)
    msg = "     (Short Term) Predicted and tested values of {}\n{:<3s}  {:<5s} {:<28s} {:<10s} {:<10s}\n".format( \
        ds.ts_name, "##", "State","Timestamp", "Predict",  "Test value")
    k=0
    for i in range(predict_size):

        msg = msg + "{:<3d}  {:<2d}    {:<28s} {:<10.4f} {:<10.4f}\n".format( \
            i, predicted_states[k], predict_timestamp[i], predict[i],  tail_ts[i])

        if (i>0) and ((i % ds.segment_size) == 0):
            k = k + 1

    logger.info(msg)
    fl_log =Path(TEST_FOLDER / Path("ShortTermForecasting_TestData")).with_suffix(".log")
    fl_chart = Path(TEST_FOLDER  / Path("ShortTermForecasting_TestData")).with_suffix(".png")
    with open(fl_log, 'w') as fout:
        fout.write(msg)
    x_axis =[i for i in range(predict_size)]
    plt.figure()
    # plt.plot(x_axis, predict_size,'b', alfa=0.75)
    plt.plot(x_axis,predict[:predict_size],'r',x_axis,tail_ts[:predict_size],'k')
    plt.legend(('Short term Predict','Test Values'), loc='best')
    plt.grid(True)
    plt.savefig( fl_chart)
    plt.close("all")
    return



if __name__ == "__main__":
    pass