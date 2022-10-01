#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
from random import seed, randint
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import logging

from sys_util.parseConfig import PATH_LOG_FOLDER , PATH_REPOSITORY, PATH_CHART_LOG

MEAN_COL = 0
STD_COL = 1
SEED = 1956

seed(SEED)

logger = logging.getLogger(__name__)

DTNOW = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

def execution_time(function):

    def timed(*args,**kw):
        time_start =perf_counter()
        ret_value=function(*args,**kw)
        time_end = perf_counter()

        msg = "\n\n {:.2f} sec for {}({},{})\n".format(time_end - time_start, function.__name__, args,kw)
        logger.info(msg)
        return ret_value

    return timed

def simpleScalingImgShow(scalogram:object=None, index:int=0,title:str="",file_png:str="xx.png"):

    fig,ax = plt.subplots()
    (scale,shift) = scalogram.shape
    extent_lst=[0,shift,1,scale]
    ax.imshow(scalogram,
              extent=extent_lst, # extent=[-1, 1, 1, 31],
              cmap='PRGn', aspect='auto',
              vmax=abs(scalogram).max(),
              vmin=-abs(scalogram).max())
    ax.set_title(title)
    ax.set_xlabel("Related Time Series")
    ax.set_ylabel("Scales")
    plt.savefig("{}__.png".format(file_png))
    plt.close("all")

    return


def setOutput(model_repository: Path =None, log_folder: Path=None, chart_log: Path =None)->(Path,Path,Path):
    m_r="model_{}".format(DTNOW)
    if model_repository is None:
        model_repository = Path(m_r)
    else:
        model_repository = Path(model_repository / Path(m_r))
    model_repository.mkdir(parents=True, exist_ok=True)

    l_f = "train_log_{}".format(DTNOW)
    if log_folder is None:
        log_folder = Path(l_f)
    else:
        log_folder = Path(log_folder / Path(l_f))
    log_folder.mkdir(parents=True, exist_ok=True)

    ch_l = "charts_{}".format(DTNOW)
    if chart_log is None:
        chart_log = Path(ch_l)
    else:
        chart_log = Path(log_folder / Path(ch_l))
    chart_log.mkdir(parents=True, exist_ok=True)

    return model_repository, log_folder, chart_log


def chartStTS(ts:np.array=None, dt:np.array = None, tsname:str="TimeSeries", dtname:str = "Date Time", states: np.array = None,
              n_steps:int = 144, sampling:int= 10 *60 , chunk:int = 512):

    if ts is None or dt is None or states is None:
        logger.error("Invalid data for charting")
        return

    


""" This function estimates emission probabilities. The normal distributions with 'mean' and 'std' is postulated.
The train data y[t], state sequence and list of states are passed as arguments.
"""


def emisMLE(yy: np.array = None, n_block:int=48, state_sequence: list = [], states: list = []) -> np.array:
    """
    :param y: observations
    :param state_sequence:  state sequences . The size of observations and size of sequence are equal.
    :param states:
    :return:
    """
    logger.info(
        "Emission estimation: \nsegment size ={}\n array size = {} \nstate sequence  size = {} \n states = {}\n".format(
        n_block, len(yy), len(state_sequence), len(states)))

    if yy is None or len(states) == 0 or len(state_sequence) ==0:
        logger.error("{} invalid arguments".format(emisMLE.__name__))
        return None
    y=np.zeros(len(state_sequence), dtype=float)
    yy=np.array(yy)
    nyy=round(len(yy)/n_block)
    logger.info("Rounded number of segments over TS array = {}".format(nyy))
    Y = yy[:nyy * n_block].reshape((nyy, n_block))
    logger.info(Y.shape)
    for k in range (nyy):
        for i in range(n_block):
            y[k]=y[k] + Y[k,i]
        y[k]=y[k]/float(n_block)
    emisDist = np.zeros((len(states), 2), dtype=float)    # allocation matrix n_states *2 . The state points on the row.
    # Each row contains a 'mean' and 'std' for this state
    (n,) = y.shape
    msg = ""
    for state in states:
        a_aux = []
        for i in range(n):
            if state_sequence[i] == state:
                a_aux.append(y[i])
        if not a_aux:
            emisDist[state][MEAN_COL] = 0.0
            emisDist[state][STD_COL] = 1e-06
            logger.error("No observations for {} state".format(state))
        elif len(a_aux) == 1:
            emisDist[state][MEAN_COL] = a_aux[0]
            emisDist[state][STD_COL] = 1e-06
        else:
            a = np.array(a_aux, dtype=float)
            emisDist[state][MEAN_COL] = round(a.mean(), 4)
            emisDist[state][STD_COL] = round(a.std(), 6)
        msg = msg + "{}: {} {}\n".format(state, emisDist[state][MEAN_COL], emisDist[state][STD_COL])
    message = f"""
    Emission Prob. Distribution
State Mean  Std 
{msg}
"""
    logger.info(message)
    return emisDist


""" This function estimates the pai (initial) distribution.
The estimate initial pay[state] is proportional of the occurence times for this state.
"""


def paiMLE(states: list = [], count: np.array = None, train_sequence_size: int = 1) -> np.array:
    """
    :param states:list of possible states like as [0,1, ... n_states-1]
    :param count: list of number occurinces for each state in states sequences
    :param train_sequence_size: size of states sequences
    :return:
    """

    if  count is None or len(states)==0 or train_sequence_size == 1:
        logger.error("{} invalid arguments".format(paiMLE.__name__))
        return None

    pai = np.array([round(count[state]/train_sequence_size, 4) for state in states])
    msg = "".join(["{}: {}\n".format(state,pai[state]) for state in states])
    message = f"""
State  Pai
{msg}
"""
    logger.info(message)

    return pai


def transitionsMLE(state_sequence: list = [], states: list = []) -> np.array:
    """
    :param state_sequence:  list of states (states sequence)  corresponding to observations.
    :param states: list of possible states like as [0,1, ... n_states-1]
    :return: transition matrix len(states) * len(states)
    """
    if len(state_sequence) == 0 or len(states) == 0:
        logger.error("{} invalid arguments".format(transitionsMLE.__name__))
        return None


    # Denominators are counts of state occurence along sequence without last item.
    _, denominators = np.unique(state_sequence[:-1], return_counts=True)
    # Note: If some state appears only once one as last item in seqiuence then this state will loss.
    transDist = np.zeros((len(states), len(states)), dtype=float)

    for statei in states:
        denominator = denominators[statei]
        msg = "State {} : ".format(statei)
        for statej in states:
            nominator =0
            for k in range(len(state_sequence)-1):
                if state_sequence[k] == statei and state_sequence[k+1] == statej:
                    nominator += 1
            transDist[statei][statej] = round(float(nominator)/float(denominator), 6)
            msg = msg + "{} ".format(transDist[statei][statej])
        message = f"""{msg}"""
        logger.info(message)
    return transDist

def getEvalSequenceStartIndex(n_train:int=64, n_eval:int=36, n_test:int=0)->int:
    rn = randint(0, n_train -(n_eval+n_test))
    msg = "Evaluation sequence begins at {}, test sequence begins at {} ".format(rn,rn+n_eval)
    if n_test == 0:
        msg = "Evaluation sequence begins at {}".format(rn)

    logger.info(msg)
    return rn


if __name__ == "__main__":

    pass