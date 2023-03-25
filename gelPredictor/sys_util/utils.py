#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
from random import seed, randint
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging

import tensorflow as tf
import tensorflow_probability as tpb

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
""" The states are set( states list), the state sequence along train path is given (state_sequence list).
Let's consider  a new extended segment (block) reoresenting the average powewr per day and the time series of daily 
power observations. In our case, its dimension will be m=145=1 +144.
   Knowing that daily observations belong to one of the states, it is possible to estimate the mean vector  and 
covariance matrix of m-dimensional observations.
"""
def emisMLE_ext(yy: np.array = None, n_block:int=48, state_sequence: list = [], states: list = []) -> np.array:
    pass
u
=
L
L
L
=
s
    """ mean values per day """
    ext_seg = []

    for n_seg  in range(len(state_sequence)):
        ext_seg.append([ yy[i] for i in range(n_seg*n_block, (n_seg+1)*n_block )])
        ext_seg.insert(0, np.sum( yy[n_seg*n_block: (n_seg+1)*n_block] )/float(n_block) )


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
     u
=
L
L
L
=
s           if state_sequence[k] == statei and state_sequence[k+1] == statej:
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


def ts2supervisedLearningData(x:np.array = None, n_steps:int = 32 )->(np.array, np.array):
    """ from vector (TS) of size 'n' to supervised Learning Data matrix X of size 'n-n_steps * n_steps' and desired
    vector y of size 'n-n_steps'
    y[i] = x[i+n_steps] i =0,1,...n-n_steps-1
    X[i,j] = x[i + j,], i=0,1,...n_n_steps-1, j=0,1,...,n_steps-1.

    """

    (n,)=x.shape
    X=np.zeros(shape =(n-n_steps,n_steps), dtype = float)
    y=np.zeros(shape=(n-n_steps), dtype=float)

    for i in range(n-n_steps):
        y[i]=x[i + n_steps]
        for j in range(n_steps):
            X[i][j]=x[i+j]

    return X, y

"""
Decorator exec_time
"""


def exec_time(function):
    def timed(*args, **kw):
        time_start = perf_counter()
        ret_value = function(*args, **kw)
        time_end = perf_counter()

        execution_time = time_end - time_start

        arguments = ", ".join([str(arg) for arg in args] + ["{}={}".format(k, kw[k]) for k in kw])

        smsg = "  {:.2f} sec  for {}({})\n".format(execution_time, function.__name__, arguments)
        print(smsg)

        with open("execution_time.log", 'a') as fel:
            fel.write(smsg)

        return ret_value

    return timed


def drive_HMM(folder_predict_log:Path = None, ts_name: str = "TS", pai: np.array=None, transitDist: np.array = None,
              emisDist: np.array =None,  observations: np.array = None, observation_labels: np.array = None,
              states_set: np.array = None) -> (np.array, str, object):
    """


    :param pai:
    :param transitDist:
    :param emisDist:
    :param observations:
    :param observation_labels:
    :param states_set:
    :return:
    """


    tfd = tpb.distributions

    imprLogDist(arDist=pai,         title = "Initial Distribution")
    imprLogDist(arDist=transitDist, title = "Transition Distribution")
    imprLogDist(arDist=emisDist,    title = "Emission Distribution")

    pai = tf.convert_to_tensor(pai, dtype=tf.float64)
    transitDist = tf.convert_to_tensor(transitDist, dtype=tf.float64)

    initial_distribution = tfd.Categorical(probs=pai)
    transition_distribution = tfd.Categorical(probs=transitDist)
    mean_list = emisDist[:, 0].tolist()
    std_list = emisDist[:, 1].tolist()

    for i in range(len(std_list)):
        if std_list[i] < 1e-06:
            std_list[i] = 1e-06

    mean_list = tf.convert_to_tensor(mean_list, dtype=tf.float64)
    std_list = tf.convert_to_tensor(std_list, dtype=tf.float64)

    observation_distribution = tfd.Normal(loc=mean_list, scale=std_list)

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=len(observations))
    mean = model.mean()
    plotHMMproperties(mean_arr= mean.numpy(), folder_predict_log=folder_predict_log, ts_name=ts_name, emisDist=emisDist,
                      observations=observations, observation_labels = observation_labels, states_set=states_set)


    observations_tenzor = tf.convert_to_tensor(observations.tolist(), dtype=tf.float64)

    post_mode = model.posterior_mode(observations_tenzor)
    msg = "Posterior mode\n\n{}\n".format(post_mode)
    logger.info(msg)

    post_marg = model.posterior_marginals(observations_tenzor)
    msg = "{}\n\n{}\n".format(post_marg.name, post_marg.logits)

    logger.info(msg)
    mean_value = model.mean()
    msg = "mean \n\n{}\n".format(mean_value)
    logger.info(msg)
    log_probability = model.log_prob(observations_tenzor)
    msg = "Log probability \n\n{}\n".format(log_probability)
    logger.info(msg)

    plotViterbiPath(str(len(observations)), observation_labels, post_mode.numpy(), states_set, folder_predict_log, ts_name)

    return post_mode.numpy(), post_marg.name, post_marg.logits


def plotHMMproperties(mean_arr:np.array = None, folder_predict_log:Path = None, ts_name: str = "TS",
              emisDist: np.array =None,  observations: np.array = None, observation_labels: np.array = None,
              states_set: np.array = None):
    pass
    if mean_arr is None or emisDist is None or observations is None or observation_labels is None:
        logging.error("Missed data for PlotHMMproperties()")
        return
    msg="\n\n##day     Timestamp              State Model Value Real Value\n"
    (n,) =observations.shape
    for i in range(n):
        msg=msg + "{:>5d} {:<30s} {:>2d} {:<10.4f} {:<10.4f}\n".format(i, observation_labels[i],
                    states_set[i], mean_arr[i], observations[i])
    msg=msg+"\n\n"
    logger.info(msg)

    file_name="HMM_properties"
    file_png = None
    if folder_predict_log is None:
        file_png = Path(file_name).with_suffix(".png")
    else:
        file_png = Path(folder_predict_log / Path(file_name)).with_suffix(".png")
    try:
        plt.plot(mean_arr,label = "HMM mean")
        plt.plot(observations,label ="Day Value" )
        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle("Hidden Markov Model Estimated Observations\n{}".format(ts_name), fontsize=24)
        plt.ylabel("Observations", fontsize=18)
        plt.xlabel("Day Number", fontsize=18)
        plt.legend()
        plt.savefig(file_png)

    except:
        logger.error("\nCan not plot plotHMMproperties()\n")
    finally:
        plt.close("all")

    try:
        plotArray(arr=states_set, title=ts_name, folder_control_log=folder_predict_log, file_name="state_sequence")
    except:
        logger.error("\nCan not plot stste sequences plotHMMproperties()\n")
    finally:
        plt.close("all")
    return


def plotViterbiPath(pref, observations, viterbi_path, hidden_sequence, folder_predict_log, ts_name):
    """

    :param df:
    :param cp:
    :return:
    """

    suffics = ".png"
    if hidden_sequence is not None:
        file_name = "{}_viterbi_sequence_vs_hidden_sequence".format(pref)
    else:
        file_name = "{}_viterbi_sequence".format(pref)
    file_png = file_name + ".png"
    vit_png = Path(folder_predict_log / file_png)

    try:
        # plt.plot(observations, viterbi_path, label="Viterbi path")
        # plt.plot(observations, hidden_sequence, label="Hidden path")
        plt.plot(viterbi_path, label="Viterbi path")

        if hidden_sequence is not None:
            plt.plot(hidden_sequence, label="Hidden path")

        else:
            pass
        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)
        if hidden_sequence is not None:
            fig.suptitle("Viterby optimal path vs hidden path for dataset\n{}".format(ts_name), fontsize=24)
        else:
            fig.suptitle("Viterby optimal path for dataset\n{}".format(ts_name), fontsize=24)
        plt.ylabel("States", fontsize=18)
        plt.xlabel("Observation timestamps", fontsize=18)
        plt.legend()
        plt.savefig(vit_png)

    except:
        pass
    finally:
        plt.close("all")

    return


def plotArray(arr: np.array = None, title: str = "TS", folder_control_log: Path = None, file_name: str = "TS"):


    file_png = file_name + ".png"
    fl_png = Path(folder_control_log / file_png)

    try:
        plt.plot(arr)
        numfig = plt.gcf().number
        fig = plt.figure(num=numfig)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle("{}".format(title), fontsize=24)

        plt.savefig(fl_png)
        message = f"""

           Plot file                : {fl_png}
           Time series              : {title}

           """
        logger.info( message)
    except:
        pass
    finally:
        plt.close("all")
    return

def imprLogDist(arDist: np.array = None, row_header: list = None, col_header: list = None, title: str= ""):


    logger.info(title)
    shp = arDist.shape
    if len(shp) == 1:

        b = arDist.reshape((-1, 1))
        (n_row, n_col) = b.shape
        auxLogDist(b, n_row, n_col, row_header, col_header, title)

    elif len(shp) == 2:
        (n_row, n_col) = shp
        auxLogDist(arDist, n_row, n_col, row_header, col_header, title)
    else:
        msg = "Incorrect array shape {}".format(shp)
        logger.error(msg)

    return

def auxLogDist(arDist: np.array, n_row:int, n_col:int,  row_header:list, col_header:list, title: str):
    if row_header is None or not row_header:
        row_header = [str(i) for i in range(n_row)]
    if col_header is None or not col_header:
        col_header = [str(i) for i in range(n_col)]
    row_header = [str(i) for i in row_header]
    col_header = [str(i) for i in col_header]
    wspace = ' '
    s = "{:<10s}".format(wspace)

    for i in col_header:
        s = s + "{:^11s}".format(i)

    logger.info(s)
    for i in range(n_row):
        s = "{:<10s}".format(row_header[i])
        for j in range(n_col):
            if isinstance(arDist[i][j], int):
                s1 = "{:>10d}".format(arDist[i][j])
            elif isinstance(arDist[i][j], float):
                s1 = "{:>10.4f}".format(arDist[i][j])
            else:
                s1 = "{:^10s}".format(arDist[i][j])
            s = s + "  " + s1

        logger.info(s)
    return

def plotClusters(kmeans: KMeans, X: np.array, file_png:Path):
    """
    The plot shows 2 first component of X
    :param kmeans: -sclearn.cluster.Kmeans object
    :param X: matrix n_samples * n_features or principal component n_samples * n_components.
    :param file_png:
    :return:
    """
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50)
    plt.savefig(file_png)
    plt.close("all")
    return

if __name__ == "__main__":

    pass
    x=np.arange(16, dtype=float)
    X,y = ts2supervisedLearningData(x=x, n_steps=4 )
    pass