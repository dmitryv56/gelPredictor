#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import logging

from sys_util.parseConfig import PATH_LOG_FOLDER , PATH_REPOSITORY, PATH_CHART_LOG

logger = logging.getLogger(__name__)

DTNOW = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

def simpleScalingImgShow(scalogram:object=None, index:int=0,title:str=""):

    fig,ax = plt.subplots()

    ax.imshow(scalogram, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(scalogram).max(),
               vmin=-abs(scalogram).max())
    ax.set_title(title)
    ax.set_xlabel("Related Time Series")
    ax.set_ylabel("Scales")
    plt.savefig("{}__.png".format(title))
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







if __name__ == "__main__":

    pass