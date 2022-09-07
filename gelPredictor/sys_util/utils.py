#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)



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






if __name__ == "__main__":

    pass