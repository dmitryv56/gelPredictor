#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pywt
import logging

from sys_util.utils import simpleScalingImgShow

logger = logging.getLogger(__name__)

class Block(object):

    def __init__(self, x:np.array=None, scales:list =[],wav:object=None, sampling:int=10*60, timestamp:str="",
                 index:int=-1, isTrain:bool = True, desire:int=-1):
        self.log=logger
        self.x = np.array(x.tolist())
        self.sampling = sampling
        self.timestamp=timestamp
        self.index=index
        self.desire=desire
        self.scalogram=None
        self.freqs = None
        self.isTrain=isTrain
        self.scales=scales
        self.wav=wav


    def scalogramEstimation(self):
        pass
        self.scalogram, self.freqs = pywt.cwt(self.x, self.scales,self.wav)
        title="Scalogram_State_{}_block_{}_Timestamp_{}".format(self.desire,self.index,self.timestamp)
        # simpleScalingImgShow(scalogram = self.scalogram, index=self.index, title = title)

        # plt.imshow(self.scalogram, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
        # vmax = abs(self.scalogram).max(), vmin = -abs(self.scalogram).max())
        # # plt.show()
        # plt.savefig("{}__.png".format(self.index))

if __name__ == "__main__":
    pass