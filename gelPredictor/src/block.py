#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pywt
import logging
from pathlib import Path

from sys_util.utils import simpleScalingImgShow

logger = logging.getLogger(__name__)

class Block(object):
    """ Class"""

    def __init__(self, x:np.array=None, scales:list =[],wav:object=None, sampling:int=10*60, timestamp:str="",
                 index:int=-1, isTrain:bool = True, desire:int=-1, average:float = 0.0):
        """Constructor"""

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
        self.average = average


    def scalogramEstimation(self, mode:int = 0, y:np.array =None, title:str ="", chart_repository:Path = None)->\
            (np.array, np.array):
        """

        :param mode: 0  - self.scalogram,self.freq  is calculated for class member self.x - observation vector
                     1  - self.scalogram,self.freq  is calculated for class member self.x - observation vector. The
                          estimated scalogram is displayed as 2D-image.
                     2 -  scalogram, freq  is calculated for passed vector y, not member of class block. The estimated
                          scalogram is displayed as 2D image.
        :param y:
        :param title:
        :return:    scalogram, frequences.
        """

        if mode ==0:
            self.scalogram, self.freqs = pywt.cwt(self.x, self.scales,self.wav)
            title="Scalogram_State_{}_block_{}_Timestamp_{}".format(self.desire,self.index,self.timestamp)
            return   self.scalogram, self.freqs
        elif mode == 1:
            self.scalogram, self.freqs = pywt.cwt(self.x, self.scales, self.wav)
            title = "Scalogram_State_{}_block_{}_Timestamp_{}".format(self.desire, self.index, self.timestamp)
            if chart_repository is None:
                file_png="{}__.png".format(title)
            else:
                file_png=Path(chart_repository/Path(title)).with_suffix(".png)")
            #   simpleScalingImgShow(scalogram = self.scalogram, index=self.index, title = title, file_png=file_png)
            return self.scalogram, self.freqs
        elif mode == 2:
            scalogram1, freqs1 = pywt.cwt(y, self.scales, self.wav)
            title = "Scalogram_{}".format( title)
            if chart_repository is None:
                file_png="{}__.png".format(title)
            else:
                file_png=Path(chart_repository/Path(title)).with_suffix(".png)")
            simpleScalingImgShow(scalogram = scalogram1, index=0, title = title, file_png=file_png)
            return scalogram1, freqs1
        else:
            return None, None
        # plt.imshow(self.scalogram, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
        # vmax = abs(self.scalogram).max(), vmin = -abs(self.scalogram).max())
        # # plt.show()
        # plt.savefig("{}__.png".format(self.index))

if __name__ == "__main__":
    pass