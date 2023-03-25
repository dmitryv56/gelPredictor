#!/usr/bin/env python3

""" Dataset Demand """

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging

from src.block import Block
from sys_util.parseConfig import PATH_LOG_FOLDER , PATH_REPOSITORY, PATH_CHART_LOG, TRAIN_RATIO, VAL_RATIO, \
    TS_DEMAND_NAME
from sys_util.utils import simpleScalingImgShow
from src.hmm import hmm_demand
from sys_util.parseConfig import STATE_0_MARGE, STATE_0, STATE_DEMAND, STATE_GENERATION
from src.dau import Dataset
logger = logging.getLogger(__name__)
FIRST_DAY_IN_WEEK = 1
LAST_DAY_IN_WEEK  = 7
FIRST_MONTH       = 1
LAST_MONTH        = 12
#  el=Hierro
# PWR_0_MIN = 4.9
# PWR_1_MIN = 5.4
# PWR_2_MIN = 5.9
# PWR_3_MIN = 6.4
# PWR_3_MAX = 8.5
# range is 7.62 .. .. 54.72  step ~ 16.0
PWR_0_MIN = 7.0
PWR_1_MIN = PWR_0_MIN +16.0
PWR_2_MIN = PWR_1_MIN +16.0
PWR_3_MIN = PWR_2_MIN + 16.0
PWR_3_MAX = PWR_3_MIN + 16.0

class DatasetDemand(Dataset):

    def __init__(self, pathTo: str = "", ts: str = "", dt: str = "Date Time", sampling: int = 10 * 60,
                 n_steps: int = 144,
                 segment_size: int = 96, norm: str = "stat", overlap: int = 0, continuous_wavelet: str = 'mexh',
                 num_classes: int = 4, num_scales: int = 16, compress: str = 'pca', n_components: int = 2,
                 model_repository: Path = PATH_REPOSITORY, log_folder: Path = PATH_LOG_FOLDER,
                 chart_log: Path = PATH_CHART_LOG):
        """ Constructor """
        self.n4cnnTrain = 0
        super().__init__(pathTo=pathTo, ts=ts, dt=dt, sampling=sampling, n_steps=n_steps, segment_size=segment_size,
                         norm=norm, overlap=overlap, continuous_wavelet=continuous_wavelet,
                         num_classes=num_classes, num_scales=num_scales, compress=compress,
                         n_components=n_components, model_repository=model_repository, log_folder=log_folder,
                         chart_log=chart_log)

        self.d_averPwr = {}
        self.n4cnnTrain = 0
        self.aver = np.zeros((self.num_classes, self.segment_size), dtype=float)
        self.count = np.zeros((self.num_classes), dtype=int)
        self.aver_of_aver = np.zeros((self.num_classes), dtype=float)
        self.hmm = hmm_demand(dt=self.dt,log_folder=self.log_folder)


        # self.pathToCsv = pathTo
        # self.df: pd.DataFrame = None
        # self.y = None
        # self.dt = None
        # self.n = 0
        # self.mean = 0.0
        # self.std = 1.0
        # self.min = 0.0
        # self.max = 1.0
        # self.n_train = 0
        # self.n_val = 0
        # self.n_test = 0
        # self.n_train_blocks = 0
        # self.n_val_blocks = 0
        # self.lstBlocks = []
        # self.lstOffsetSegment = []
        # self.hmm = hmm()

        return

    """ THis method aims on the aggregation observations. Common algorithm description is below:
       TBD
       """

    """ This method aims on the aggregation a day's observations in the segments (blocks). Each block has desired label 
          according by belongs the day average power demand to one of ranges 5.0-5.55, 5.55-5.9,5.9-6.4 MWT 
          The average for each day is calculated and saved in the dict{offset day : aver}.
          Because day is vector of 144 samples, then offsets are 0, 144,288, ...."""

    def createSegmentLstPerDayPerPower(self):

        self.n4cnnTrain = self.n_train + self.n_val
        self.lstOffsetSegment = []
        n_seg = 0

        while n_seg * self.segment_size <= self.n4cnnTrain:
            self.lstOffsetSegment.append(n_seg * self.segment_size)  # segments without overlap

            averPwr = np.sum([self.y[i] for i in
                        range(n_seg * self.segment_size, n_seg * self.segment_size + self.segment_size)])

            self.d_averPwr[n_seg * self.segment_size] = averPwr / float(self.segment_size)
            n_seg = n_seg + 1
        #############################################################################################################
        msg = ""

        self.lstOffsetSegment.sort()
        k = 0
        for item in self.lstOffsetSegment:
            msg = msg + "{:<6d}( {:<20s}):{:<8.2f} ".format(item,  self.dt[item],self.d_averPwr[item])
            k = k + 1
            if k % 4 == 0:
                msg = msg + "\n"

        message = f"""
The size of train part of TS for CNN learning: {self.n4cnnTrain} 
The number of segments (blocks)              : {len(self.lstOffsetSegment)} 
The overlap                                  : {self.overlap}

Segment  Timestamp :    mean power consumption (MWT)
(Offset)
{msg}

                      """
        self.log.info(message)

        minAverPwr = min(self.d_averPwr.values())
        maxAverPwr = max(self.d_averPwr.values())
        rangeAverPwr = maxAverPwr - minAverPwr
        subrangeAverPwr = rangeAverPwr/self.num_classes
        self.l_left_bounds=[minAverPwr + subrangeAverPwr *i for i in range(self.num_classes)]
        self.l_right_bounds = [minAverPwr + subrangeAverPwr * (i +1) for i in range(self.num_classes)]
        self.l_left_bounds[0] =self.l_left_bounds[0] - subrangeAverPwr *0.1
        self.l_right_bounds[self.num_classes-1] = self.l_right_bounds[self.num_classes-1] + subrangeAverPwr * 0.1

        msg = ""
        n_class = 0
        for lf,rg in zip(self.l_left_bounds, self.l_right_bounds):
            msg = msg + "{:>5d} {:<10.4f} {:<10.4f}\n".format(n_class, lf,rg)
            n_class = n_class + 1
        message = f"""
Min Average Pwr(per segment)        : {minAverPwr}
Max Average Pwr(per segment)        : {maxAverPwr}
Number of classes (states)          : {self.num_classes}
Bin size ( first and last expanded) : {subrangeAverPwr}

           Bins
Class Left     Right
       Bound   Bound
{msg}

"""
        self.log.info(message)
        #################################################################################


        """ Explisit segment labeling according by rules
        Create Block objects for each segment. The Block contains a following data:
        -segment observations is a vector of SEGMENT_SIZE extracted from TS starting from start_block offset,
        -sampling in sec,
        - timestamp for begin of segment, 
        - index of segment,
        - flag, is not used now,
        - wavelet object for scalogram and other,
        - number of scales for wavelet,
        - label, bbelonging to class (state),
        - average value for segment observations.
        """
        count_abnormal =0
        for start_block in self.lstOffsetSegment:

            desire_label = 0
            for lf,rg in zip(self.l_left_bounds, self.l_right_bounds):
                if  self.d_averPwr[start_block]>=lf and self.d_averPwr[start_block]<rg:
                    break
                desire_label=desire_label+1
            if desire_label == self.num_classes:
                count_abnormal = count_abnormal + 1
                self.log.error("Abnormal values counter : {} Start_block : {}".format(count_abnormal, start_block))
                desire_label=self.num_classes -1      # last state

            self.lstBlocks.append(
                Block(x=self.y[start_block:start_block + self.segment_size],
                      sampling=self.sampling,
                      timestamp=self.dt[start_block],
                      index=start_block,
                      isTrain=True,
                      wav=self.wav,
                      scales=self.scales,
                      desire=desire_label,
                      average=self.d_averPwr[start_block])
            )

        return

    def avrObservationPerState(self):

        # accumulation in vector[segment_size]
        for item in self.lstBlocks:
            self.count[item.desire] = self.count[item.desire]+1
            self.aver_of_aver[item.desire] = self.aver_of_aver[item.desire] + item.average
            for j in range(self.segment_size):
                self.aver[item.desire][j] = self.aver[item.desire][j] +item.x[j]
        # averaging
        for i in range(self.num_classes):
            for j in range(self.segment_size):
                if self.count[i]>0:
                    self.aver[i][j] = self.aver[i][j] /self.count[i]
            self.aver_of_aver[i] = self.aver_of_aver[i]/self.count[i]

        msg = "\n"
        for item in range(self.num_classes):
            msg = msg + "State {:>5d}  :  {:<5d} hits. Average of averages is {:<10.4f}\n".format(item,
                        self.count[item], self.aver_of_aver[item])
            k =0
            for i in range(self.segment_size):
                msg = msg + "{:<4d}:{:<8.2} ".format(i, self.aver[item][i])
                k =k +1
                if k%8 == 0:
                    msg = msg + "\n"
            msg = msg + "\n"
        message = f""" Cluster center vectors  per State 
{msg}
"""
        self.log.info(message)
        return

    def scalogramEstimationCenter(self):
        pass
        for item in self.lstBlocks:

            item.scalogramEstimation(mode=1, title="Segment_{}_scalogram_{}_state".format(item.index, item.desire),
                                     chart_repository=self.chart_log)
        for i in range(self.num_classes):
            self.lstBlocks[0].scalogramEstimation( mode=2, y=self.aver[i], title="Average_for_{}_state".format(i),
                                                   chart_repository=self.chart_log)
    if __name__ == "__main__":
        pass
