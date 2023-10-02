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
from PIL import Image
import PIL

from src.block import Block
from sys_util.parseConfig import PATH_LOG_FOLDER , PATH_REPOSITORY, PATH_CHART_LOG, TRAIN_RATIO, VAL_RATIO, \
    TS_DEMAND_NAME, PATH_WV_IMAGES, DETREND
from sys_util.utils import simpleScalingImgShow
from src.hmm import hmm_demand
from sys_util.parseConfig import STATE_0_MARGE, STATE_0, STATE_DEMAND, STATE_GENERATION
from src.dau import Dataset, plotClusters
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

class DatasetSVSPred(Dataset):

    def __init__(self, pathTo: str = "", ts: str = "", dt: str = "Date Time", sampling: int = 10 * 60,
                 n_steps: int = 144,
                 segment_size: int = 96, norm: str = "stat", overlap: int = 0, continuous_wavelet: str = 'mexh',
                 num_classes: int = 4, num_scales: int = 16, compress: str = 'pca', n_components: int = 2,
                 detrend : bool = False, model_repository: Path = PATH_REPOSITORY, log_folder: Path = PATH_LOG_FOLDER,
                 chart_log: Path = PATH_CHART_LOG, wavelet_image = PATH_WV_IMAGES):
        """ Constructor """
        self.n4cnnTrain = 0
        super().__init__(pathTo=pathTo, ts=ts, dt=dt, sampling=sampling, n_steps=n_steps, segment_size=segment_size,
                         norm=norm, overlap=overlap, continuous_wavelet=continuous_wavelet,
                         num_classes=num_classes, num_scales=num_scales, compress=compress, detrend=detrend,
                         n_components=n_components, model_repository=model_repository, log_folder=log_folder,
                         chart_log=chart_log, wavelet_image = wavelet_image)

        self.d_averPwr = {}
        self.n4cnnTrain = 0
        self.aver = np.zeros((self.num_classes, self.segment_size), dtype=float)
        self.count = np.zeros((self.num_classes), dtype=int)
        self.aver_of_aver = np.zeros((self.num_classes), dtype=float)
        self.hmm = hmm_demand(dt=self.dt,log_folder=self.log_folder)

        return


    """ This method aims on the aggregation a day's observations in the segments (blocks). Each block has desired label 
          according by belongs the day average power demand to one of ranges 5.0-5.55, 5.55-5.9,5.9-6.4 MWT 
          The average for each day is calculated and saved in the dict{offset day : aver}.
          Because day is vector of n_day samples, then offsets are 0, n_day,2*n_day, ....
          For example, for discretization is 10 minutes ,  n_day = 144 samples.
    """

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
            msg = msg + "          {:<6d} {:<24s} :{:<10.4f} ".format(item,  self.dt[item],self.d_averPwr[item])
            k = k + 1
            if k % 4 == 0:
                msg = msg + "\n"

        message = f"""
The size of train part of TS for CNN learning: {self.n4cnnTrain} 
The number of segments (blocks)              : {len(self.lstOffsetSegment)} 
The overlap                                  : {self.overlap}

Segment (Offset) Timestamp                :    mean power consumption (MWT)
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
        file_log = Path(Path(self.log_folder)/Path("state_range")).with_suffix(".log")
        with open(file_log, 'w') as fout:
            fout.write(message)
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
        - label, belonging to class (state),
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
        self.log.info("The list of the segments (blocks) has been created.")
        return

    def printAveragePerBlock(self):
        y_list=[]
        x_list=[]
        state_list=[]
        message="{:>5s} {:<30s} {:<12s} {:<5s}\n\n".format("#####", "Timestamp", "Day Average", "State")
        ind=0
        for item in self.lstBlocks:
            msg_row="{:>5d} {:<30s} {:<10.4f} {:>3d}\n".format(ind,item.timestamp, item.average, item.desire)
            y_list.append(item.average)
            x_list.append(item.timestamp)
            state_list.append(item.desire)
            ind = ind +1
            message=message + msg_row
        file_log = Path(Path(self.log_folder)/Path("Segment_Average_Values")).with_suffix(".log")
        with open(file_log,'w') as fout:
            fout.write(message)
        self.log.info(" Segment average values has been put in {}.".format(file_log))
        return


    def avrObservationPerState(self):
        """  The Average per segments belongs to same state is calculated.

        :return:
        """
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
                msg = msg + "{:<4d}           :     {:<10.4f} ".format(i, self.aver[item][i])
                k =k +1
                if k%8 == 0:
                    msg = msg + "\n"
            msg = msg + "\n"
        message = f""" Cluster center vectors  per State 
{msg}
"""
        file_log = Path(Path(self.log_folder) / Path("Cluster_center_vectors_per_State ")).with_suffix(".txt")
        with open(file_log, 'w') as fout:
            fout.write(msg)
        self.log.info(message)
        self.log.info(" Cluster center vectors  per State  has been put in {}.".format(file_log))
        return

    def scalogramEstimationCenter(self):
        pass
        for item in self.lstBlocks:

            item.scalogramEstimation(mode=1, title="Segment_{}_scalogram_{}_state".format(item.index, item.desire),
                                     chart_repository=self.wavelet_image)
        for i in range(self.num_classes):
            self.lstBlocks[0].scalogramEstimation( mode=2, y=self.aver[i], title="Average_for_{}_state".format(i),
                                                   chart_repository=self.wavelet_image)



class DatasetSTF(DatasetSVSPred):

    def __init__(self, pathTo: str = "", ts: str = "", dt: str = "Date Time", sampling: int = 10 * 60,
                 n_steps: int = 144,
                 segment_size: int = 96, norm: str = "stat", overlap: int = 0, continuous_wavelet: str = 'mexh',
                 num_classes: int = 4, num_scales: int = 16, compress: str = 'pca', n_components: int = 2,
                 detrend : bool = False, model_repository: Path = PATH_REPOSITORY, log_folder: Path = PATH_LOG_FOLDER,
                 chart_log: Path = PATH_CHART_LOG, wavelet_image = PATH_WV_IMAGES, train_folder: Path= None,
                 test_folder: Path =None):

        """ Constructor """
        self.n4cnnTrain = 0
        super().__init__(pathTo=pathTo, ts=ts, dt=dt, sampling=sampling, n_steps=n_steps, segment_size=segment_size,
                         norm=norm, overlap=overlap, continuous_wavelet=continuous_wavelet,
                         num_classes=num_classes, num_scales=num_scales, compress=compress, detrend=detrend,
                         n_components=n_components, model_repository=model_repository, log_folder=log_folder,
                         chart_log=chart_log, wavelet_image = wavelet_image)

        self.d_averPwr = {}
        self.n4cnnTrain = 0
        self.aver = np.zeros((self.num_classes, self.segment_size), dtype=float)
        self.count = np.zeros((self.num_classes), dtype=int)
        self.aver_of_aver = np.zeros((self.num_classes), dtype=float)
        self.hmm = hmm_demand(dt=self.dt,log_folder=self.log_folder)

        self.Y = None
        self.stateSequence=[]
        self.desire = []
        self.train_folder = Path("train_folder") if train_folder is None else Path (train_folder)
        self.test_folder  = Path("test_folder") if test_folder is None else Path(test_folder)
        self.training_set = Path(self.train_folder) / Path("training_set")
        self.testing_set  = Path(self.test_folder) / Path("testing_set")
        self.training_set.mkdir(parents=True, exist_ok=True)
        self.testing_set.mkdir( parents=True, exist_ok=True)
        self.trainingDS   = []
        self.testingDS    = []
        return

    def ts2Matrix(self):
        """

        :return:
        """
        self.n_samples = int(self.n/self.segment_size)
        self.Y = self.y[:self.n_samples * self.segment_size].reshape(self.n_samples, self.segment_size)

    def getStates(self)->(KMeans,np.array):
        # transformation by PCA to compress data
        if self.compress == "pca":
            self.log.info(
                "P(rincipial) C(omponent) A(nalysis) method is used for compress to data till {} components\n". \
                    format(self.n_components))

            pca = PCA(n_components=self.n_components)
            obj = pca.fit(self.Y)
            self.log.info("PCA object for transformation\n{}\n".format(obj))

            Xpca = pca.fit_transform(self.Y)
            self.log.info("compressed Data\n{}".format(Xpca))
            file_png = str(Path(Path(self.chart_log) / Path("KMeans_clasterization_PCA")).with_suffix(".png"))
            kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(Xpca)
            plotClusters(kmeans, Xpca, file_png)
        else:
            kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(self.X)
            file_png = str(Path(Path(self.chart_log) / Path("KMeans clasterization")).with_suffix(".png"))
            plotClusters(kmeans, self.Y, file_png)

        for i in range(self.n_samples):
            self.stateSequence.append(kmeans.labels_[i])
            self.desire.append(kmeans.labels_[(i+1) % self.n_samples])   # last desire item is undefined, instead first

        return kmeans, Xpca

    def cluster_centers_after_pca(self, kmeans:KMeans = None, Xpca:np.array =None)->np.array:
        # generation blocks for centers
        after_pca_centers = None
        if self.compress == "pca":
            pca_centers = kmeans.cluster_centers_
            after_pca_centers = self.getCentersAfterPCA(pca_centers=pca_centers)
            self.aux_log(X=self.Y, Xpca=Xpca, after_pca_centers=after_pca_centers)
        pass
        return  after_pca_centers

    def setSegmentList(self):

        for i in range (self.n_samples):
            start_block = i * self.segment_size
            self.lstBlocks.append(
                Block(x=self.Y[i,:] , # self.y[start_block:start_block + self.segment_size],
                      sampling=self.sampling,
                      timestamp=self.dt[start_block],
                      index=start_block,
                      isTrain=True,
                      wav=self.wav,
                      scales=self.scales,
                      desire=self.stateSequence[i]
                      ))
        return

    def datasetFolders(self):
        for i in range(self.num_classes):
            fld = Path(self.training_set)/ Path("{}".format(i))
            fld.mkdir(parents=True, exist_ok=True)
            self.trainingDS.append(fld)
            fld = Path(self.testing_set) / Path("{}".format(i))
            fld.mkdir(parents=True, exist_ok=True)
            self.testingDS.append(fld)

    def createImageDS(self):
        ntrain=int(self.n_samples*0.7)
        ntest = self.n_samples - ntrain

        for i in range(self.n_samples):
            # class
            fig, ax = plt.subplots()
            file_name = Path(self.trainingDS[self.stateSequence[i]])/Path("{}".format(i)).with_suffix(".png") if i < ntrain \
                        else Path(self.testingDS[self.stateSequence[i]])/Path("{}".format(i)).with_suffix(".png")

            scalogram, freq = self.lstBlocks[i].scalogramEstimation( )
            img = ax.imshow(scalogram, interpolation='nearest')

            plt.axis('off')
            plt.savefig(file_name, bbox_inches='tight')
            plt.close("all")




import math
if __name__ == "__main__":

    x=np.array( [math.sin(2.0* math.pi *a /144.0)+0.2 for a in range(288) ])
    n_scale=36
    scalogram, freqs = pywt.cwt(x, np.arange(1,36), "mexh")
    fig, ax = plt.subplots()
    plt.figure()
    (n_scale, shift) = scalogram.shape
    # extent_lst = [0, shift, 1, n_scale]
    # ax.imshow(scalogram,
    #           extent=extent_lst,  # extent=[-1, 1, 1, 31],
    #           cmap='PRGn', aspect='auto',
    #           vmax=abs(scalogram).max(),
    #           vmin=-abs(scalogram).max())
    plt.matshow(scalogram)
    plt.savefig("{}__.png".format("____aaa.png"))
    img = plt.imshow(scalogram, interpolation='nearest')
    # img.set_cmap('hot')
    plt.axis('off')
    plt.savefig("___c.png", bbox_inches='tight')
    plt.close("all")
    img = Image.fromarray(scalogram, mode="RGB")
    img.save("___b.png")
