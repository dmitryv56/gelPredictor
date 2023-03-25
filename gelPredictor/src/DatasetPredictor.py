#!/usr/bin/env python3

""" Dataset Predictor """

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
from src.hmm import hmm
from sys_util.parseConfig import STATE_0_MARGE, STATE_0, STATE_DEMAND, STATE_GENERATION
from src.dau import Dataset
logger = logging.getLogger(__name__)
FIRST_DAY_IN_WEEK = 1
LAST_DAY_IN_WEEK  = 7
FIRST_MONTH       = 1
LAST_MONTH        = 12

PWR_0_MIN = 5.0
PWR_1_MIN = 5.55
PWR_2_MIN = 5.9
PWR_2_MAX = 6.4


class DatasetPredictor(Dataset):

    def __init__(self, pathTo: str = "", ts: str = "", dt: str = "Date Time", sampling: int = 10 * 60,
                 n_steps: int = 144,
                 segment_size: int = 96, norm: str = "stat", overlap: int = 0, continuous_wavelet: str = 'mexh',
                 num_classes: int = 4, num_scales: int = 16, compress: str = 'pca', n_components: int = 2,
                 model_repository: Path = PATH_REPOSITORY, log_folder: Path = PATH_LOG_FOLDER,
                 chart_log: Path = PATH_CHART_LOG):
        """ Constructor """

        super().__init__(pathTo=pathTo, ts=ts, dt=dt, sampling=sampling, n_steps=n_steps, segment_size=segment_size,
                         norm=norm, overlap=overlap, continuous_wavelet=continuous_wavelet,
                         num_classes=num_classes, num_scales=num_scales, compress=compress,
                         n_components=n_components, model_repository=model_repository, log_folder=log_folder,
                         chart_log=chart_log)
        """ Constructor"""

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

    def createSegmentLst(self):

        """ Create list of offset for all segments of self.segment_size along train part of TS.
        The segments may overlap.
        """

        self.n4cnnTrain = self.n_train + self.n_val
        self.lstOffsetSegment = []
        n_seg = 0
        while (n_seg * self.segment_size <= self.n4cnnTrain):
            self.lstOffsetSegment.append(n_seg * self.segment_size)  # segments without overlap

            if self.overlap > 0:  # segments with overlap
                n_overlap = 1
                # 1: check overlap into segment bounds  2: end of overlapped segment into train TS bound
                while (n_overlap * self.overlap < self.segment_size) and \
                        (n_seg * self.segment_size + n_overlap * self.overlap + self.segment_size < self.n4cnnTrain):
                    self.lstOffsetSegment.append(n_seg * self.segment_size + n_overlap * self.overlap)
                    n_overlap = n_overlap + 1

            n_seg = n_seg + 1

        msg = ""

        self.lstOffsetSegment.sort()
        k = 0
        for item in self.lstOffsetSegment:
            msg = msg + "{:<6d} ".format(item)
            k = k + 1
            if k % 16 == 0:
                msg = msg + "\n"

        message = f"""
    The size of train part of TS for CNN learning: {self.n4cnnTrain} 
    The number of segments (blocks)              : {len(self.lstOffsetSegment)} 
    The overlap                                  : {self.overlap}

                           Segment offsets
    {msg}
            """
        print(message)
        self.log.info(message)
        for start_block in self.lstOffsetSegment:
            self.lstBlocks.append(
                Block(x=self.y[start_block:start_block + self.segment_size],
                      sampling=self.sampling,
                      timestamp=self.dt[start_block],
                      index=start_block,
                      isTrain=True,
                      wav=self.wav,
                      scales=self.scales)
            )

        # start_block=0
        # for i in range(self.n_train_blocks):
        #     self.lstBlocks.append(Block(x=self.y[start_block:start_block + self.segment_size], sampling=self.sampling,
        #                                 timestamp=self.dt[start_block], index=i, isTrain=True,
        #                                 wav=self.wav, scales=self.scales))
        #     start_block = start_block + self.segment_size
        #
        # for i in range(self.n_val_blocks):
        #     self.lstBlocks.append(Block(x=self.y[start_block:start_block + self.segment_size], sampling=self.sampling,
        #                                 timestamp=self.dt[start_block], index=i + self.n_train_blocks,
        #                                 isTrain=False,
        #                                 wav=self.wav, scales=self.scales))
        #     start_block = start_block + self.segment_size

    def ExtStatesExtraction(self):

        # if len(self.lstOffsetSegment )==0:
        #     self.StatesExtraction()
        #     return

        X = np.zeros(shape=(len(self.lstOffsetSegment), self.segment_size))
        (n, m) = X.shape
        for i in range(n):
            for j in range(m):
                X[i, j] = self.y[self.lstOffsetSegment[i] + j]

        # transformation by PCA to compress data
        if self.compress == "pca":
            self.log.info(
                "P(rincipial) C(omponent) A(nalysis) method is used for compress to data till {} components\n". \
                    format(self.n_components))

            pca = PCA(n_components=self.n_components)
            obj = pca.fit(X)
            self.log.info("PCA object for transformation\n{}\n".format(obj))

            Xpca = pca.fit_transform(X)
            self.log.info("compressed Data\n{}".format(Xpca))
            file_png = str(Path(Path(self.chart_log) / Path("KMeans_clasterization_PCA")).with_suffix(".png"))
            kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(Xpca)
            plotClusters(kmeans, Xpca, file_png)
        else:
            kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(X)
            file_png = str(Path(Path(self.chart_log) / Path("KMeans clasterization")).with_suffix(".png"))
            plotClusters(kmeans, X, file_png)

        for i in range(n):
            self.lstBlocks[i].desire = kmeans.labels_[i]

        # generation blocks for centers
        for i in range(self.num_classes):
            blck = Block(x=kmeans.cluster_centers_[i, :],
                         sampling=self.sampling,
                         timestamp="N/A",
                         index=i,
                         isTrain=True,
                         wav=self.wav,
                         scales=self.scales,
                         desire=i)
            blck.scalogramEstimation()
            title = "Scalogram  Center Class_{}".format(i)
            file_png = str(Path(Path(self.chart_log) / Path(title)).with_suffix(".png"))
            simpleScalingImgShow(scalogram=blck.scalogram, index=i, title=title, file_png=file_png)
        return