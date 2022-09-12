#!/usr/bin/env python3

""" DAU (Data Aqusition Unit)"""

from pathlib import Path


import numpy as np
import pandas as pd
import pywt
from sklearn.cluster import KMeans
import logging

from src.block import Block
from sys_util.parseConfig import PATH_LOG_FOLDER , PATH_REPOSITORY, PATH_CHART_LOG

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.25



class Dau(object):
    """Data Aquisiation Unit is responsible for communicating with the data source."""

    def __init__(self,ts: str = "",dt: str = "Date Time", sampling: int = 10*60, n_steps: int = 144, norm: str = "stat",
                 overlap: int = 0, continuous_wavelet: str = 'mexh', num_classes:int = 4, num_scales:int = 16,
                 model_repository:Path = None,  log_folder:Path = None, chart_log: Path = None):
        pass
        self.log=logger
        self.ts_name = ts
        self.dt_name = dt
        self.sampling = sampling
        self.n_steps = n_steps
        self.num_scales =num_scales
        self.scales = [i + 1 for i in range(self.num_scales)]
        self.frequencies = None
        self.wavelet = continuous_wavelet   #   'mexh'   #'cmor1.5-1.0'
        self.wav = pywt.ContinuousWavelet(self.wavelet)
        self.width = self.wav.upper_bound - self.wav.lower_bound
        self.max_wav_len = 0
        self.norm = norm
        self.overlap = overlap
        self.num_classes=num_classes
        self.model_repository = model_repository
        self.log_folder = log_folder
        self.chart_log = chart_log




class Dataset(Dau):

    def __init__(self, pathTo:str="",ts:str="",dt:str="Date Time", sampling:int=10*60, n_steps:int=144,norm:str="stat",
                 overlap:int=0, continuous_wavelet: str = 'mexh',num_classes:int=4, num_scales:int = 16,
                 model_repository:Path = PATH_REPOSITORY, log_folder:Path = PATH_LOG_FOLDER,
                 chart_log: Path = PATH_CHART_LOG):


        super().__init__(ts = ts, dt=dt,sampling = sampling, n_steps = n_steps, norm = norm, overlap = overlap,
                         continuous_wavelet = continuous_wavelet, num_classes =num_classes, num_scales = num_scales,
                         model_repository = model_repository, log_folder = log_folder, chart_log = chart_log)
        self.pathToCsv = pathTo
        self.df:pd.DataFrame = None
        self.y       = None
        self.dt      = None
        self.n       = 0
        self.mean    = 0.0
        self.std     = 1.0
        self.min     = 0.0
        self.max     = 1.0
        self.n_train = 0
        self.n_val   = 0
        self.n_test  = 0
        self.n_train_blocks = 0
        self.n_val_blocks = 0
        self.lstBlocks=[]
        self.lstOffsetSegment = []


    def __str__(self):
        msg=f"""
        
        
Dataset   : {self.pathToCsv}
TS name   : {self.ts_name}  Timestamp labels : {self.dt_name} Data Normalization : {self.norm}
TS mean   : {self.mean}     TS std : {self.std} TS length : {self.n} Sampling : {self.sampling} sec 
Block Size: {self.n_steps}  Train blocks : {self.n_train_blocks} Validation blocks : {self.n_val_blocks}
Train Size: {self.n_train}  Validation Size : {self.n_val}  Test Size: {self.n_test} 

Wavelet: {self.wav}
Scales : {self.scales}
Frequencies,Hz : {self.frequencies}
Wavelet wigth : {self.width} Max len :{self.max_wav_len }

Model Repository : {self.model_repository}
Aux Log Folder   : {self.log_folder}
Charts           : {self.chart_log}

"""
        self.log.info(msg)
        print(msg)
        return msg

    def readDataset(self):
        self.df=pd.read_csv(self.pathToCsv)
        self.n=len(self.df)
        self.y = self.df[self.ts_name].values
        self.dt = self.df[self.dt_name].values

        self.frequencies = pywt.scale2frequency(self.wavelet, self.scales) / self.sampling
        self.max_wav_len = (0 if not self.scales else int(np.max(self.scales) * self.width) )

    def data_normalization(self):

        (self.n,)=np.shape(self.y)
        self.mean=np.mean(self.y,axis = 0)
        self.min = np.min(self.y,axis = 0)
        self.max = np.max(self.y, axis=0)
        self.std = np.std(self.y,axis = 0)

        if self.norm == "stat":
            for i in range(self.n):
                self.y[i]=(self.y[i]-self.mean)/self.std
        elif self.norm == "norm":
            for i in range(self.n):
                self.y[i]=(self.y[i]-self.min)/(self.max-self.min)
        else:
            pass

    def data_inv_normalization(self):
        
        if self.norm == "stat":
            for i in range(self.n):
                self.y[i] = self.y[i] * self.std +self.mean
        elif self.norm == "norm":
            for i in range(self.n):
                self.y[i] = self.y[i] * (self.max - self.min) + self.min
        else:
            pass


    def setTrainValTest(self):

        nblocks=round(self.n/self.n_steps)
        self.n_train_blocks=round(TRAIN_RATIO * nblocks)
        self.n_val_blocks = round(VAL_RATIO * nblocks)
        self.n_train=self.n_steps * self.n_train_blocks
        self.n_val = self.n_steps * self.n_val_blocks
        self.n_test = self.n - self.n_train -self.n_val

    def createSegmentLst(self):

        """ Create list of offset for all segments of self.n_steps along train part of TS.
        The segments may overlap.
        """

        self.n4cnnTrain = self.n_train + self.n_val
        self.lstOffsetSegment = []
        n_seg=0
        while (n_seg * self.n_steps <=self.n4cnnTrain):
            self.lstOffsetSegment.append(n_seg * self.n_steps)  # segments without overlap

            if self.overlap >0:       # segments with overlap
                n_overlap=1
                # 1: check overlap into segment bounds  2: end of overlapped segment into train TS bound
                while ( n_overlap * self.overlap < self.n_steps ) and \
                      ( n_seg * self.n_steps + n_overlap * self.overlap + self.n_steps < self.n4cnnTrain):
                    self.lstOffsetSegment.append(n_seg * self.n_steps + n_overlap * self.overlap)
                    n_overlap=n_overlap + 1

            n_seg = n_seg +1

        msg = ""

        self.lstOffsetSegment.sort()
        k =0
        for item in self.lstOffsetSegment:
            msg = msg + "{:<6d} ".format(item)
            k=k+1
            if k % 16 == 0:
                msg=msg +"\n"

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
                Block(x = self.y[start_block:start_block + self.n_steps],
                      sampling = self.sampling,
                      timestamp = self.dt[start_block],
                      index = start_block,
                      isTrain = True,
                      wav = self.wav,
                      scales = self.scales)
            )


        # start_block=0
        # for i in range(self.n_train_blocks):
        #     self.lstBlocks.append(Block(x=self.y[start_block:start_block + self.n_steps], sampling=self.sampling,
        #                                 timestamp=self.dt[start_block], index=i, isTrain=True,
        #                                 wav=self.wav, scales=self.scales))
        #     start_block = start_block + self.n_steps
        #
        # for i in range(self.n_val_blocks):
        #     self.lstBlocks.append(Block(x=self.y[start_block:start_block + self.n_steps], sampling=self.sampling,
        #                                 timestamp=self.dt[start_block], index=i + self.n_train_blocks,
        #                                 isTrain=False,
        #                                 wav=self.wav, scales=self.scales))
        #     start_block = start_block + self.n_steps


    def ExtStatesExtraction(self):

        if len(self.lstOffsetSegment )==0:
            self.StatesExtraction()
            return

        X=np.zeros(shape=(len(self.lstOffsetSegment),self.n_steps))
        (n,m) =X.shape
        for i in range(n):
            for j in range(m):
                X[i,j] =self.y[self.lstOffsetSegment[i]+j]

        kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(X)

        for i in range(n):
            self.lstBlocks[i].desire = kmeans.labels_[i]



    def StatesExtraction(self):

        n1 = self.n_train_blocks + self.n_val_blocks
        m1 = self.n_steps
        X = self.y[:n1 * m1].reshape((n1, m1))
        kmeans = KMeans(n_clusters=4, random_state =0).fit(X)

        for i in range(n1):
            self.lstBlocks[i].desire = kmeans.labels_[i]

        message = "\nBlock  Is Train State"
        for i in range(n1):
            msg = f"""
{i}    {self.lstBlocks[i].isTrain}   {self.lstBlocks[i].desire} """
            message = message + msg

        self.log.info(message)
        for i in range(n1):
            self.lstBlocks[i].scalogramEstimation()

    def Data4CNN(self)->(np.ndarray,np.ndarray):
        n=len(self.lstBlocks)
        (m1, m2) = self.lstBlocks[0].scalogram.shape
        X=np.zeros(shape=(n,len(self.scales),self.n_steps)).astype( dtype=np.float32)
        Y=np.zeros(shape=(n),dtype=np.int32)
        for k in range(n):
            Y[k]=self.lstBlocks[k].desire
            (m1,m2)=self.lstBlocks[k].scalogram.shape
            for i in range (m1):
                for j in range(m2):
                    X[k,i,j]=self.lstBlocks[k].scalogram[i,j]

        return X,Y

    def logClasses(self):
        self.log.info("segment - class log printing...")
        print("segment - class log printing...")
        outFile = Path(self.log_folder /"segment_class").with_suffix(".txt")
        with open(outFile, 'w') as outf:
            header="\n#### Segment  Class Timestamp\nStart\nOffst\n==================================================\n"
            outf.write(header)
            i=0
            for item in self.lstBlocks:
                msg ="{:<4d} {:<5d}     {:<5d}  {:<30s}\n".format(i, item.index, item.desire, item.timestamp)
                outf.write(msg)
                i=i+1
        self.log.info("class - segment  log printing...")
        print("class - segment  log printing...")
        outFile = Path(self.log_folder / "class_segment").with_suffix(".txt")
        with open(outFile, 'w') as outf:
            header = "\n#### Class Segment   Timestamp\n      Start\n      Offst\n==================================================\n"
            outf.write(header)
            i=0
            for class_index in range(self.num_classes):
                for item in self.lstBlocks:
                    if item.desire!=class_index:
                        continue
                    msg = "{:<4d} {:<5d}  {:<5d}     {:<30s}\n".format(i,item.desire, item.index,item.timestamp)
                    outf.write(msg)
                    i=i+1

        self.log.info("class - segment -class logs finished")
        print("class - segment -class logs finished")
        return

    def scalogramEstimation(self):
        pass
        for item in self.lstBlocks:
            item.scalogramEstimation()

if __name__ == "__main__":
    pass