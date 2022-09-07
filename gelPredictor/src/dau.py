#!/usr/bin/env python3

""" DAU (Data Aqusition Unit)"""

import numpy as np
import pandas as pd
import pywt
from sklearn.cluster import KMeans
import logging

from src.block import Block

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.25

class Dau(object):
    """Data Aquisiation Unit is responsible for communicating with the data source."""

    def __init__(self,ts:str="",dt:str="Date Time", sampling:int=10*60, n_steps:int = 144):
        pass
        self.log=logger
        self.ts_name = ts
        self.dt_name = dt
        self.sampling =sampling
        self.n_steps = n_steps
        self.scales =None
        self.frequencies = None
        self.wavelet = 'mexh'   #'cmor1.5-1.0'
        self.wav=pywt.ContinuousWavelet(self.wavelet)
        self.width =self.wav.upper_bound - self.wav.lower_bound
        self.max_wav_len = 0


class Dataset(Dau):

    def __init__(self, pathTo:str="",ts:str="",dt:str="Date Time", sampling:int=10*60, n_steps:int=144):
        super().__init__(ts,dt,sampling,n_steps)
        self.pathToCsv = pathTo
        self.df:pd.DataFrame = None
        self.y       = None
        self.dt      = None
        self.n       = 0
        self.mean    = 0.0
        self.std     = 1.0
        self.n_train = 0
        self.n_val   = 0
        self.n_test  = 0
        self.n_train_blocks = 0
        self.n_val_blocks = 0
        self.lstBlocks=[]
    def __str__(self):
        msg=f"""
        
        
Dataset   : {self.pathToCsv}
TS name   : {self.ts_name}  Timestamp labels : {self.dt_name}
TS mean   : {self.mean}     TS std : {self.std} TS length : {self.n} Sampling : {self.sampling} sec 
Block Size: {self.n_steps}  Train blocks : {self.n_train_blocks} Validation blocks : {self.n_val_blocks}
Train Size: {self.n_train}  Validation Size : {self.n_val}  Test Size: {self.n_test} 

Wavelet: {self.wav}
Scales : {self.scales}
Frequencies,Hz : {self.frequencies}
Wavelet wigth : {self.width} Max len :{self.max_wav_len }

"""
        self.log.info(msg)
        print(msg)
        return msg

    def readDataset(self):
        self.df=pd.read_csv(self.pathToCsv)
        self.n=len(self.df)
        self.y = self.df[self.ts_name].values
        self.dt = self.df[self.dt_name].values
        self.scales = [i + 1 for i in range(16)]
        self.frequencies = pywt.scale2frequency(self.wavelet, self.scales) / self.sampling
        self.max_wav_len = (0 if not self.scales else int(np.max(self.scales) * self.width) )

    def statnorm(self):

        (self.n,)=np.shape(self.y)
        self.mean=np.mean(self.y,axis=0)
        self.std = np.std(self.y,axis=0)
        for i in range(self.n):
            self.y[i]=(self.y[i]-self.mean)/self.std

    def setTrainValTest(self):
        nblocks=round(self.n/self.n_steps)
        self.n_train_blocks=round(TRAIN_RATIO*nblocks)
        self.n_val_blocks = round(VAL_RATIO*nblocks)
        self.n_train=self.n_steps * self.n_train_blocks
        self.n_val = self.n_steps * self.n_val_blocks
        self.n_test = self.n - self.n_train -self.n_val

    def crtListBlocks(self):
        start_block=0
        for i in range(self.n_train_blocks):
            self.lstBlocks.append(Block(x=self.y[start_block:start_block + self.n_steps], sampling=self.sampling,
                                        timestamp=self.dt[start_block], index=i, isTrain= True,
                                        wav=self.wav,scales=self.scales))
            start_block=start_block+self.n_steps

        for i in range(self.n_val_blocks):
            self.lstBlocks.append(Block(x=self.y[start_block:start_block + self.n_steps], sampling=self.sampling,
                                        timestamp=self.dt[start_block], index=i+self.n_train_blocks, isTrain= False,
                                        wav=self.wav,scales=self.scales))
            start_block=start_block+self.n_steps

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
if __name__ == "__main__":
    pass