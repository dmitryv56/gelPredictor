#!/usr/bin/env python3

import logging
import pandas as pd
from pathlib import Path
import numpy as np

from src.block import Block
from sys_util.parseConfig import SHRT_MIN_TRAIN_DATA_SIZE, SHRT_TRAIN_PART, SHRT_VAL_PART, TS_NAME
from sys_util.utils import ts2supervisedLearningData

logger = logging.getLogger(__name__)

class VeryShortTerm(object):
    """ Prepare data sets per cluster (state) for very short-term forecasting  """

    def __init__(self, num_classes:int=5, segment_size:int=96, n_steps: int = 32, df:pd.DataFrame = None,
                 dt_name:str="Date Time", ts_name:str="Power", exogen_list:list=[],  list_block:list=[],
                 repository_path:Path=None, model_repository_path: Path = None, chart_repository_path:Path = None):
        """ Constructor """

        self.log = logger
        self.num_classes = num_classes
        self.segment_size = segment_size
        self.n_steps = n_steps
        self.df = df
        self.list_block = list_block
        self.dt_name=dt_name
        self.ts_name = ts_name
        self.exogen_list = exogen_list
        self.repository_path = repository_path
        self.model_repository_path = model_repository_path
        self.chart_repository_path = chart_repository_path
        self.d_df={}
        self.d_size ={}
        self.d_data_df = {}     # not be used

    def __str__(self):
        """ log class implementation"""

        msg = f"""
        
Class : {self.__class__.__name__}
Logger : {self.log.__str__()}
Number of states(classes): {self.num_classes}
Segment (block) size     : {self.segment_size}
N_steps for Supervised Learning Data: {self.n_steps}
Data Frame object                   : {self.df.__str__()}
List Blocks (each block is object)  : {self.list_block}
Time Series name                    : {self.ts_name}
Shapshots label name                : {self.dt_name}
Exogenius Factor List               : {self.exogen_list}
Repository (syntetic TS) path       : {self.repository_path}  
Repository ANN Models Path          : {self.model_repository_path}
Repository Chart Path               : {self.chart_repository_path}
Class:Path to csv (dict)            : {self.d_df}
Class:Size of dataset (dict)        : {self.d_size}
Class:Data Frame (dict)             : {self.d_data_df}

"""
        print(msg)
        self.log.info(msg)
        return

    def createDS(self, class_label:int=0):
        """ create dataset for class from source DataFrame.
        Save dataset (crated DataFrame object) as csv-file.
        The pairs {class: path to csv}  and {class: length of dataset} added into 'd_df' and d_size directories are
        members of the class.

        Note: This data is not normalized"""

        if self.df is None:
            self.log.error("DataFrame is no valid")
            return
        if not self.list_block or len(self.list_block)==0:
            self.log.error("Segment list is not valid")
            return
        if self.repository_path is None:
            dataset_path = Path("shrtrm_class_{}".format(class_label)).with_suffix(".csv")
        else:
            dataset_path=Path( self.repository_path / Path("shrtrm_class_{}".format(class_label))).with_suffix(".csv")

        dt=[]
        dv=[]
        exogen=[]

        for item in self.exogen_list:
            exogen.append([])
        for block in self.list_block:
            if block.desire != class_label:
                continue
            start=block.index
            for i in range(self.segment_size):
                dt.append(self.df[self.dt_name].values[start+i])
                dv.append(self.df[self.ts_name].values[start+i])
                k=0
                for item in self.exogen_list:
                    exogen[k].append(self.df[item].values[start+i])
                    k=k+1
            pass
        pass

        dd={self.dt_name:dt[:],self.ts_name:dv[:]}
        k=0
        for item in self.exogen_list:
            dd["e{}".format(item)]=exogen[k][:]
            k=k+1

        df1= pd.DataFrame(dd)
        self.d_df[class_label] = dataset_path
        self.d_size[class_label]=len(df1)

        df1.to_csv(dataset_path)
        self.log.info("Class :{} Sample size: {} Repository: {}".format(class_label,len(df1),dataset_path))
        return

    def createTrainData(self,class_label:int=-1)->(np.array,np.array, np.array, np.array):
        '''
        Create train and validate data for given class (state).
        :param class_label: [in] label of class (index)
        :return: X -
        '''


        # read data set from serialized Pandas' DataFrame object
        len_df_data = self.d_size[class_label]
        n_train = round(len_df_data*SHRT_TRAIN_PART)
        n_val= len_df_data - n_train
        if n_train < SHRT_MIN_TRAIN_DATA_SIZE:
            self.log.error("For {} class is not enough data for ANN training, size is {}".format(class_label,
                                                                                                 len_df_data))
            self.log.error("Exit!")
            return (None,None,None,None)

        path_to_serialized_df = self.d_df[class_label]
        df = pd.read_csv(path_to_serialized_df)
        xx=np.array(df[TS_NAME].values)
        x=xx[:n_train]
        x_val = xx[n_train:]
        del xx
        X, y = self.ts2supervisedLearningData( x=x)
        X_val, y_val = self.ts2supervisedLearningData(x=x)

        return X, y, X_val, y_val


    def ts2supervisedLearningData(self, x:np.array=None)->(np.array, np.array):
        '''
        Transform vector observations to supervised learning data matrix X and desired output y.
        :param x:  [in] vector observation of 'n' -length.
        :return: X-supervised learning matrix of (n-self.n_steps, self.n_steps) shape,
                 y- desired output vector of (n-self.n_steps) shape.
        '''

        X, y = ts2supervisedLearningData(x=x, n_steps=self.n_steps )
        return X,y



