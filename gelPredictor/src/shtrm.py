#!/usr/bin/env python3

import logging
import pandas as pd
from pathlib import Path

from src.block import Block

logger = logging.getLogger(__name__)

class ShortTerm(object):
    """ Prepare data sets per cluster (state) for short term forecasting  """

    def __init__(self, num_classes:int=5, segment_size:int=96, df:pd.DataFrame = None, dt_name:str="Date Time",
                 ts_name:str="Power", exogen_list:list=[],  list_block:list=[], repository_path:Path=None):
        """ Constructor """
        self.log = logger
        self.num_classes = num_classes
        self.segment_size = segment_size
        self.df = df
        self.list_block = list_block
        self.dt_name=dt_name
        self.ts_name = ts_name
        self.exogen_list = exogen_list
        self.repository_path = repository_path
        self.d_df={}
        self.d_size ={}

    def createDS(self, class_label:int=0):
        """ create data for class for source data frame"""

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