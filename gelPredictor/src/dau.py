#!/usr/bin/env python3

""" DAU (Data Aqusiation Unit)"""

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
    TRAIN_FOLDER, AUX_TRAIN_FOLDER, PATH_SHRT_CHARTS, TS_DEMAND_NAME, PATH_WV_IMAGES
from sys_util.utils import simpleScalingImgShow, matrix2string
from src.hmm import hmm
from sys_util.parseConfig import STATE_0_MARGE, STATE_0, STATE_DEMAND, STATE_GENERATION

logger = logging.getLogger(__name__)
FIRST_DAY_IN_WEEK = 1
LAST_DAY_IN_WEEK  = 7
FIRST_MONTH       = 1
LAST_MONTH        = 12

class Dau(object):
    """Data Aquisiation Unit is responsible for communicating with the data source.
    The class contains all hypeparameters for time series processing.
    """

    def __init__(self,ts: str = "",dt: str = "Date Time", sampling: int = 10*60, n_steps: int = 144,
                 segment_size: int = 96,norm: str = "stat", overlap: int = 0, continuous_wavelet: str = 'mexh',
                 num_classes: int = 4, num_scales: int = 16, compress: str = 'pca', n_components: int = 2,
                 model_repository: Path = None,  log_folder: Path = None, chart_log: Path = None,
                 wavelet_image : Path = None):
        """ Constructor """

        self.log = logger
        self.ts_name = ts
        self.dt_name = dt
        self.sampling = sampling
        self.n_steps = n_steps
        self.segment_size = segment_size
        self.num_scales = num_scales
        self.scales = [i + 1 for i in range(self.num_scales)]
        self.frequencies = None
        self.wavelet = continuous_wavelet
        #   'mexh'   'cmor1.5-1.0'
        self.wav = pywt.ContinuousWavelet(self.wavelet)
        self.width = self.wav.upper_bound - self.wav.lower_bound
        self.max_wav_len = 0
        self.norm = norm
        self.overlap = overlap
        self.num_classes = num_classes
        self.compress = compress
        self.n_components = n_components
        self.model_repository = model_repository
        self.log_folder = log_folder
        self.chart_log = chart_log
        self.wavelet_image = wavelet_image


class Dataset(Dau):
    """ Class for csv -file reading and processing.
    Class members:
    pathToCsv - path to file with historical observations.
    df - pandas' DataFrame.
    y - time series (TS), historical observations.
    dt - time labels (timestamps) for observation.
    n - size of TS.
    mean, std,min, max -simple stat.parameters of TS.
    n_train, n_val, n_test -split TS of size n on train sequence of size n_train, validation and test sequences.
    n_train_blocks, n_val_blocks - TS splits on segments (blocks) are using for mid term forecasting.
    lstBlocks - the list which contains block objects (implementation of Block class.
    lstOffsetSegment -the which contains the offset of segments in  TS for each block.
    hmm - hidden markov model object

    Class methods:
    __init__  - constructor
    __str__
    readDataset
    data_normalization
    data_inv_normalization
    setTrainValTest
    createSegmentLst
    ExtStatesExtraction
    Data4CNN
    initHMM_logClasses
    scalogramEstimation
    """

    def __init__(self, pathTo: str = "", ts: str = "", dt: str = "Date Time", sampling: int = 10*60, n_steps: int = 144,
                 segment_size: int = 96, norm: str = "stat", overlap: int = 0, continuous_wavelet: str = 'mexh',
                 num_classes: int = 4, num_scales: int = 16, compress: str = 'pca', n_components: int = 2,
                 detrend: bool =False, model_repository: Path = PATH_REPOSITORY, log_folder: Path = PATH_LOG_FOLDER,
                 chart_log: Path = PATH_CHART_LOG, wavelet_image = PATH_WV_IMAGES):
        """ Constructor """


        super().__init__(ts=ts, dt =dt, sampling = sampling, n_steps = n_steps, segment_size = segment_size,
                         norm = norm, overlap = overlap, continuous_wavelet = continuous_wavelet,
                         num_classes = num_classes, num_scales = num_scales, compress = compress,
                         n_components = n_components, model_repository = model_repository, log_folder = log_folder,
                         chart_log = chart_log, wavelet_image = wavelet_image)
        """ Constructor"""

        self.pathToCsv = pathTo
        self.df: pd.DataFrame = None
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
        self.hmm = hmm()
        self.printed = False
        self.detrend = detrend
        self.f_Naik  = 0.0
        self.delta_f = 0.0
   
    def __str__(self):
        msg=f"""
        
        
Dataset              : {self.pathToCsv}
TS name              : {self.ts_name}  Timestamp labels : {self.dt_name} Data Normalization : {self.norm}
TS mean              : {self.mean}     TS std : {self.std} TS length : {self.n} Sampling : {self.sampling} sec 
Detrend TS (High Pass Filter y[t]:=y[t]-y[t-1]): {self.deternd} 
Segment Size         : {self.segment_size}  Train blocks : {self.n_train_blocks} Validation blocks : {self.n_val_blocks}
Train Size           : {self.n_train}  Validation Size : {self.n_val}  Test Size: {self.n_test} 

Wavelet              : {self.wav}
Scales               : {self.scales}
Frequencies,Hz       : {self.frequencies}
Wavelet width        : {self.width} Max len :{self.max_wav_len }

Classification
Data Compress Method : {self.compress}
Number components    : {self.n_components}

Model Repository     : {self.model_repository}
Aux Log Folder       : {self.log_folder}
Charts               : {self.chart_log}

"""
        self.log.info(msg)

        return msg

    def readDataset(self):
        self.df=pd.read_csv(self.pathToCsv)
        self.n=len(self.df)
        self.y = self.df[self.ts_name].values
        self.dt = self.df[self.dt_name].values

        self.frequencies = pywt.scale2frequency(self.wavelet, self.scales) / self.sampling
        self.max_wav_len = (0 if not self.scales else int(np.max(self.scales) * self.width) )

        min_y = self.y.min()
        ind_min =self.y.argmin()
        max_y = self.y.max()
        ind_max = self.y.argmax()
        aver  = self.y.mean()
        std   = self.y.std()
        self.f_Naik= 1.0/(2.0 *self.sampling)
        self.delta_f =1.0/(self.n * self.sampling)

        message = f"""  
        
        Time Series {self.ts_name } 
Aquisited (from/to) : {self.dt[0]}  - {self.dt[-1]}   
Min Value           : {min_y} MWT, index {ind_min}, date {self.dt[ind_min]}
Max_Value           : {max_y} MWT, index {ind_max}, date {self.dt[ind_max]}
Aver. Value         : {aver}  MWT
S.T.D.              : {std}
Sampling            : {self.sampling} sec, {self.sampling/60} minutes
Nayquist freq       : {self.f_Naik} Hz
Frequency sampling  : {self.delta_f} Hz

"""
        print(message)
        self.log.info(message)
        return

    def data_normalization(self):
        """  Two normalization types: statistical and [0.0 -1.0} """
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

        nblocks=round(self.n/self.segment_size)
        self.n_train_blocks=round(TRAIN_RATIO * nblocks)
        self.n_val_blocks = round(VAL_RATIO * nblocks)
        self.n_train=self.segment_size * self.n_train_blocks
        self.n_val = self.segment_size * self.n_val_blocks
        self.n_test = self.n - self.n_train -self.n_val
        return




    """ This method aims on the aggregation a day's observations in the segments (blocks). Each block has desired label 
    'day in week'."""
    def createSegmentLstPerDay(self):
        self.n4cnnTrain = self.n_train + self.n_val
        self.lstOffsetSegment = []
        n_seg = 0
        while (n_seg * self.segment_size <= self.n4cnnTrain):
            self.lstOffsetSegment.append(n_seg * self.segment_size)  # segments without overlap
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
                      scales=self.scales,
                      desire=self.df[TS_DEMAND_NAME].values[start_block])
                            )

        return

    """ This method aims for segments (blocks) averaging."""
    def averSegments(self):
        self.d_aver={}
        d_count={}
        for i in range(FIRST_DAY_IN_WEEK-1,LAST_DAY_IN_WEEK + 1):
            self.d_aver[i] = [0 for i in range(self.segment_size)]
            d_count[i] = 0


        wav =None
        for block in self.lstBlocks:
            i= block.desire
            if wav is None:
                wav=block.wav
            d_count[i] = d_count[i] +1
            for j in range(self.segment_size):
                self.d_aver[i][j] =self.d_aver[i][j] + block.x[j]
        for i in range(FIRST_DAY_IN_WEEK,LAST_DAY_IN_WEEK + 1):
            for j in range(self.segment_size):
                self.d_aver[i][j] = self.d_aver[i][j]/d_count[i]

        for day in range(FIRST_DAY_IN_WEEK,LAST_DAY_IN_WEEK + 1):
            self.scalogramImage(x=np.array(self.d_aver[day]), day=day, month = 0, wav=wav)


        return

    def averMonthSegments(self):
        self.d_aver_month={}
        d_count={}
        for month in range(FIRST_MONTH-1,LAST_MONTH + 1):
            self.d_aver_month[month] = {day:[0 for i in range(self.segment_size)] for day in range\
                (FIRST_DAY_IN_WEEK-1, LAST_DAY_IN_WEEK + 1)}
            d_count[month] = [0 for day in range(FIRST_DAY_IN_WEEK-1,LAST_DAY_IN_WEEK + 1)]

        wav = None
        for block in self.lstBlocks:
            try:
                month = self.df["Month_"].values[block.index]
                day= block.desire
                d_count[month][day] = d_count[month][day] +1
                if wav is None:
                    wav = block.wav
            except:
                pass
            for j in range(self.segment_size):
                self.d_aver_month[month][day][j] =self.d_aver_month[month][day][j] + block.x[j]
        for month in range(FIRST_MONTH,LAST_MONTH + 1):
            for day in range(FIRST_DAY_IN_WEEK, LAST_DAY_IN_WEEK + 1):
                for j in range(self.segment_size):
                    self.d_aver_month[month][day][j] = self.d_aver_month[month][day][j]/d_count[month][day]
        for month in range(FIRST_MONTH, LAST_MONTH + 1):
            for day in range(FIRST_DAY_IN_WEEK,LAST_DAY_IN_WEEK + 1):
                self.scalogramImage(x=np.array(self.d_aver_month[month][day]), day=day, month = month, wav=wav)

        return

    def scalogramImage(self,x:np.array=None, scales:list = [i + 1 for i in range(32)], day:int = 0, month:int=0,wav:object = None):

        if wav is None:
            return
        if day == 0:
            return
        if month == 0:
            title="Average for {}day ".format(day)
        else :
            title="Average for {} month {} day".format(month,day)
        scalogram, freqs = pywt.cwt(x, scales, wav)

        file_png = str(Path(Path(self.chart_log) / Path(title)).with_suffix(".png"))
        simpleScalingImgShow(scalogram=scalogram, index=0, title=title, file_png=file_png)
        return


    """ THis method aims on the aggregation observations. Common algorithm description is below:
    TBD    moved to predictorPath
    """
    def createSegmentLst(self):

        """ Create list of offset for all segments of self.segment_size along train part of TS.
        The segments may overlap.
        """

        self.n4cnnTrain = self.n_train + self.n_val
        self.lstOffsetSegment = []
        n_seg=0
        while (n_seg * self.segment_size <=self.n4cnnTrain):
            self.lstOffsetSegment.append(n_seg * self.segment_size)  # segments without overlap

            if self.overlap >0:       # segments with overlap
                n_overlap=1
                # 1: check overlap into segment bounds  2: end of overlapped segment into train TS bound
                while ( n_overlap * self.overlap < self.segment_size ) and \
                      ( n_seg * self.segment_size + n_overlap * self.overlap + self.segment_size < self.n4cnnTrain):
                    self.lstOffsetSegment.append(n_seg * self.segment_size + n_overlap * self.overlap)
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
                Block(x = self.y[start_block:start_block + self.segment_size],
                      sampling = self.sampling,
                      timestamp = self.dt[start_block],
                      index = start_block,
                      isTrain = True,
                      wav = self.wav,
                      scales = self.scales)
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

    def ImbalanceStateExtraction(self):
        pass

    def ExtStatesExtraction(self):

        # if len(self.lstOffsetSegment )==0:
        #     self.StatesExtraction()
        #     return

        after_pca_centers = None
        Xpca = None
        X=np.zeros(shape=(len(self.lstOffsetSegment),self.segment_size))
        (n,m) =X.shape
        for i in range(n):
            for j in range(m):
                X[i,j] =self.y[self.lstOffsetSegment[i]+j]

        # transformation by PCA to compress data
        if self.compress == "pca":
            self.log.info(
            "P(rincipial) C(omponent) A(nalysis) method is used for compress to data till {} components\n".\
                format(self.n_components))

            pca = PCA(n_components=self.n_components)
            obj = pca.fit(X)
            self.log.info("PCA object for transformation\n{}\n".format(obj))

            Xpca=pca.fit_transform(X)
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
        if self.compress == "pca":
            pca_centers = kmeans.cluster_centers_
            after_pca_centers = self.getCentersAfterPCA(pca_centers =  pca_centers)
            self.aux_log(X=X, Xpca=Xpca, after_pca_centers=after_pca_centers)

        for i in range(self.num_classes):

            if self.compress == "pca":
                bclk_class_center = Block(x=after_pca_centers[i, :],
                             sampling=self.sampling,
                             timestamp="N/A",
                             index=i,
                             isTrain=True,
                             wav=self.wav,
                             scales=self.scales,
                             desire=i)

            else:
               bclk_class_center =Block(x=kmeans.cluster_centers_[i,:],
                     sampling=self.sampling,
                     timestamp="N/A",
                     index=i,
                     isTrain=True,
                     wav=self.wav,
                     scales=self.scales,
                     desire=i)

            bclk_class_center.scalogramEstimation()
            title = "Scalogram  Center Class_{}".format(i)
            file_png = str( Path( Path(self.chart_log)/Path(title)).with_suffix(".png"))
            simpleScalingImgShow(scalogram = bclk_class_center.scalogram, index=i, title = title, file_png=file_png)
            self.blck_class_centers.append(bclk_class_center)  # save in the list of objects for the class centers

        self.chartClassCenters()
        self.logClassCenters()
        return

    def chartClassCenters(self):

        fl_chart=Path(PATH_SHRT_CHARTS/"States_examples").with_suffix(".png")
        legend_color ={"state 0":'b',     "state 1":'g',     "state 2":'r',     "state 3":'k',    "state 4":'o',
                       "state 5": 'b-',   "state 6": 'g-',   "state 7": 'r-',   "state 8": 'k-',  "state 9": 'o-',
                       "state 10": 'b--', "state 11": 'g--', "state 12": 'r--', "state 13": 'k-', "state 14": 'o--'
                       }
        legend =tuple([name for name,clr  in list(legend_color.items())[:self.num_classes]])
        x_axis = [i for i in range(self.segment_size)]
        plt.figure()
        i=0
        for key in (list(legend_color.values())[:self.num_classes]):
            plt.plot(x_axis, self.blck_class_centers[i].x, key)
            i = i +1

        plt.legend(legend, loc='best')
        plt.grid(True)
        plt.savefig(fl_chart)
        plt.close("all")

    def logClassCenters(self):
        """ Log"""
        msg = ""
        for item in self.blck_class_centers:
            msg_class = "{:<3d} ".format(item.desire)
            (m,)=item.x.shape
            for j in range(m):
                if j>0 and (j%8 == 0):
                    msg_class = msg_class + "\n     "
                msg_class = msg_class + "{:<10.4f}".format(item.x[j])
            msg = msg + "{}\n".format(msg_class)
        fl_out =Path(TRAIN_FOLDER / Path("State_typical_vector")).with_suffix(".log")
        with open(fl_out,'w') as fout:
            fout.write(msg)
        self.log.info("Typical values for states logged into {}".format(fl_out))
        return

    def getCentersAfterPCA(self, pca_centers:np.array = None)->np.array:

        after_pca_centers=np.zeros(shape=(self.num_classes, self.segment_size), dtype=float)
        number_in_cluster=np.zeros(shape=(self.num_classes), dtype=int)
        for item in self.lstBlocks:
            for i in range(self.segment_size):
                number_in_cluster[item.desire]=number_in_cluster[item.desire] + 1
                after_pca_centers[item.desire,i] = after_pca_centers[item.desire,i] + item.x[i]
        (cl,n) = after_pca_centers.shape
        for cluster in range(cl):
            for i in range(n):
                after_pca_centers[cluster,i] = after_pca_centers[cluster,i]/float(number_in_cluster[cluster])
        return after_pca_centers

    def aux_log(self, X: np.array =None, Xpca:np.array = None, after_pca_centers:np.array = None):
        aux_log_file = Path(AUX_TRAIN_FOLDER / Path("aux_pca_transform")).with_suffix(".log")
        self.log.info(" PCA transfor aux.data are logging in {}\n".format(str(aux_log_file)))

        msg_pca_centers = matrix2string(after_pca_centers)
        msg_X = matrix2string(X=X)
        msg_Xpca = matrix2string(X=Xpca)



        message=f"""
        Center clusters after pca-transformation
{msg_pca_centers} 

       
        Matrix Observations being was transformed
{msg_X}


        Matrix pca-transformed oservations
{msg_Xpca}


"""
        with open(aux_log_file,'w') as fot:
            fot.write(message)
        return

    def Data4CNN(self)->(np.ndarray,np.ndarray):
        n=len(self.lstBlocks)
        (m1, m2) = self.lstBlocks[0].scalogram.shape
        X=np.zeros(shape=(n,len(self.scales),self.segment_size)).astype( dtype=np.float32)
        Y=np.zeros(shape=(n),dtype=np.int32)
        for k in range(n):
            Y[k]=self.lstBlocks[k].desire
            (m1,m2)=self.lstBlocks[k].scalogram.shape
            for i in range (m1):
                for j in range(m2):
                    X[k,i,j]=self.lstBlocks[k].scalogram[i,j]

        return X,Y

    def initHMM_logClasses(self):
        """ Prepare segment-class and class-segment logs. Init HMM parameters (initial , transition matrix, emission).
        """

        self.log.info("segment - class log printing...")
        print("segment - class log printing...")
        outFile = Path(self.log_folder /"segment_class").with_suffix(".txt")
        lstStates =[]

        with open(outFile, 'w') as outf:
            header="\n#### Segment  Class Timestamp\nStart\nOffst\n==================================================\n"
            outf.write(header)
            i=0
            for item in self.lstBlocks:
                msg ="{:<4d} {:<5d}     {:<5d}  {:<30s}\n".format(i, item.index, item.desire, item.timestamp)
                outf.write(msg)
                if item.index % self.segment_size ==0:
                    lstStates.append(item.desire)
                    self.hmm.d_states[self.dt[item.index]] = item.desire
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
        self.hmm.state_sequence = np.array(lstStates)
        self.hmm.states=np.array([i for i in range(self.num_classes)])

        self.hmm.setModel()
        return

    def scalogramEstimation(self):
        pass
        for item in self.lstBlocks:
            item.scalogramEstimation()

    def createExtendedDataset(self):
        """ Create """
        df1 = pd.read_csv(self.pathToCsv)
        state_seq = []
        for item in self.hmm.state_sequence:
            state_seq.extend([item for k in range(self.segment_size)])
        n_tail=len(df1)- len(state_seq)
        if n_tail >0:
            state_seq.extend([-1 for k in range(n_tail)])
        df1["{}".format(self.num_classes)]=state_seq
        ext_DS=self.pathToCsv.stem
        path_ext_DS= Path( PATH_LOG_FOLDER / Path("{}_{}_states".format(ext_DS, self.num_classes))).with_suffix(".csv")
        df1.to_csv(path_ext_DS, index=False)

def plotClusters(kmeans: KMeans, X: np.array, file_png:Path):
    """
    The plot shows 2 first component of X
    :param kmeans: -sclearn.cluster.Kmeans object
    :param X: matrix n_samples * n_features or principal component n_samples * n_components.
    :param file_png:
    :return:
    """
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50)
    plt.savefig(file_png)
    plt.close("all")
    return

1
# ----------------------------------------------------------------------------------------------------------

class DatasetImbalance(Dau):
    """ Class for csv -file reading and processing.
    Class members:
    pathToCsv - path to file with historical observations.
    df - pandas' DataFrame.
    y - time series (TS), historical observations.
    x1- time series (TS), historical observations.
    x2 - time series (TS), historical observations.
    dt - time labels (timestamps) for observation.
    n - size of TS.
    mean, std,min, max -simple stat.parameters of TS.
    n_train, n_val, n_test -split TS of size n on train sequence of size n_train, validation and test sequences.
    n_train_blocks, n_val_blocks - TS splits on segments (blocks) are using for mid term forecasting.
    lstBlocks - the list which contains block objects (implementation of Block class.
    lstOffsetSegment -the which contains the offset of segments in  TS for each block.
    hmm - hidden markov model object

    Class methods:
    __init__  - constructor
    __str__
    readDataset
    data_normalization
    data_inv_normalization
    setTrainValTest
    createSegmentLst
    ExtStatesExtraction
    Data4CNN
    initHMM_logClasses
    scalogramEstimation
    """

    def __init__(self, pathTo: str = "", ts: str = "", ts_x1: str = "", ts_x2: str = "", dt: str = "Date Time",
                 sampling: int = 10 * 60, n_steps: int = 144, segment_size: int = 96, norm: str = "stat",
                 overlap: int = 0, continuous_wavelet: str = 'mexh', num_classes: int = 4, num_scales: int = 16,
                 compress: str = 'pca', n_components: int = 2, model_repository: Path = PATH_REPOSITORY,
                 log_folder: Path = PATH_LOG_FOLDER, chart_log: Path = PATH_CHART_LOG, wavelet_image = PATH_WV_IMAGES ):
        """ Constructor """

        super().__init__(ts=ts, dt=dt, sampling=sampling, n_steps=n_steps, segment_size=segment_size,
                         norm=norm, overlap=overlap, continuous_wavelet=continuous_wavelet,
                         num_classes=num_classes, num_scales=num_scales, compress=compress,
                         n_components=n_components, model_repository=model_repository, log_folder=log_folder,
                         chart_log=chart_log, wavelet_image = wavelet_image)
        """ Constructor"""

        self.pathToCsv = pathTo
        self.df: pd.DataFrame = None
        self.y = None
        self.x1 = None
        self.x2 = None
        self.x1_name = ts_x1
        self.x2_name = ts_x2
        self.dt = None
        self.n = 0
        self.mean = 0.0
        self.std = 1.0
        self.min = 0.0
        self.max = 1.0
        self.n_train = 0
        self.n_val = 0
        self.n_test = 0
        self.n_train_blocks = 0
        self.n_val_blocks = 0
        self.lstBlocks = []
        self.lstOffsetSegment = []
        self.hmm = hmm()
        self.states = []
        self.ext_states = []


    def __str__(self):
        msg = f"""

Dataset              : {self.pathToCsv}
TS name              : {self.ts_name}  
TS (X1) name         : {self.x1_name}  
TS (X2) name         : {self.x2_name}
Timestamp labels     : {self.dt_name}  
Data Normalization   : {self.norm}
TS mean              : {self.mean}     TS std : {self.std} TS length : {self.n} 
Sampling             : {self.sampling} sec 
Segment Size         : {self.segment_size}  Train blocks : {self.n_train_blocks} Validation blocks : {self.n_val_blocks}
Train Size           : {self.n_train}  Validation Size : {self.n_val}  Test Size: {self.n_test} 

Wavelet              : {self.wav}
Scales               : {self.scales}
Frequencies,Hz       : {self.frequencies}
Wavelet wigth        : {self.width} Max len :{self.max_wav_len}

Classification
Data Compress Method : {self.compress}
Number components    : {self.n_components}

Model Repository     : {self.model_repository}
Aux Log Folder       : {self.log_folder}
Charts               : {self.chart_log}

"""
        self.log.info(msg)
        print(msg)
        return msg

    def readDataset(self):
        self.df = pd.read_csv(self.pathToCsv)
        self.n = len(self.df)
        self.y = self.df[self.ts_name].values
        self.x1 = self.df[self.x1_name].values
        self.x2 = self.df[self.x1_name].values
        self.dt = self.df[self.dt_name].values

        self.frequencies = pywt.scale2frequency(self.wavelet, self.scales) / self.sampling
        self.max_wav_len = (0 if not self.scales else int(np.max(self.scales) * self.width))

    def StatesExtraction(self):
        self.states =[]
        for i in range(0,len(self.y)):
            if abs(self.y[i])<=STATE_0_MARGE :
                self.states.append(STATE_0)
            elif  self.y[i]*(-1)>STATE_0_MARGE :
                self.states.append(STATE_DEMAND)
            elif self.y[i]  > STATE_0_MARGE:
                self.states.append(STATE_GENERATION)

    def createExtendedDataset(self):
        """ Create """
        df1 = pd.read_csv(self.pathToCsv)

        df1["states"]=self.states
        ext_DS=self.pathToCsv.stem
        path_ext_DS= Path( PATH_LOG_FOLDER / Path("{}_{}_states".format(ext_DS, self.num_classes))).with_suffix(".csv")
        df1.to_csv(path_ext_DS, index=False)
if __name__ == "__main__":
    pass
