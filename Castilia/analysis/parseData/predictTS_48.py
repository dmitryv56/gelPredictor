#!/usr/bin/env python3

import sys
from pathlib import Path

import keras.models
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from datetime import datetime
from sklearn.decomposition import PCA
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime,timedelta

from corrAnalysis import walkOverDSets, HEADER_NAME, ACTUAL_HEADER_NAME, DST_DIR
from tsAnalysis import autocorr, psd, DATASET_PATH, TRAIN_METEO_PATH, TEST_METEO_PATH, STAT_DIR, PATH_MAIN_LOG, \
    REPO_DIR, AUX_METEO_DIR ,  AUX_PCA_DIR,  METEO_EXOGEN_PATH, OUT_DIR, TITLE, TS_NAME, dt_name, START_YEAR , \
    START_MONTH , START_DAY ,    FINISH_YEAR , FINISH_MONTH ,FINISH_DAY, TS_NAME, DT_NAME,\
    TEST_DATA_PATH_TEMPLATE

from predictExogen import modelLSTM,fitModel, chartHistory,ANNmodel
# from api_aux import univar_cnn, cnn_lstm, bidir_lstm, stacked_lstm, vanilla_lstm, mlp_2, mlp_1,
import api_aux
from api_aux import     chart_predict ,  chart_2series, exec_time


# NN_MODEL_LIST =[ 'univar_cnn', 'mlp_2', 'mlp_1', 'vanilla_LSTM_50', 'vanilla_lstm', 'cnn_lstm','bidir_lstm', 'stacked_lstm']
NN_MODEL_LIST =[ 'univar_cnn', 'mlp_2', 'mlp_1', 'vanilla_LSTM_50', 'vanilla_lstm', 'cnn_lstm', 'stacked_lstm']
#NN_MODEL_LIST =[ 'univar_cnn', 'mlp_2']

VALID="valid"
NOT_VALID="not enough data"
NOT_VALID_FEATURE="not valid feature data "
ANY_EXCEPTION="exception at data processing"
TIMESTAMP ="Timestamp"
FORECAST  ="Forecast"
size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(4*1024*1024),
                                 backupCount=int(2))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)

# OUT_DIR = "/home/dmitry/LaLaguna/Castilia/out_more_48_"

@exec_time
def createLearnedDS(ts:np.array = None, X_exogen: np.array =None, n_steps: int=48,  discretization_rate: int=24):
    X, y = list(), list()
    (n,)=ts.shape
    (n_exogen, m_exogen) = X_exogen.shape

    i_exogen =0
    for i in range(n - n_steps):
        if i>0 and (i %discretization_rate == 0):
            i_exogen=i_exogen+1
        #     index ignore in X-exogen (first column) ...in range(1, m_exogen)
        tmp =[ts[k] for k in range(i, i+n_steps)] + [X_exogen[i_exogen,j] for j in range(1, m_exogen)]
        X.append(tmp)
    y=[ts[i+n_steps] for i in range(n-n_steps)]

    return np.array(X), np.array(y)

@exec_time
def logTrainDataset(X:np.array=None, y:np.array=None,exogen_names:list =ACTUAL_HEADER_NAME, ts_name:str=TS_NAME, \
                    n_steps:int=36, title:str ="training_input",dt:np.array = None):

    columns =["{}-{}".format(ts_name,i) for i in range(n_steps)] + exogen_names
    df = pd.DataFrame(X, columns)
    df.to_csv(Path(Path(OUT_DIR)/Path(title)).with_suffix(".csv"))
    if dt is None:
        columns = ["{}".format(ts_name) ]
        df = pd.DataFrame(y, columns)

    else:
        data={}
        data["Date Time"]=dt[n_steps:]
        data[ts_name] = y
        df = pd.DataFrame(data)
    df.to_csv(Path(Path(OUT_DIR) / Path("desired")).with_suffix(".csv"))
    return

@exec_time
def driveData(func_NNmodel:object = None, n_steps:int=24, epochs:int=50, pred_horizon:int = 48,norm:str="no norm")->(str,dict):
    # n_steps =12
    discretization_rate=24
    # pred_horizon = 48
    # epochs=50
    df =pd.read_csv(DATASET_PATH)
    ts=df[TS_NAME].values
    (ts_size,) = ts.shape
    dt=df[DT_NAME].values
    tsmin=ts.min()
    tsmax=ts.max()
    logger.info("TS size: {} TS min: {} TS max: {} TS mean: {} TS std : {}".format(ts_size, tsmin,tsmax,ts.mean(),ts.std()))
    if norm=="norm":
        tsmin = ts.min()
        tsmax = ts.max()
        ts = np.array([(ts[i] - tsmin) / (tsmax - tsmin) for i in range(ts_size)])
    elif norm=="no norm":
        tsmin=0.0
        tsmax=1.0
        ts = np.array([(ts[i] - tsmin) / (tsmax - tsmin) for i in range(ts_size)])
    elif norm=="stat":
        ts = np.array([ (ts[i]-ts.mean())/(ts.std()) for i in range(ts_size)])
    else:
        pass
    # ts = np.array([ (ts[i]-tsmin)/(tsmax-tsmin) for i in range(ts_size)])
    df_exogen =pd.read_csv(METEO_EXOGEN_PATH)
    X_exogen =df_exogen.to_numpy()
    X, y = createLearnedDS(ts=ts, X_exogen=X_exogen, n_steps=n_steps, discretization_rate=discretization_rate)
    (n,m)=X.shape
    model =None

    model = func_NNmodel(n_steps = m)

    if model is None:
        return
    X1, history = fitModel(model, X=X, y=y,epochs=epochs,validation_split =0.8)
    chartHistory(history=history, model_name=model._name)
    # prepare data for forecast
    last_row=X[-1,:]
    print(last_row)
    last_desired=y[-1]
    print(last_desired)
    df_predicted_exogen=pd.read_csv(Path(Path(OUT_DIR)/Path("meteo_exogenious_predict_lag_16")).with_suffix(".csv"))
    X_exogen_predicted=df_predicted_exogen.to_numpy()
    print(X_exogen_predicted)

    pred_dict = predict(model,  last_row = last_row, last_desired=last_desired, last_timestamp = dt[-1], \
                        X_exogen_predicted = X_exogen_predicted, pred_horizon = pred_horizon, \
                        discretization_rate = discretization_rate, n_steps=n_steps, tsmin = tsmin, tsmax = tsmax, \
                        model_name = model._name, norm=norm)

    logForecast(data=pred_dict, dt=dt, ts_name=TS_NAME, pred_horizon=48, model_name=model._name)

    return model._name, pred_dict

def initLearningDS(n_steps:int=24, norm:str="no norm", discretization_rate:int=24)->( \
        np.array, np.array, list, list,  np.array, np.array, float, float, float,float):
    pass
    df = pd.read_csv(DATASET_PATH)
    ts = df[TS_NAME].values
    (ts_size,) = ts.shape
    dt = df[DT_NAME].values
    tsmin = ts.min()
    tsmax = ts.max()
    logger.info(
        "TS size: {} TS min: {} TS max: {} TS mean: {} TS std : {}".format(ts_size, tsmin, tsmax, ts.mean(), ts.std()))
    if norm == "norm":
        tsmin = ts.min()
        tsmax = ts.max()
        ts = np.array([(ts[i] - tsmin) / (tsmax - tsmin) for i in range(ts_size)])
    elif norm == "no norm":
        tsmin = 0.0
        tsmax = 1.0
        ts = np.array([(ts[i] - tsmin) / (tsmax - tsmin) for i in range(ts_size)])
    elif norm == "stat":
        ts = np.array([(ts[i] - ts.mean()) / (ts.std()) for i in range(ts_size)])
    else:
        pass
    # ts = np.array([ (ts[i]-tsmin)/(tsmax-tsmin) for i in range(ts_size)])
    df_exogen = pd.read_csv(TRAIN_METEO_PATH)
    X_train_exogen = df_exogen.to_numpy()
    X, y = createLearnedDS(ts=ts, X_exogen=X_train_exogen, n_steps=n_steps, discretization_rate=discretization_rate)
    (n, m) = X.shape

    df_test_exogen = pd.read_csv(TEST_METEO_PATH)
    X_test_exogen = df_test_exogen.to_numpy()
    print(X_test_exogen)


    return X,y,list(dt) ,list(ts), X_train_exogen, X_test_exogen, tsmin, tsmax,ts.mean(),ts.std()

""" ANN learning """
def learning_stage(func_NNmodel:object = None,X:np.array =None, y:np.array=None, epochs:int=50)->(keras.models.Model, str):

    (n, m) = X.shape
    model = func_NNmodel(n_steps=m)

    if model is None:
        return None,""
    X1, history = fitModel(model, X=X, y=y, epochs=epochs, validation_split=0.8)
    chartHistory(history=history, model_name=model._name)

    return model, model._name,

def predict_stage(model):
    pass

def model_fit():
    pass


@exec_time
def predict(model, last_row:np.array=None, last_desired:float = None, X_exogen_predicted:np.array = None, \
            last_timestamp:str = None, pred_horizon:int=48, discretization_rate: int =24, n_steps:int = 36, \
            tsmin:float = 0.0, tsmax:float = 1.0, tsmean:float=0.0, tsstd:float=1.0, model_name:str="", \
            norm:str="no norm")->dict:

    pred_normalized= []
    pred           = []
    dt_pred        = []
    (n_exogen,m_exogen) =X_exogen_predicted.shape
    i_exogen =0
    #  X_exogen predict contains the index column. It ignores.
    x_input =[last_row[i] for i in range(1,n_steps)] + \
             [last_desired]+ [X_exogen_predicted[i_exogen,j] for j in range(1,m_exogen)]
    datetime_obj = datetime.strptime(last_timestamp, "%Y-%m-%d %H")
    try:
        for pred_step in range(pred_horizon):
            if pred_step>0 and (pred_step % discretization_rate == 0):
                i_exogen = i_exogen + 1

            x=np.array(x_input)
            (n_input,)=x.shape
            xtensor = x.reshape((1, n_input, 1))
            ypred = model.predict(xtensor, verbose=0)
            tupShape=ypred.shape
            if len(tupShape)==1:
                y=ypred[0]
            elif len(tupShape) == 2:
                y=ypred[0][0]
            elif len(tupShape) == 3:
                y=ypred[0][0][0]
            else:
                msg = "Predicted value is a tensor of infinite order{} : pred_step  {}".format(tupShape, pred_step)
                logger.error(msg)
            pred_normalized.append(y)
            if norm=="no norm" or norm == "norm":
                pred.append(tsmin + y*(tsmax-tsmin))
            elif norm=="stat":
                pred.append(tsmean + y*tsstd)
            else:
                pred.append(y )
            i_exogen=i_exogen % n_exogen

            x_input = [x[i ] for i in range(1,n_steps )] \
                            + \
                      [y ] + [X_exogen_predicted[i_exogen, j] for j in range(1, m_exogen)]
            datetime_obj = datetime_obj + timedelta(hours=1)
            dt_pred.append( datetime_obj.strftime("%Y-%m-%d %H"))
    except:
        msg = "\nOoops! Unexpected error: {}\n{}\n".format(sys.exc_info()[0], sys.exc_info()[1])
        logger.error(msg)
        msg="Exception for :: pred_step : {}, i_exogen : {} timestamp : {}".format(pred_step, i_exogen, datetime_obj.strftime("%Y-%m-%d %H"))
        logger.error(msg)
    finally:
        pass
    data = {}
    data[TIMESTAMP]       = dt_pred
    data[FORECAST]        = pred
    data["Scaled Forecast"] = pred_normalized
    return data

@exec_time
def predict_step(model, last_row:np.array=None, last_desired:float = None, X_exogen_predicted:np.array = None, index_in_X_exogen: int =0,\
            last_timestamp:str = None, pred_horizon:int=48, discretization_rate: int =24, n_steps:int = 36, \
            tsmin:float = 0.0, tsmax:float = 1.0, tsmean:float=0.0, tsstd:float=1.0, model_name:str="", \
            norm:str="no norm")->(datetime, str, list):

    pred_normalized= []
    pred           = []
    dt_pred        = []
    (n_exogen,m_exogen) =X_exogen_predicted.shape
    i_exogen =index_in_X_exogen
    #  X_exogen predict contains the index column. It ignores.
    x_input =[last_row[i] for i in range(1,n_steps)] + \
             [last_desired]+ [X_exogen_predicted[i_exogen,j] for j in range(1,m_exogen)]
    datetime_obj = datetime.strptime(last_timestamp, "%Y-%m-%d %H")
    datetime_obj = datetime_obj + timedelta(hours=1)
    dt_next = datetime_obj.strftime("%Y-%m-%d %H")
    try:
        for pred_step in range(pred_horizon):
            if pred_step>0 and (pred_step % discretization_rate == 0):
                i_exogen = i_exogen + 1

            x=np.array(x_input)
            (n_input,)=x.shape
            xtensor = x.reshape((1, n_input, 1))
            ypred = model.predict(xtensor, verbose=0)
            tupShape=ypred.shape
            if len(tupShape)==1:
                y=ypred[0]
            elif len(tupShape) == 2:
                y=ypred[0][0]
            elif len(tupShape) == 3:
                y=ypred[0][0][0]
            else:
                msg = "Predicted value is a tensor of infinite order{} : pred_step  {}".format(tupShape, pred_step)
                logger.error(msg)
            pred_normalized.append(y)
            if norm=="no norm" or norm == "norm":
                pred.append(tsmin + y*(tsmax-tsmin))
            elif norm=="stat":
                pred.append(tsmean + y*tsstd)
            else:
                pred.append(y )
            i_exogen=i_exogen % n_exogen

            x_input = [x[i ] for i in range(1,n_steps )] \
                            + \
                      [y ] + [X_exogen_predicted[i_exogen, j] for j in range(1, m_exogen)]
            # datetime_obj = datetime_obj + timedelta(hours=1)
            dt_pred.append( datetime_obj.strftime("%Y-%m-%d %H"))
    except:
        msg = "\nOoops! Unexpected error: {}\n{}\n".format(sys.exc_info()[0], sys.exc_info()[1])
        logger.error(msg)
        msg="Exception for :: pred_step : {}, i_exogen : {} timestamp : {}".format(pred_step, i_exogen, datetime_obj.strftime("%Y-%m-%d %H"))
        logger.error(msg)
    finally:
        pass
    data = {}
    data[TIMESTAMP]       = dt_pred
    data[FORECAST]        = pred
    data["Scaled Forecast"] = pred_normalized
    return datetime_obj, dt_next, pred

@exec_time
def logForecast(data:dict=None, dt:np.array = None, ts_name:str=TS_NAME, pred_horizon:int=48, model_name:str ="", \
                n_steps:int=24, epochs:int=50, norm:str="no norm"):
    df=pd.DataFrame(data)
    df.to_csv(Path(Path(OUT_DIR) / Path("Forecast")).with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(20, 15.0))
    df['Forecast'].plot(ax=ax, label='Forecast')
    ax.legend()
    norm=norm.replace(' ','_')
    ax.set_title("{}: Forecast for {} hours horizon by {} \nLag is {}. Training epochs {} Normalization {}".format(\
        ts_name, pred_horizon, model_name, n_steps, epochs, norm))
    plt.savefig(Path(Path(OUT_DIR) / Path("{}_Forecast_by_{}_model_lag_{}_epochs_{}_".format(\
        ts_name, model_name, n_steps,epochs, norm))).with_suffix(".png"))
    plt.close("all")
    return


def main():
    n_steps =24
    n_epochs = 50
    pred_horizon = 48
    norm = "no norm"
    discretization_rate = 24



    X, y, dt, ts, X_train_exogen, X_test_exogen, tsmin, tsmax, tsmean, tsstd = \
        initLearningDS(n_steps = n_steps, norm = norm, discretization_rate = discretization_rate)
    model_ANN_list=[]
    for func_name in NN_MODEL_LIST:
        func_NNmodel = getattr(api_aux, func_name)
        # func_NNmodel(n_steps=32)
        model_name =""
        try:

            model , model._name =  learning_stage(func_NNmodel=func_NNmodel, X=X, y=y, epochs=n_epochs)
            # model_name, data = driveData(func_NNmodel=func_NNmodel, n_steps=n_steps, epochs=epochs, \
            #                              pred_horizon = pred_horizon, norm=norm)
            model_ANN_list.append(model)
        except :
            msg = "\nOoops! Unexpected error: {}\n{}\nModel name : {}".format(sys.exc_info()[0], sys.exc_info()[1],model_name)
            logger.error(msg)
        finally:
           pass
    pass

    """ predict path"""
    df_test_template = pd.read_csv(TEST_DATA_PATH_TEMPLATE)
    cols=list(df_test_template.columns)
    skip_cols=4
    for k in range(pred_horizon):
        df_test_template =df_test_template.astype({cols[k+skip_cols]:np.float64})


    dt_predict =df_test_template['DateTime'].values
    (n,m)=X.shape
    hourly_predict_size = len(dt_predict)
    last_row_ = X[-1,:]
    last_desired = y[-1]
    last_row=last_row_
    for pred_index in range(hourly_predict_size):


        aver_pred_list=[0.0 for i in range(pred_horizon)]
        for model in model_ANN_list:

            model_name = model._name
            try:
                objt, strt, pred_list =\
                predict_step(model,last_row=last_row, last_desired=last_desired, X_exogen_predicted=X_test_exogen,\
                             index_in_X_exogen=pred_index, last_timestamp=dt_predict[pred_index], \
                             pred_horizon=pred_horizon, discretization_rate=discretization_rate, n_steps=n_steps, \
                             model_name=model_name)
            except:
                msg = "\nOoops! Unexpected error: {}\n{}\nModel name : {}".format(sys.exc_info()[0], sys.exc_info()[1],\
                                                                                  model_name)
                logger.error(msg)
            finally:
                pass
            for k in range(pred_horizon):
                aver_pred_list[k]=aver_pred_list[k] + pred_list[k]

        # average predict for one hour
        for k in range(pred_horizon):
            aver_pred_list[k] = round(aver_pred_list[k]/len(model_ANN_list),3)
            df_test_template[cols[k+skip_cols]].values[pred_index]=aver_pred_list[k]
        if pred_index % 100 == 0 :
            df_test_template.to_csv(Path(OUT_DIR)/Path("aux_{}".format(pred_index)).with_suffix(".csv"))
    df_test_template.to_csv(Path(OUT_DIR)/Path("df_test_Wind_CASTILLA-LEON_only_Power_forecast").with_suffix(".csv"))


@exec_time
def main1():
    n_steps=24
    epochs =80

    pred_horizon = 72  #48
    norm="no norm"
    bundle_forecast_dict ={}
    for func_name in NN_MODEL_LIST:
        func_NNmodel = getattr(api_aux, func_name)
        # func_NNmodel(n_steps=32)
        model_name =""
        try:
            model_name, data = driveData(func_NNmodel=func_NNmodel, n_steps=n_steps, epochs=epochs, \
                                         pred_horizon = pred_horizon, norm=norm)
        except :
            msg = "\nOoops! Unexpected error: {}\n{}\nModel name : {}".format(sys.exc_info()[0], sys.exc_info()[1],model_name)
            logger.error(msg)
        finally:
           pass
        try:
            if not bundle_forecast_dict:
                bundle_forecast_dict[DT_NAME] = data[TIMESTAMP]
            bundle_forecast_dict[model_name] = data[FORECAST]
            logger.info("Model : {} forecast added {}". format(model_name, FORECAST))
        except :
            msg = "\nOoops! Unexpected error: {}\n{}\n".format(sys.exc_info()[0], sys.exc_info()[1])
            logger.error(msg)
        finally:
           pass
    n_predict=0
    mean_predict =[0.0 for i in range(pred_horizon) ]
    try:
        for k,values in bundle_forecast_dict.items():
            if k ==DT_NAME:
                continue
            mean_predict=[mean_predict[i] + values[i] for i in range(pred_horizon)]
            n_predict = n_predict +1
        mean_predict = [mean_predict[i]/n_predict for i in range(pred_horizon)]
        bundle_forecast_dict["average"] = np.array(mean_predict)
    except :
        msg = "\nOoops! Unexpected error: {}\n{}\n".format(sys.exc_info()[0], sys.exc_info()[1])
        logger.error(msg)
    finally:
        pass

    df_bundle =pd.DataFrame(bundle_forecast_dict)
    df_bundle.to_csv(Path(Path(OUT_DIR)/Path("bundle_predicts_epochs_{}_lags_{}".format( \
        epochs, n_steps))).with_suffix(".csv"))
    try:
        df_bundle.plot()
        # plt.show()
        plt.savefig(Path(Path(OUT_DIR)/Path("bundle_predicts_epochs_{}_lags_{}".format(\
            epochs, n_steps))).with_suffix(".png"))
    except:
        msg = "\nOoops! Unexpected error: {}\n{}\n".format(sys.exc_info()[0], sys.exc_info()[1])
        logger.error(msg)
    finally:
        plt.close("all")
    df = pd.read_csv(DATASET_PATH)
    # chart_predict(dict_predict= bundle_forecast_dict, n_predict=48, df=df, title=TS_NAME, Y_label=TS_NAME, \
    #               discret_in_hours=1, log_folder=OUT_DIR)
if __name__ == "__main__":
    # driveData(mode=1)
    logger.info("Start...\n\n\n")
    main()
    logger.info("Finish ....")
    pass
    # discretization_rate=6
    # n_steps = 8
    # n=128
    # m_exogen=4
    # ts=np.arange(n)
    # print(ts)
    # X_exogen=np.arange(int(n/discretization_rate)* m_exogen)
    # print(X_exogen)
    # X_exogen=X_exogen.reshape((int(n/discretization_rate), m_exogen))
    # print(X_exogen)
    # X,y = createLearnedDS(ts=ts, X_exogen=X_exogen, n_steps= n_steps, discretization_rate = discretization_rate)
    # print(X)
    # print(y)
    pass
