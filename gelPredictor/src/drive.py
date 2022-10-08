#!/usr/bin/env python3

""" Contains a functions for medium - and short - term forecast gorizont. """

import logging
import pandas as pd

from sys_util.parseConfig import SHRT_ANN_MODEL_TYPES, SHRT_ANN_MODEL_DICT, N_STEPS, SHRT_EPOCHS, SHRT_HIDDEN_LAYERS, \
    SHRT_DROPOUT, SHRT_FEATURES, SHRT_UNITS, TS_NAME, TS_TIMESTAMP_LABEL, PATH_REPOSITORY, LOG_FOLDER_NAME
from sys_util.utils import  exec_time
from src.shrttermModels import MLP, CNN, LSTM
from src.shtrm import ShortTerm

logger=logging.getLogger(__name__)

def drive_all_classes(shrt_data:ShortTerm = None):
    """ Train ANN models for all classes"""

    if shrt_data is None:
        logger.log.error("No object for short term data . Exit!")
        return
    if shrt_data.d_df is None or shrt_data.d_size is None:
        logger.log.error("Incorrect short term data. Exit!")
        return

    for class_label,class_df in shrt_data.d_df.items():
        df=pd.read_csv(class_df)
        x=df[TS_NAME].values
        dt =df[TS_TIMESTAMP_LABEL].values
        logger.info("Train ANN for {} class (state)".format(class_label))
        nret = drive_train(shrt_data=shrt_data, class_label=class_label)
        if (nret == 0):
            logger.info("ANN models for {} class(state) trained successfully".format(class_label))
        else:
            logger.error("ANN models for {} class(state) training failed".format(class_label))

    return

def drive_train(shrt_data:ShortTerm = None, class_label:int=0)->int:
    """ Train ANN models"""

    d_models = {}
    for keyType in SHRT_ANN_MODEL_TYPES:
        if gatherModels(d_models=d_models, keyType = keyType) > 0 :
            logger.error('ANN model of {} -type: error'.format(keyType))

    if len(d_models)==0:
        logger.error(" No created ANN models!")
        return 1
    X,y,X_val,y_val =shrt_data.createTrainData(class_label=class_label)
    trainData = (X,y,X_val,y_val)
    histories = fit_models(d_models=d_models, trainData=trainData)

    msg = f"""
    
Train Histories for {class_label} class(state)

{histories}
    
    """
    logger.info(msg)
    return 0

def createTrainData()->tuple:
    """ Create traindata (learning data X and desired vector y) from 'syntetical' time series"""

    pass

def gatherModels(d_models:dict={}, keyType:str='MLP')->int:
    """

    :param d_models: [in], [out] =dictionary {index:<wrapper for model>}. Through parameter
    <wrapper for model>.model, the access to tensorflow-based model is given.
    :param keyType: [in] string value. type of NN model likes as 'MLP', 'CNN','LSTM'.
    :return: 0-ok, 1 -error
    """

    if keyType not in SHRT_ANN_MODEL_TYPES:
        msg  = "Undefined type of ANN Model\n It is not supported by gelPredictor!".format(keyType)
        print(msg)
        logger.error(msg)
        return 1

    for keyType,listType in SHRT_ANN_MODEL_DICT.items():
        for tuple_item in listType:
            (index_model,name_model) = tuple_item
            if keyType == "MLP":
                curr_model=MLP(name_model, keyType, N_STEPS, SHRT_EPOCHS, None)
                curr_model.param = (N_STEPS, SHRT_FEATURES, SHRT_HIDDEN_LAYERS, SHRT_DROPOUT)
            elif keyType == "LSTM":
                curr_model = LSTM(name_model, keyType, N_STEPS, SHRT_EPOCHS, None)
                curr_model.param = (SHRT_UNITS, N_STEPS, SHRT_FEATURES)
            elif keyType == "CNN":
                curr_model = CNN(name_model, keyType, N_STEPS, SHRT_EPOCHS, None)
                curr_model.param = ( N_STEPS, SHRT_FEATURES)
            else:
                msg = "Type model error"
                print(msg)
                logger.error(msg)
                return 1

            curr_model.timeseries_name = TS_NAME
            curr_model.path2modelrepository = PATH_REPOSITORY

            funcname = getattr(curr_model, name_model)
            curr_model.set_model_from_template(funcname)

            d_models[index_model] = curr_model
            logger.info(curr_model)

        return 0

@exec_time
def fit_models(d_models:dict = {}, trainData:tuple =())->dict:
    r""" A method fits NN models and STS models.
    :param d_models:
    :param cp:
    :param ds:
    :return:
    """

    (X,y,X_val,y_val) = trainData
    histories = {}
    for k, v in d_models.items():
        curr_model = v


        # #LSTM
        if curr_model.typeM == "CNN" or curr_model.typeM == "LSTM":
            X = X.reshape((X.shape[0], X.shape[1], SHRT_FEATURES))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], SHRT_FEATURES))

        curr_model.param_fit = (
            X, y, X_val, y_val, N_STEPS, SHRT_FEATURES, SHRT_EPOCHS, LOG_FOLDER_NAME, None)
        logger.info("\n\n {} model  fitting\n".format(curr_model.nameM))
        history = curr_model.fit_model()

        # if curr_model.typeModel == "CNN" or curr_model.typeModel == "LSTM" or curr_model.typeModel == "MLP":
        #
        #     chart_MAE(curr_model.nameModel, cp.rcpower_dset, history, cp.n_steps, cp.folder_train_log,
        #               cp.stop_on_chart_show)
        #     chart_MSE(curr_model.nameModel, cp.rcpower_dset, history, cp.n_steps, cp.folder_train_log,
        #               cp.stop_on_chart_show)
        # elif curr_model.typeModel == "tsARIMA":
        #     curr_model.fitted_model_logging()

        histories[k] = history

    return histories

