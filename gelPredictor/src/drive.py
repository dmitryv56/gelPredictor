#!/usr/bin/env python3

""" Contains a functions for medium - and short - term forecast gorizont. """

import logging
import pandas as pd
from pathlib import Path

from sys_util.parseConfig import SHRT_ANN_MODEL_TYPES, SHRT_ANN_MODEL_DICT, N_STEPS, SHRT_EPOCHS, SHRT_HIDDEN_LAYERS, \
    SHRT_DROPOUT, SHRT_FEATURES, SHRT_UNITS, TS_NAME, TS_TIMESTAMP_LABEL, PATH_REPOSITORY, LOG_FOLDER_NAME
from sys_util.utils import  exec_time
from src.vshrtrmModels import MLP, CNN, LSTM
from src.vshtrm import VeryShortTerm

logger=logging.getLogger(__name__)

TRAIN_FOLDER = Path(LOG_FOLDER_NAME / Path("TrainPath"))
TRAIN_FOLDER.mkdir(parents=True, exist_ok=True)

def drive_all_classes(shrt_data:VeryShortTerm = None):
    """ Train ANN models for all classes"""

    if shrt_data is None:
        logger.log.error("No object for Very Short-Term data . Exit!")
        return
    if shrt_data.d_df is None or shrt_data.d_size is None:
        logger.log.error("Incorrect Very Short-Term data. Exit!")
        return

    for class_label,class_df in shrt_data.d_df.items():
        df=pd.read_csv(class_df)
        x=df[TS_NAME].values
        dt =df[TS_TIMESTAMP_LABEL].values
        logger.info("\n\n          Train ANN for {} class (state)\n\n".format(class_label))
        nret = drive_train(shrt_data=shrt_data, class_label=class_label)
        if (nret == 0):
            logger.info("\n        ANN models for {} class(state) trained successfully\n\n".format(class_label))
        else:
            logger.error("\n       ANN models for {} class(state) training failed\n\n".format(class_label))

    return

def drive_train(shrt_data:VeryShortTerm = None, class_label:int=0)->int:
    """ Train ANN models"""

    d_models = {}

    if gatherModels(d_models=d_models,  class_label = class_label) > 0 :
        logger.error('ANN model gathering error')

    if len(d_models)==0:
        logger.error(" No created ANN models!")
        return 1
    X,y,X_val,y_val =shrt_data.createTrainData(class_label=class_label)
    trainData = (X,y,X_val,y_val)
    histories = fit_models(d_models=d_models, trainData=trainData, class_label = class_label)

    history_msg = ""
    for key, value in histories.items():
        history_msg = history_msg + "{} : \n    {}\n".format(key,value)

    msg = f"""
    
Train Histories for {class_label} class(state)

{history_msg}
    
    """
    logger.info(msg)
    return 0


def gatherModels(d_models:dict={}, keyType:str='MLP', class_label:int =0 )->int:
    """

    :param d_models: [in], [out] =dictionary {index:<wrapper for model>}. Through parameter
    <wrapper for model>.model, the access to tensorflow-based model is given.
    :param keyType: [in] string value. type of NN model likes as 'MLP', 'CNN','LSTM'.
    :return: 0-ok, 1 -error
    """

    # if keyType not in SHRT_ANN_MODEL_TYPES:
    #     msg  = "Undefined type of ANN Model\n It is not supported by gelPredictor!".format(keyType)
    #     print(msg)
    #     logger.error(msg)
    #     return 1

    logger.info("\n          ANN models assembling from templates for {} class(state)\n".format(class_label))
    for keyType,listType in SHRT_ANN_MODEL_DICT.items():
        msg = f"""
        
ANN type : {keyType}
Models   :
{listType}

        """
        logger.info(msg)

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

            file_name = "TS_{}_{}_{}".format(curr_model.timeseries_name, curr_model.typeM, curr_model.nameM)
            model_log = Path(TRAIN_FOLDER / Path(file_name)).with_suffix(".log")
            logger.info("\n\n {} model  is logging in {}\n".format(curr_model.nameM, str(model_log)))

            funcname = getattr(curr_model, name_model)
            curr_model.set_model_from_template(funcname, model_log = model_log)

            d_models[index_model] = curr_model
            logger.info(curr_model)
        pass
    pass

    return 0

@exec_time
def fit_models(d_models:dict = {}, trainData:tuple =(), class_label:int = 0)->dict:
    """
    Fit Models for train data
    :param [in] d_models
    :param [in] trainData -the tuple contains (X,y,X_val,y_val)
    :param [in] class_label
    :return: keras' histories dict
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
        file_name = "ANN_{}_{}".format( curr_model.typeM, curr_model.nameM)
        train_class_folder = Path(TRAIN_FOLDER /Path("state_{}".format(class_label)))
        train_class_folder.mkdir( parents = True, exist_ok = True)
        fit_log = Path(train_class_folder /Path(file_name) ).with_suffix(".log")
        logger.info("\n\n {} model  fitting is logging in {}\n".format(curr_model.nameM, str(fit_log)))
        history = curr_model.fit_model(fit_log = fit_log)

        histories[k] = history

    return histories

