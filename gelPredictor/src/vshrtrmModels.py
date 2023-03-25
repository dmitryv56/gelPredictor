#!/usr/bin/env python3


#!/usr/bin/python3
""" This module contains the NeuronNet Model implementations."""

import copy
import logging
from pathlib import Path

import tensorflow as tf
from keras import layers
from keras import metrics, models
from keras.layers import Bidirectional, TimeDistributed, Flatten
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import MaxPooling1D
from keras.models import Sequential

from sys_util.utils import exec_time

logger=logging.getLogger(__name__)

""" NNmodel class is a base class for MLP, LSTM and CNN classes"""

class baseNNmodel(object):
    """ Base class for Artifical Neural Net (ANN) models"""

    def __init__(self, nameM:str="MLP", typeM:str="MLP", n_steps:int=32,n_epochs:int = 10,f:object =None):
        """Constructor """

        self.nameM=nameM
        self.typeM=typeM
        self.n_steps =n_steps
        self.n_epochs=n_epochs
        self.log = logger
        self.f = f

class NNmodel(baseNNmodel):
    _param = ()
    _param_fit = ()
    model = None

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        pass
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    # getter/setter
    def set_param(self, val):
        type(self)._param = copy.deepcopy(val)

    def get_param(self):
        return type(self)._param

    param = property(get_param, set_param)

    def set_param_fit(self, val):
        type(self)._param_fit = copy.copy(val)

    def get_param_fit(self):
        return type(self)._param_fit

    param_fit = property(get_param_fit, set_param_fit)

    # def set_model_from_template(self,func, *args):
    #     self.model = func(*args)
    def set_model_from_template(self, func, model_log: Path = None):
        self.model = func()
        self.model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        self.model.summary()
        msg = "{} -model has been set from template. ".format(self.model.name)
        self.log.info(msg)
        if model_log is None:
            self.model.summary(print_fn=lambda x: self.log.info(x + '\n'))
        else:
            with open(model_log, 'w') as fout:
                fout.write(msg + '\n')
                self.model.summary(print_fn=lambda x: fout.write(x + '\n'))
        return

    @exec_time
    def set_model_from_saved(self, path_2_saved_model):

        old_model = models.load_model(path_2_saved_model)
        old_model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        old_model.summary()
        if self.f is not None:
            old_model.summary(print_fn=lambda x: self.f.write(x + '\n'))
        self.model = old_model
        return old_model

    @exec_time
    def updOneLeInSavedModel(self, old_model, layer_number, key, value):
        pass

        model_config = old_model.get_config()
        if self.f is not None:
            self.f.write('\n The model configuration\n')
            self.f.write(model_config)
        for lr in range(len(model_config['layers'])):
            print(model_config['layers'][lr])
            print(model_config['layers'][lr]['config'])
            if self.f is not None:
                self.f.write(model_config['layers'][lr])
                self.f.write(model_config['layers'][lr]['config'])

        # model_config['layers'][0]['config']['batch_input_shape'] = (None, 36, 1)
        model_config['layers'][layer_number]['config'][key] = value
        print(model_config['layers'][layer_number])
        print(model_config['layers'][layer_number]['config'])

        if self.f is not None:
            self.f.write('\nUpdated layer\n')
            self.f.write(model_config['layers'][layer_number])
            self.f.write(model_config['layers'][layer_number]['config'])

        self.model = models.Sequential.from_config(model_config)
        self.model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        self.model.summary()

        if self.f is not None:
            self.f.write('\n New model \n')
            self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

        return

    """
    updates key:value in the layer 'config' dictionary
    """

    @exec_time
    def updConfigSavedModel(self, old_model,
                            list_updates):  # [(layer_number, key, value),(layer_number, key, value)...]
        """

        :param old_model: tensorflow.keras Sequential model that was loaded from saved model
        :param list_updates: list contains a tuples (layer_number, key, new_value)
        :return:
        """

        model_config = old_model.get_config()
        if self.f is not None:
            self.f.write('\n The model configuration\n')
            self.f.write(model_config)
        for lr in range(len(model_config['layers'])):
            print(model_config['layers'][lr])
            print(model_config['layers'][lr]['config'])
            if self.f is not None:
                self.f.write(model_config['layers'][lr])
                self.f.write(model_config['layers'][lr]['config'])

        for tuple_item in list_updates:
            layer_number, key, value = tuple_item

            if layer_number >= len(model_config['layers']):
                self.log.info(self.updConfigSavedModel.__name__, "Layer {} dont exist".format(layer_number), self.f)
                continue
            if key not in model_config['layers'][layer_number]['config']:
                self.log.info(self.updConfigSavedModel.__name__,
                        "Key {} dont exist on {} layerr config".format(key, layer_number), self.f)
                continue

            # model_config['layers'][0]['config']['batch_input_shape'] = (None, 36, 1)
            model_config['layers'][layer_number]['config'][key] = value
            print(model_config['layers'][layer_number])
            print(model_config['layers'][layer_number]['config'])

            if self.f is not None:
                self.f.write('\nUpdated layer\n')
                self.f.write(model_config['layers'][layer_number])
                self.f.write(model_config['layers'][layer_number]['config'])

        self.model = models.Sequential.from_config(model_config)
        self.model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
        self.model.summary()

        if self.f is not None:
            self.write('\n New model \n')
            self.model.summary(print_fn=lambda x: self.f.write(x + '\n'))

        return

    @exec_time
    def fit_model(self, fit_log:Path = None):
        X, y, X_val, y_val, n_steps, n_features, n_epochs, logfolder, f = self.param_fit
        history = self.model.fit(X, y, epochs=n_epochs, verbose=0, validation_data=(X_val, y_val), )
        msg_history = "\n\nEpochs : {}\n".format(history.epoch)
        for key,value in history.history.items():
            msg_history = msg_history + "\n{} :\n    {}".format(key,value)

        msg0 = "{}-model has been successfully fitted ".format(self.model.name)
        self.log.info(msg0 + '\n')
        msg =f"""

Model {self.model.name}
Training history 

{msg_history}

        """
        if fit_log is None:
           self.log.info(msg)
        else:
            with open(fit_log, 'w') as fout:

                fout.write(msg0 + '\n')
                fout.write(msg)
                self.log.info("{}-model has been successfully logged in {}".format(self.model.name, str(fit_log)))

        return history

    def predict_one_step(self, vec_data):
        pass
        print("{} {}".format(self.__class__.__name__, self.predict_one_step.__name__))
        print(self.predict_one_step.__name__)
        xx_ = vec_data.reshape((1, vec_data.shape[0], 1))
        y_pred = self.model.predict(xx_)
        return y_pred


########################################################################################################################
########################################################################################################################
class MLP(NNmodel):
    pass

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    def myprint(self):
        print("kuku MLP")

    # def mlp_1(self, param):  # n_steps, n_features = 1,hidden_neyron_number=100, dropout_factor=0.2
    def mlp_1(self):
        # define model
        n_steps, n_features, hidden_neyron_number, dropout_factor = self.param
        model = Sequential(name=self.mlp_1.__name__)
        # model.add(tf.keras.Input(shape=( n_steps,1)))
        model.add(Dense(hidden_neyron_number, activation='relu', input_dim=n_steps, name='Layer_0'))

        model.add(layers.Dropout(dropout_factor, name='Layer_1'))
        model.add(Dense(32, name='Layer_2'))
        model.add(layers.Dropout(dropout_factor, name='Layer_3'))
        model.add(Dense(1, name='Layer_4'))
        return model

    def mlp_2(self):  # n_steps, n_features = 1,hidden_neyron_number=100, dropout_factor=0.2
        # define model
        n_steps, n_features, hidden_neyron_number, dropout_factor = self.param
        model = Sequential(name=self.mlp_2.__name__)
        # model.add(tf.keras.Input(shape=( n_steps,1)))
        model.add(Dense(hidden_neyron_number, activation='relu', input_dim=n_steps, name='Layer_0'))

        model.add(layers.Dropout(dropout_factor, name='Layer_1'))
        model.add(Dense(32, name='Layer_2'))
        model.add(Dense(16, name='Layer_3'))

        model.add(Dense(1, name='Layer_4'))

        return model

    ####################################################################################################################
    ####################################################################################################################
    def predict_one_step(self, vec_data):
        pass
        print("{} {}".format(self.__class__.__name__, self.predict_one_step.__name__))
        xx_ = vec_data.reshape((1, vec_data.shape[0]))
        y_pred = self.model.predict(xx_)
        return y_pred


class LSTM(NNmodel):
    pass

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    def vanilla_lstm(self):  # (units, n_steps, n_features) ):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.vanilla_lstm.__name__)
        # model.add( LSTM( units,  activation='relu', input_shape=(n_steps, n_features), name='Layer_0'))
        model.add(tf.keras.layers.LSTM(units, activation='relu', input_shape=(n_steps, n_features), name='Layer_0'))
        model.add(Dense(1, name='Layer_1'))
        try:
            model_name = self.vanilla_lstm.__name__
        except:
            print("cant set model._name")

        return model

    def stacked_lstm(self):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.stacked_lstm.__name__)
        model.add(
            tf.keras.layers.LSTM(units, activation='relu', return_sequences=True, input_shape=(n_steps, n_features),
                                 name='Layer_0'))
        model.add(tf.keras.layers.LSTM(units, activation='relu', name='Layer_1'))
        model.add(Dense(1, name='Layer_2'))
        try:
            model_name = self.stacked_lstm.__name__
        except:
            print("cant set model._name")

        return model

    def bidir_lstm(self):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.bidir_lstm.__name__)
        model.add(Bidirectional(tf.keras.layers.LSTM(units, activation='relu'), input_shape=(n_steps, n_features),
                                name='Layer_0'))
        model.add(Dense(1, name='Layer_1'))
        try:
            model_name = self.bidir_lstm.__name__
        except:
            print("cant set model._name")

        return model

    def cnn_lstm(self):
        units, n_steps, n_features = self.param
        model = None
        model = Sequential()  # name=self.cnn_lstm.__name__)
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                  input_shape=(None, n_steps / 2, n_features), name='Layer_0'))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='Layer_1'))
        model.add(TimeDistributed(Flatten(), name='Layer_2'))
        model.add(tf.keras.layers.LSTM(units, activation='relu', name='Layer_3'))
        model.add(Dense(1, name='Layer_4'))
        try:
            model_name = self.cnn_lstm.__name__
        except:
            print("cant set model._name")

        return model


########################################################################################################################
########################################################################################################################


class CNN(NNmodel):
    pass

    def __init__(self, nameM, typeM, n_steps, n_epochs, f=None):
        super().__init__(nameM, typeM, n_steps, n_epochs, f)

    def univar_cnn(self):
        n_steps, n_features = self.param
        model = Sequential(name=self.univar_cnn.__name__)
        model.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features), name='Layer_0'))
        model.add(MaxPooling1D(pool_size=2, name='Layer_1'))
        model.add(Flatten(name='Layer_2'))
        model.add(Dense(50, activation='relu', name='Layer_3'))
        model.add(Dense(1, name='Layer_4'))

        return model