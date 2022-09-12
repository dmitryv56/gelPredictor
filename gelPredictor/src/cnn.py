#!/usr/bin/env python3

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import batch_normalization
from keras.layers.activation import LeakyReLU
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sys_util.parseConfig import BATCH_SIZE, EPOCHS, NUM_CLASSES, ALFA_RELU, DROPOUT, DENSE_INPUT, NUM_KERNELS

logger = logging.getLogger(__name__)

(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
logger.info('Training data shape : ', train_X.shape, train_Y.shape)

logger.info('Testing data shape : ', test_X.shape, test_Y.shape)

classes = np.unique(train_Y)
nClasses = len(classes)
logger.info('Total number of outputs : ', nClasses)
logger.info('Output classes : ', classes)
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))
plt.savefig("fashio.png")
plt.close("all")

train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, test_X.shape

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
logger.info('Original label:', train_Y[0])
logger.info('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
msg=print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
logger.info(msg)

class HyperParams(object):

    def __init__(self, num_classes:int = NUM_CLASSES):
        self.epochs =EPOCHS
        self.batch_size =BATCH_SIZE
        self.num_classes=num_classes
        self.alfa_relu =  ALFA_RELU
        self.dropout = DROPOUT
        self.dense_input = DENSE_INPUT
        self.num_kernels = NUM_KERNELS
        self.kernel_size = (3,3)
        self.pool_size = (2,2)

class CNN(object):


    def __init__(self , model_repository: Path =None, log_folder:Path=None, chart_folder:Path=None):
        self.cnn_model=None
        self.hp =HyperParams(num_classes=0 )
        self.input_shape=()
        self.log=logger
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        if model_repository is None:
            self.model_repository =  Path("train_dropout").with_suffix(".h5py")
        else:
            self.model_repository = Path(model_repository)/Path("train_dropout").with_suffix(".h5py")
        self.log_folder = log_folder
        self.chart_folder=chart_folder

    def model(self):

        if self.train_X is None :
            self.log.error("Exit! No input data!")
            return -1
        self.cnn_model = Sequential()
        self.cnn_model.add(
            Conv2D(self.hp.num_kernels, kernel_size=self.hp.kernel_size, activation='linear', padding='same',
                   input_shape=self.input_shape))
        self.cnn_model.add(LeakyReLU(alpha=self.hp.alfa_relu))
        self.cnn_model.add(MaxPooling2D(pool_size=self.hp.pool_size, padding='same'))
        self.cnn_model.add(Dropout(self.hp.dropout))
        self.cnn_model.add(Conv2D(self.hp.num_kernels * 2, (3, 3), activation='linear', padding='same'))
        self.cnn_model.add(LeakyReLU(alpha=self.hp.alfa_relu))
        self.cnn_model.add(MaxPooling2D(pool_size=self.hp.pool_size, padding='same'))
        self.cnn_model.add(Dropout(self.hp.dropout))
        self.cnn_model.add(Conv2D(self.hp.num_kernels * 4, (3, 3), activation='linear', padding='same'))
        self.cnn_model.add(LeakyReLU(alpha=self.hp.alfa_relu))
        self.cnn_model.add(MaxPooling2D(pool_size=self.hp.pool_size, padding='same'))
        self.cnn_model.add(Dropout(self.hp.dropout + 0.15))
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(self.hp.dense_input, activation='linear'))
        self.cnn_model.add(LeakyReLU(alpha=self.hp.alfa_relu))
        self.cnn_model.add(Dropout(self.hp.dropout + 0.05))
        self.cnn_model.add(Dense(self.hp.num_classes, activation='softmax'))

        self.log.info(" Convolution Neural Net")
        if self.log_folder is None:
            model_log ="cnn_nodel.txt"
        else:
            model_log =Path(self.log_folder /"cnn_nodel").with_suffix(".txt")
        with open(model_log, 'w') as fw:
           self.cnn_model.summary(print_fn=lambda x: fw.write(x + '\n'))

        self.cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

        return 0

    def prepare_train_data(self, train_X:np.ndarray = None,train_Y:np.ndarray = None, test_X: np.ndarray = None,
                           test_Y:np.ndarray = None):
        pass
        (k,n1,n2)=train_X.shape
        self.log.info('Training data shape : {}  -  {}'.format( train_X.shape, train_Y.shape))

        self.log.info('Testing data shape : {}  -  {}'.format( test_X.shape, test_Y.shape))
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        classes = np.unique(self.train_Y)
        self.hp.num_classes = len(classes)
        self.log.info('Total number of outputs : {}'.format( self.hp.num_classes))
        self.log.info('Output classes : {}'.format(classes))


        self.train_X = self.train_X.reshape(-1, n1, n2, 1)
        self.test_X = self.test_X.reshape(-1, n1, n2, 1)
        self.input_shape=(n1,n2,1)
        self.log.info("After reshaping\n train_X ={} test_X {}\n".format(self.train_X.shape, self.test_X.shape))

        self.train_X = self.train_X.astype('float32')
        self.test_X = self.test_X.astype('float32')
        # self.train_X = self.train_X / 255.
        # self.test_X = self.test_X / 255.

        self.train_Y_one_hot = to_categorical(self.train_Y)
        (ntrain,mtrain)=self.train_Y_one_hot.shape
        self.test_Y_one_hot = to_categorical(self.test_Y)
        (ntest, mtest) = self.test_Y_one_hot.shape

        self.log.info("test_Y_one_hot.shape : {}".format(self.test_Y_one_hot.shape))
        if (mtest<mtrain):
            self.test_Y_one_hot = np.hstack((self.test_Y_one_hot,np.zeros((ntest, mtrain -mtest))))
            self.log.info("test_Y_one_hot.shape after correction: {}".format(self.test_Y_one_hot.shape))
        # Display the change for category label using one-hot encoding

        return

    def fit_cnn(self):
        self.train_X, self.valid_X, self.train_label, self.valid_label = train_test_split(self.train_X,
                                                    self.train_Y_one_hot, test_size=0.2, random_state=13)
        msg = f""" 
After split
Train shape:             {self.train_X.shape}
Validation shape:        {self.valid_X.shape} 
Train labels shape:      {self.train_label.shape} 
Validation labels shape: {self.valid_label.shape}

        """
        self.log.info(msg)

        self.train_dropout = self.cnn_model.fit(self.train_X, self.train_label, batch_size=self.hp.batch_size,
                                           epochs=self.hp.epochs, verbose=1,
                                           validation_data=(self.valid_X, self.valid_label))

        self.cnn_model.save(self.model_repository)
        self.log.info("Trained mode saved : {}".format(self.model_repository))

        self.test_eval = self.cnn_model.evaluate(self.test_X, self.test_Y_one_hot, verbose=1)
        msg = f"""
Test loss     :  {self.test_eval[0]}
Test accuracy :  {self.test_eval[1]}

"""
        print(msg)
        self.log.info(msg)
        return

    def AccuracyChart(self):

        if self.chart_folder is None:
            train_val_acc="Training_Validation_Accuracy.png"
            train_val_loss = "Training_Validation_Loss.png"
        else:
            train_val_acc = Path(self.chart_folder/Path("Training_Validation_Accuracy")).with_suffix(".png")
            train_val_loss = Path(self.chart_folder/Path("Training_Validation_Loss")).with_suffix(".png")

        accuracy = self.train_dropout.history['accuracy']
        val_accuracy = self.train_dropout.history['val_accuracy']
        loss = self.train_dropout.history['loss']
        val_loss = self.train_dropout.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.savefig(train_val_acc)
        plt.close("all")
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(train_val_loss)
        plt.close("all")



def drive():
    cnnModel(train_X, train_label, valid_X, valid_label, test_X, test_Y_one_hot)


def cnnModel(train_X, train_label, valid_X, valid_label, test_X, test_Y_one_hot)->keras.models.Sequential:

    cnn_model = Sequential()
    cnn_model.add(Conv2D(NUM_KERNELS, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28, 28, 1)))
    cnn_model.add(LeakyReLU(alpha=ALFA_RELU))
    cnn_model.add(MaxPooling2D((2, 2), padding='same'))
    cnn_model.add(Dropout(DROPOUT))
    cnn_model.add(Conv2D(NUM_KERNELS*2, (3, 3), activation='linear', padding='same'))
    cnn_model.add(LeakyReLU(alpha=ALFA_RELU))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(Dropout(DROPOUT))
    cnn_model.add(Conv2D(NUM_KERNELS*4, (3, 3), activation='linear', padding='same'))
    cnn_model.add(LeakyReLU(alpha=ALFA_RELU))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(Dropout(DROPOUT+0.15))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(DENSE_INPUT, activation='linear'))
    cnn_model.add(LeakyReLU(alpha=ALFA_RELU))
    cnn_model.add(Dropout(DROPOUT + 0.05))
    cnn_model.add(Dense(NUM_CLASSES, activation='softmax'))

    logger.info(" Convolution Neural Net")
    cnn_model.summary(print_fn=lambda x: logger.info(x + '\n'))

    cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    train_dropout = cnn_model.fit(train_X, train_label, batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,
                                  validation_data=(valid_X, valid_label))
    cnn_model.save("train_dropout.h5py")

    test_eval = cnn_model.evaluate(test_X, test_Y_one_hot, verbose=1)

    logger.info("Test loss: test_eval[0]")
    logger.info("Test accuracy: test_eval[1]")

    accuracy = train_dropout.history['acc']
    val_accuracy = train_dropout.history['val_acc']
    loss = train_dropout.history['loss']
    val_loss = train_dropout.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("cnn_train_dropout.png")
    plt.close("all")

    return cnn_model






if __name__ == "__main__":
    pass