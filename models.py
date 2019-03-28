import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline

### SVM Imports ###
from sklearn import svm


### NN Imports ###
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras import optimizers
from keras import regularizers
from keras.layers.core import *
from keras.layers import (
    Dense, Flatten,Input,
    Dropout,BatchNormalization
)

from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler



class Network():

    def __init__(self,
                 X,y,
                 hidden= [300,200,100],
                 dropout = [.25,.25,.25],
                 bn = [False, False, False],
                 multiclass = True):

        self.X = X
        self.y = y
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.hidden = hidden
        self.dropout = dropout,
        self.bn = bn
        self.multiclass = multiclass


    def build_network(self):

        self.model = Sequential()

        for l in range(len(self.hidden)):
            self.model.add(Dense(self.hidden[l], activation = 'relu', kernel_initializer='normal'))
            self.model.add(Dropout(self.dropout[l]))
            if self.bn[l]:
                self.model.add(BatchNormalization())

        if self.output_dim ==1:
            self.model.add(Dense(self.output_dim, activation='sigmoid', kernel_initializer='normal'))
        else:
            self.model.add(Dense(self.output_dim, activation='softmax', kernel_initializer='normal'))




class SVM():

    def __init__(self):
        pass