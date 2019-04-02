from Corpus import *
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
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

from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler,Callback

# F1 Callback https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
class Metrics(Callback):

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        x_val = self.validation_data[0]
        y_true = self.validation_data[1]
        val_predict = (np.asarray(self.model.predict(x_val))).round()
        _val_f1 = f1_score(y_true, val_predict)
        _val_recall = recall_score(y_true, val_predict)
        _val_precision = precision_score(y_true, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return


class Network():

    def __init__(self,
                 X,y,
                 X_val, y_val,
                 X_test, y_test,
                 hidden= [300,200,100],
                 dropout = [],
                 bn = [],
                 multiclass = True):

        self.X = X
        self.X_val = X_val
        self.X_test = X_test
        self.y = y
        self.y_val = y_val
        self.y_test = y_test
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.hidden = hidden
        self.n_hidden = len(hidden)

        if not dropout or len(dropout)!=self.n_hidden:
            self.dropout = [.2 for i in range(self.n_hidden)]
        else:
            self.dropout = dropout
        if not bn or len(bn)!=self.n_hidden:
            self.bn = [True for i in range(self.n_hidden)]
        else:
            self.bn = bn

        self.multiclass = multiclass


    def build_network(self):

        #self.model = Sequential()
        input = Input(shape = (self.input_dim,))

        for l in range(self.n_hidden):
            if l == 0:
                x = Dense(self.hidden[l], activation='relu', kernel_initializer='normal')(input)
            else:
                x = Dense(self.hidden[l], activation='relu', kernel_initializer='normal')(x)
            # self.model.add(Dense(self.hidden[l], activation = 'relu', kernel_initializer='normal'))
            # self.model.add(Dropout(self.dropout[l]))
            x = Dropout(self.dropout[l])(x)
            if self.bn[l]:
                x = BatchNormalization()(x)
                #self.model.add(BatchNormalization())

        if self.output_dim ==1:
            output = Dense(self.output_dim, activation='sigmoid', kernel_initializer='normal')(x)
            #self.model.add(Dense(self.output_dim, activation='sigmoid', kernel_initializer='normal'))
        else:
            output = Dense(self.output_dim, activation='softmax', kernel_initializer='normal')(x)
            #self.model.add(Dense(self.output_dim, activation='softmax', kernel_initializer='normal'))

        self.model = Model(inputs = input, outputs = output)


    def fit(self,
            optimizer = optimizers.Adam,
            lr = .01,
            epochs = 20,
            batch_size = 32):

        #metrics = Metrics(validation_data=(corpus.X.values[450:,], corpus.y.values[450:]))
        net.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics = ['accuracy'])
        net.model.fit(self.X, self.y,
                      validation_data=(self.X_val, self.y_val),
                      epochs=epochs, batch_size=batch_size
                      #callbacks=[metrics]
        )

class SVM():

    def __init__(self):
        pass



if __name__ == "__main__":
    import pickle

    with open('test_corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test = corpus.generate_model_data()

    net = Network(X_train,y_train,
                  X_val, y_val,
                  X_test, y_test,
                  hidden = [100,100,100,100], dropout=[.6,.6,.6,.6])

    net.build_network()
    net.fit(epochs=40)

