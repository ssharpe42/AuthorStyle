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

class Network():

    def __init__(self,
                 X,y,
                 X_val, y_val,
                 X_test, y_test,
                 hidden= [300,200,100],
                 dropout = [],
                 bn = [],
                 multiclass = True,
                 encoder = None):

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
        self.encoder = encoder

        if not dropout or len(dropout)!=self.n_hidden:
            self.dropout = [.2 for i in range(self.n_hidden)]
        else:
            self.dropout = dropout
        if not bn or len(bn)!=self.n_hidden:
            self.bn = [True for i in range(self.n_hidden)]
        else:
            self.bn = bn

        self.multiclass = multiclass

        if self.multiclass:
            self.loss = 'categorical_crossentropy'
        else:
            self.loss = 'binary_crossentropy'

    def build_network(self):

        #self.model = Sequential()
        input = Input(shape = (self.input_dim,))

        for l in range(self.n_hidden):
            if l == 0:
                x = Dense(self.hidden[l], activation='relu', kernel_initializer='normal')(input)
            else:
                x = Dense(self.hidden[l], activation='relu', kernel_initializer='normal')(x)

            x = Dropout(self.dropout[l])(x)
            if self.bn[l]:
                x = BatchNormalization()(x)


        if self.output_dim ==1:
            output = Dense(self.output_dim, activation='sigmoid', kernel_initializer='normal')(x)
        else:
            output = Dense(self.output_dim, activation='softmax', kernel_initializer='normal')(x)

        self.model = Model(inputs = input, outputs = output)


    def fit(self,
            optimizer = optimizers.Adam,
            epochs = 20,
            batch_size = 32,
            early_stopping = True,
            stop_patience = 10):


        net.model.compile(optimizer='adam', loss=self.loss, metrics=[self.loss, 'accuracy'])

        if early_stopping:
            earlystopper = EarlyStopping(patience=stop_patience, verbose=1, restore_best_weights=True)
            net.model.fit(self.X, self.y,
                          validation_data=(self.X_val, self.y_val),
                          epochs=epochs, batch_size=batch_size,
                          callbacks=[earlystopper]
            )
        else:
            net.model.fit(self.X, self.y,
                          validation_data=(self.X_val, self.y_val),
                          epochs=epochs, batch_size=batch_size
            )


    def predict_test(self):

        self.y_pred = net.model.predict(self.X_test)

        self.y_pred_text = encoder.inverse_transform(self.y_pred)
        self.y_true_text = encoder.inverse_transform(self.y_test)

        return {'y_true':self.y_test, 'y_pred':self.y_pred,
                'y_true_text':self.y_true_text,'y_pred_text':self.y_pred_text}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)



if __name__ == "__main__":
    import pickle
    import numpy as np
    import tensorflow as tf
    import random as rn
    import os


    ########  SET FOR REPRODUCIBLE RESULTS #########

    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers in a well-defined state.
    rn.seed(42)
    # Force TensorFlow to use single thread.
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    from keras import backend as K
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state
    tf.set_random_seed(42)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    ################################################


    with open('test_corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, encoder = corpus.generate_model_data(type='multiclass',
                                                                                #model_authors = [corpus.authors['author'].iloc[0]],
                                                                                sampling = 'oversample',
                                                                                feature_sets = ['coref'])

    net = Network(X_train,y_train,
                  X_val, y_val,
                  X_test, y_test,
                  hidden = [100,100,100],
                  dropout=[.6,.6,.6],
                  multiclass= y_train.shape[1]>1,
                  encoder = encoder)

    net.build_network()
    net.fit(epochs=100)
    print(net.predict_test())
