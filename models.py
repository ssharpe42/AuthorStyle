from Corpus import *
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline

### SVM Imports ###
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

### NN Imports ###
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras import optimizers
from keras import regularizers
from keras.layers.core import *
from keras.layers import (
    Dense, Flatten,Input,
    Dropout,BatchNormalization
)
from AMSGrad import AMSGrad
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler,Callback


class SVM():

    def __init__(self,
                 X_train,y_train,
                 X_val, y_val,
                 X_test, y_test,
                 encoder=None):

        self.X = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.encoder = encoder
        self.model = SVC(probability=True)

    def tune(self):

        params = {'kernel':['rbf'],'gamma':[1, .5, .2, .1, 1e-3, 1e-4],
                  'C':[.5,1,2,5,8,10,15,20,25,30,50,100]}

        self.tuner = RandomizedSearchCV(self.model,
                                        param_distributions=params,
                                        n_iter=300,
                                        cv = 4,
                                        #scoring='neg_log_loss',
                                        verbose = 3)

        self.tuner.fit(self.X, self.y)
        self.model =  SVC(**self.tuner.best_params_)

    def fit_svm(self):

        self.model = SVC()
        self.model.fit(self.X, self.y)

    def predict_test(self):

        self.y_pred = self.model.predict(self.X_test)

        self.y_pred_text = self.encoder.inverse_transform(self.y_pred)
        self.y_true_text = self.encoder.inverse_transform(self.y_test)

        return {'y_true':self.y_test, 'y_pred':self.y_pred,
                'y_true_text':self.y_true_text,'y_pred_text':self.y_pred_text}



class Network():

    def __init__(self,
                 X_train,y_train,
                 X_val, y_val,
                 X_test, y_test,
                 encoder = None):

        self.X = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.input_dim = X_train.shape[1]
        self.output_dim = y_train.shape[1]
        self.encoder = encoder

        self.multiclass = y_train.shape[1]>1

        if self.multiclass:
            self.loss = 'categorical_crossentropy'
        else:
            self.loss = 'binary_crossentropy'


    def build_network(self,
                      hidden= [300,200,100],
                     dropout = [],
                     bn = [],
                     kernel_regulizer = 0.0001,
                     bias_regulizer = 0.0001):

        self.hidden = hidden
        self.n_hidden = len(hidden)
        self.kernel_regulizer = regularizers.l2(kernel_regulizer) if kernel_regulizer else None
        self.bias_regulizer = regularizers.l2(bias_regulizer) if bias_regulizer else None

        if not dropout or len(dropout)!=self.n_hidden:
            self.dropout = [.2 for i in range(self.n_hidden)]
        elif len(dropout)!=self.n_hidden and len(dropout)==1:
            self.dropout = [dropout[0] for i in range(self.n_hidden)]
        else:
            self.dropout = dropout
        if not bn or len(bn)!=self.n_hidden:
            self.bn = [True for i in range(self.n_hidden)]
        else:
            self.bn = bn

        input = Input(shape = (self.input_dim,))

        for l in range(self.n_hidden):
            if l == 0:
                x = Dense(self.hidden[l], activation='relu', kernel_initializer='normal',
                          kernel_regularizer=self.kernel_regulizer, bias_regularizer=self.bias_regulizer)(input)
            else:
                x = Dense(self.hidden[l], activation='relu', kernel_initializer='normal',
                          kernel_regularizer=self.kernel_regulizer, bias_regularizer=self.bias_regulizer)(x)

            x = Dropout(self.dropout[l])(x)
            if self.bn[l]:
                x = BatchNormalization()(x)


        if self.output_dim ==1:
            output = Dense(self.output_dim, activation='sigmoid', kernel_initializer='normal',
                           kernel_regularizer=self.kernel_regulizer, bias_regularizer=self.bias_regulizer)(x)
        else:
            output = Dense(self.output_dim, activation='softmax', kernel_initializer='normal',
                           kernel_regularizer=self.kernel_regulizer, bias_regularizer=self.bias_regulizer)(x)

        self.model = Model(inputs = input, outputs = output)


    def fit(self,
            optimizer = tf.train.AdamOptimizer(),
            epochs = 20,
            batch_size = 32,
            early_stopping = True,
            stop_patience = 10):

            #1.4547118e-05, 1.3314760e-06, 5.3525422e-05

        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['categorical_accuracy'])

        if early_stopping:
            earlystopper = EarlyStopping(patience=stop_patience, verbose=5, restore_best_weights=True)
            self.model.fit(self.X, self.y,
                          validation_data=(self.X_val, self.y_val),
                          epochs=epochs, batch_size=batch_size,
                          callbacks=[earlystopper]
            )
        else:
            self.model.fit(self.X, self.y,
                          validation_data=(self.X_val, self.y_val),
                          epochs=epochs, batch_size=batch_size
            )


    def predict_test(self):

        self.y_pred = self.model.predict(self.X_test)

        self.y_pred_text = self.encoder.inverse_transform(self.y_pred)
        self.y_true_text = self.encoder.inverse_transform(self.y_test)

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
                  hidden = [1000],
                  dropout=[.1],
                  multiclass= y_train.shape[1]>1,
                  encoder = encoder)

    net.build_network()
    net.fit(epochs=100)
    print(net.predict_test())
