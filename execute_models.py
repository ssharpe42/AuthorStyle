import pickle
import numpy as np
import tensorflow as tf
import random as rn
import os
import itertools
import warnings
from Corpus import *
from models import Network, SVM

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
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

with open('data/full_corpus_100.pkl', 'rb') as f:
    corpus = pickle.load(f)


feature_tests = {'lexico_syntactic': ('lex','pos','word','char'),
                'coref': ('coref',),
                'voice':('voice',),
                'coref+voice':('coref','voice'),
                'lexico+voice+coref': ('lex','pos','word','char','coref','voice'),
                 'lexico+voice':('lex','pos','word','char','voice'),
                 'lexico+coref': ('lex','pos','word','char','coref')}

##########################################
# Compile One vs One Acc Estimates
##########################################

onevsone_results = {x: {f:[] for f in feature_tests} for x in itertools.combinations(np.unique(corpus.data['author'].values), 2)}

for comb in onevsone_results:

    print('Fitting Authors: {}'.format(comb))

    for type, feat in feature_tests.items():
        model_data  = corpus.generate_model_data(type='onevsone',
                                                model_authors = list(comb),
                                                feature_sets = list(feat))

        n_features = model_data['X_train'][0].shape[1]
        n_neurons = int(n_features*2.0/3)

        #Cross validation acc
        cv_acc = []

        for k in range(5):

            print('Fitting Type {} Fold {}'.format(type, k + 1))
            net = Network(X_train=model_data['X_train'][k],
                          X_val = model_data['X_val'][k],
                          X_test=model_data['X_test'][k],
                          y_train=model_data['y_train'][k],
                          y_val=model_data['y_val'][k],
                          y_test=model_data['y_test'][k],
                          encoder=model_data['encoder'])
            net.build_network(hidden = [n_neurons],
                           dropout=[.1],
                           kernel_regulizer=.0001,
                           bias_regulizer=.0001)
            net.fit(epochs=500, stop_patience=5, batch_size=32, optimizer=tf.train.AdamOptimizer(), verbose=0)
            pred_dict = net.predict_test()

            cv_acc.append(np.mean(pred_dict['y_true_text']==pred_dict['y_pred_text']))

        onevsone_results[comb][type] = cv_acc


##########################################
# Compile One vs All Acc Estimates
##########################################

onevsall_results = {x: {f:[] for f in feature_tests} for x in np.unique(corpus.data['author'].values)}

for auth in onevsall_results:

    print('Fitting Author: {}'.format(auth))

    for type, feat in feature_tests.items():


        model_data = corpus.generate_model_data(type='onevsall',
                                                model_authors=[auth],
                                                feature_sets=list(feat))

        n_features = model_data['X_train'][0].shape[1]
        n_neurons = int(n_features*2.0/3)

        #Cross validation acc
        cv_acc = []

        for k in range(5):

            print('Fitting Type {} Fold {}'.format(type, k+1))

            net = Network(X_train=model_data['X_train'][k],
                          X_val = model_data['X_val'][k],
                          X_test=model_data['X_test'][k],
                          y_train=model_data['y_train'][k],
                          y_val=model_data['y_val'][k],
                          y_test=model_data['y_test'][k],
                          encoder=model_data['encoder'])
            net.build_network( hidden = [n_neurons],
                           dropout=[.1],
                           kernel_regulizer=.0001,
                           bias_regulizer=.0001)
            net.fit(epochs=500, stop_patience=5, batch_size=32, optimizer=tf.train.AdamOptimizer(), verbose=0)
            pred_dict = net.predict_test()

            cv_acc.append(np.mean(pred_dict['y_true_text']==pred_dict['y_pred_text']))

        onevsall_results[auth][type] = cv_acc

    print(onevsall_results)

##########################################
# Compile Multiclass Estimates
##########################################
multiclass_results = {f:[] for f in feature_tests}


for type, feat in feature_tests.items():

    model_data  = corpus.generate_model_data(type='multiclass',
                                        model_authors = [],
                                        feature_sets = list(feat))

    n_features = model_data['X_train'][0].shape[1]
    n_neurons = int(n_features * 2.0 / 3)

    #Cross validation acc
    cv_acc = []
    for k in range(5):

        print('Fitting Fold {}'.format(k+1))
        net = Network(X_train=model_data['X_train'][k],
                      X_val = model_data['X_val'][k],
                      X_test=model_data['X_test'][k],
                      y_train=model_data['y_train'][k],
                      y_val=model_data['y_val'][k],
                      y_test=model_data['y_test'][k],
                      encoder=model_data['encoder'])
        net.build_network( hidden = [n_neurons],
                           dropout=[.1],
                           kernel_regulizer=.0001,
                           bias_regulizer=.0001)
        net.fit(epochs=500, stop_patience=10, batch_size=32, optimizer=tf.train.AdamOptimizer(), verbose=0)
        pred_dict = net.predict_test()

        cv_acc.append(np.mean(pred_dict['y_true_text']==pred_dict['y_pred_text']))

    multiclass_results[type] = cv_acc



# model_data  = corpus.generate_model_data(type='onevsone',
#                                         model_authors = corpus.authors['author'].iloc[[0,1000]].values,
#                                         sampling = 'oversample',
#                                         feature_sets = ['lex', 'pos', 'word', 'char','voice','coref'],
#                                         #feature_sets = [ 'voice','coref'],
#                                         encoding_type = 'label')
#
#
# svm = SVM(**model_data)
#
# svm.tune(n_iter=50, cv = 3)
# print(svm.model)
# svm.fit_svm()
# pred_dict = svm.predict_test()
# print(np.mean(pred_dict['y_true_text']==pred_dict['y_pred_text']))
