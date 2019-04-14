import pickle
import numpy as np
import tensorflow as tf
import random as rn
import os
from Document import *
from Corpus import *
from models import Network, SVM

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

with open('data/full_corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

print(corpus.authors['author'].iloc[[0,100]].values)

# model_data  = corpus.generate_model_data(type='onevsone',
#                                         model_authors = corpus.authors['author'].iloc[[0,100]].values,
#                                         sampling = 'oversample',
#                                         #feature_sets = ['lex', 'pos', 'word', 'char']
#                                          feature_sets = ['voice', 'coref']
#                                                  )

# model_data  = corpus.generate_model_data(type='multiclass',
#                                         model_authors = [],
#                                         sampling = 'oversample',
#                                         feature_sets = ['lex', 'pos', 'word', 'char','voice','coref'],
#                                        # feature_sets = [ 'lex'],
#                                         encoding_type = 'onehot')
#
#
# net = Network(**model_data)
#
# net.build_network( hidden = [1000],
#                    dropout=[.1],
#                    kernel_regulizer=None,
#                    bias_regulizer=None)
# net.fit(epochs=500, stop_patience=10, batch_size=50, optimizer=tf.train.AdamOptimizer())
# print(net.predict_test())




model_data  = corpus.generate_model_data(type='multiclass',
                                        model_authors = [],
                                        sampling = 'oversample',
                                        #feature_sets = ['lex', 'pos', 'word', 'char','voice','coref'],
                                        feature_sets = [ 'voice','coref'],
                                        encoding_type = 'label')


svm = SVM(**model_data)

svm.tune()
print(svm.model)
svm.fit_svm()
#print(net.predict_test())