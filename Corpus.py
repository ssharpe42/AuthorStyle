import pickle
import string

import numpy as np
import pandas as pd
# from imblearn.datasets import make_imbalance,
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer,  LabelEncoder
from Document import Document

class Corpus():

    def __init__(self):

        self.documents = []
        self.feature_sets = {}

    def init_docs(self,
                  char_ngrams=(2, 2),
                  char_topk=1000,
                  word_ngrams=(1, 1),
                  word_topk=10000,
                  pos_ngrams=(1, 1),
                  word_lemma=True,
                  word_entities=False,
                  word_punct=False,
                  pos_detailed=False,
                  char_punct=True,
                  char_lower=False,
                  coref_n=2,
                  coref_pos_types=['DT', 'NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$'],
                  coref_dependencies=['dobj', 'nsubj', 'nsubjpass', 'pobj', 'poss'],
                  coref_group=True,
                  passive_mapper=None):

        """Initialize and process documents"""

        self.char_ngrams = char_ngrams
        self.char_topk = char_topk
        self.word_ngrams = word_ngrams
        self.word_topk = word_topk
        self.pos_ngrams = pos_ngrams
        self.word_lemma = word_lemma
        self.word_entities = word_entities
        self.word_punct = word_punct
        self.pos_detailed = pos_detailed
        self.char_punct = char_punct
        self.char_lower = char_lower

        self.coref_n = coref_n
        self.coref_pos_types = coref_pos_types
        self.coref_dependencies = coref_dependencies
        self.coref_group = coref_group
        self.passive_mapper = passive_mapper

        self.word_ngrams = word_ngrams
        self.pos_ngrams = pos_ngrams
        self.char_vectorizer = CountVectorizer(analyzer='char', preprocessor=None if char_punct else self.preprocessor,
                                               lowercase=char_lower, ngram_range=char_ngrams, max_features=char_topk)
        self.word_vectorizer = CountVectorizer(tokenizer=self.tokenization, ngram_range=word_ngrams,
                                               max_features=word_topk)
        self.pos_vectorizer = CountVectorizer(tokenizer=self.tokenization, ngram_range=pos_ngrams)

        for i in range(len(self.documents)):
            print('Processing doc {} of {}'.format(i, len(self.documents)))
            self.documents[i].process_doc(
                word_lemma=self.word_lemma,
                word_entities=self.word_entities,
                word_punct=self.word_punct,
                pos_detailed=self.pos_detailed,
                coref_n=self.coref_n,
                coref_pos_types=self.coref_pos_types,
                coref_dependencies=self.coref_dependencies,
                coref_group=self.coref_group,
                passive_mapper=self.passive_mapper
            )

        self.n_docs = len(self.documents)

    def tokenization(self, doc):
        return doc.split(' ')

    def preprocessor(self, doc):
        return doc.translate(str.maketrans('', '', string.punctuation))

    def add_document(self, document):

        assert isinstance(document, Document), "Add only documents"

        self.documents.append(document)

    def fit_char_vectorizer(self):

        doc_text = [d.text for d in self.documents]

        self.char_vectorizer.fit(doc_text)
        self.char_vocab = self.char_vectorizer.get_feature_names()
        char_counts = self.char_vectorizer.fit_transform(doc_text)
        self.char_mat = pd.DataFrame(char_counts.toarray(), columns=self.char_vocab)

        #Normalize to probabilities
        self.char_mat = self.char_mat.div(self.char_mat.sum(axis = 1), axis = 0)

        self.feature_sets['char'] = self.char_vocab

    def fit_word_vectorizer(self):

        clean_doc_text = [' '.join(d.words) for d in self.documents]

        self.word_vectorizer.fit(clean_doc_text)
        self.word_vocab = self.word_vectorizer.get_feature_names()
        word_counts = self.word_vectorizer.fit_transform(clean_doc_text)
        self.word_mat = pd.DataFrame(word_counts.toarray(), columns=self.word_vocab)

        # Normalize to probabilities
        self.word_mat = self.word_mat.div(self.word_mat.sum(axis=1), axis=0)


        self.feature_sets['word'] = self.word_vocab

    def fit_pos_vectorizer(self):

        clean_doc_pos = [' '.join(d.pos_tokens) for d in self.documents]

        self.pos_vectorizer.fit(clean_doc_pos)
        self.pos_vocab = self.pos_vectorizer.get_feature_names()
        pos_counts = self.pos_vectorizer.fit_transform(clean_doc_pos)
        self.pos_mat = pd.DataFrame(pos_counts.toarray(), columns=self.pos_vocab)

        # Normalize to probabilities
        self.pos_mat = self.pos_mat.div(self.pos_mat.sum(axis=1), axis=0)

        self.feature_sets['pos'] = self.pos_vocab

    # ....etc
    def coref_features(self):

        self.coref_prob = pd.DataFrame([self.documents[i].coref_prob for i in range(self.n_docs)]).fillna(0)
        self.coref_spans = pd.DataFrame([self.documents[i].coref_spans for i in range(self.n_docs)]).fillna(0)

        self.coref_misc = pd.DataFrame({'mean_mentions': [np.mean(d.coref_mentions) for d in self.documents],
                                        'sd_mentions': [np.std(d.coref_mentions) for d in self.documents],
                                        'mentions_per_sent': [
                                            np.mean(np.array(d.coref_mentions) / np.array(d.coref_unq_sents)) for d in
                                            self.documents]})

        self.coref_misc = self.coref_misc.fillna(0)

        self.coref_mat = pd.concat([self.coref_prob, self.coref_spans, self.coref_misc], axis=1)

        self.feature_sets['coref'] = self.coref_mat.columns.values

    def lexical_features(self):

        self.lex_mat = pd.DataFrame({'sent_length': [np.mean(d.sent_lengths) for d in self.documents],
                                     'sent_std': [np.std(d.sent_lengths) for d in self.documents],
                                     'word_length': [np.mean(d.word_lengths) for d in self.documents],
                                     'word_std': [np.std(d.word_lengths) for d in self.documents],
                                     'pct_doc_stopword': [np.sum([1 for w in d.words if w in STOP_WORDS ])/len(d.words) for d in self.documents],
                                     'pct_vocab_stopword': [ len(set(d.words) & STOP_WORDS)/len(set(d.words)) for d in self.documents],
                                     'vocab_richness': [d.VR for d in self.documents] })

        self.feature_sets['lex'] = self.lex_mat.columns.values

    def voice_features(self):

        self.voice_mat = pd.DataFrame({"hattrick_freq" : [d.hattrick_freq for d in self.documents],
                                       "agentless_freq" : [d.agentless_freq for d in self.documents],
                                       "passive_desc_freq" : [d.passive_desc_freq for d in self.documents],
                                       "no_active_freq" : [d.no_active_freq for d in self.documents],
                                       "get_freq" : [d.get_freq for d in self.documents],
                                       "be_freq" : [d.be_freq for d in self.documents],
                                       "other_freq" : [d.other_freq for d in self.documents]
                                       }).fillna(0)

        self.feature_sets['voice'] = self.voice_mat.columns.values


    def build_data(self):

        self.fit_char_vectorizer()
        self.fit_word_vectorizer()
        self.fit_pos_vectorizer()
        self.lexical_features()
        self.coref_features()
        self.voice_features()

        self.authors = pd.DataFrame({'author': [d.author for d in self.documents]})
        self.X_ = pd.concat([self.char_mat, self.word_mat, self.pos_mat, self.lex_mat, self.coref_mat, self.voice_mat],
                            axis=1)
        self.feature_ids = np.array(range(self.X_.shape[1]))
        self.features = {f: indx for indx, f in enumerate(self.X_)}
        self.data = pd.concat([self.authors, self.X_], axis=1)

        self.X_ = self.X_.values

    def generate_model_data(self, type='multiclass',
                            model_authors=[],
                            feature_sets=['lex', 'coref', 'pos', 'word', 'char', 'voice'],
                            encoding_type = 'onehot',
                            random_state = 42):

        if encoding_type == 'onehot':
            encoder = LabelBinarizer()
        else:
            encoder = LabelEncoder()


        features = []
        for f in feature_sets:
            features.extend(self.feature_sets[f])

        feature_indx = np.array([self.features[f] for f in features])

        X = self.X_[:, feature_indx]

        # Type is one of 'multiclass' or 'onevsone' or 'onevsall'
        if type == 'onevsone':

            author_indx = np.where(self.authors.isin(model_authors))[0]
            X = X[author_indx]
            authors = self.authors.iloc[author_indx]
            y = encoder.fit_transform(authors)

        elif type == 'onevsall':

            author_indx = ~self.authors.isin(model_authors)
            authors = self.authors.copy()
            authors[author_indx] = 'Other'
            y = encoder.fit_transform(authors)

        elif type == 'multiclass':

            authors = self.authors
            y = encoder.fit_transform(authors)

        #Split into k folds
        kf = KFold(n_splits=5, shuffle = True, random_state=42)
        train_indx = []
        test_indx = []
        for train, test in kf.split(X, y):
            train_indx.append(train)
            test_indx.append(test)

        data_dict = {'X_train':[],
                     'X_val':[],
                     'X_test':[],
                     'y_train': [],
                     'y_val':[],
                     'y_test':[],
                     'encoder':encoder}

        for k in range(5):
            X_train, X_val, y_train, y_val, author_train, author_val = train_test_split(X[train_indx[k]], y[train_indx[k]],
                                                                                        authors.iloc[train_indx[k]],
                                                                                        test_size=.2,
                                                                                        random_state=42)

            X_test = X[test_indx[k]]
            y_test = y[test_indx[k]]

            ros = RandomOverSampler(sampling_strategy='not majority', random_state=42, return_indices=True)

            _, _, indx = ros.fit_resample(X_train, y_train)
            X_train = X_train[indx]
            y_train = y_train[indx]

            _, _, indx = ros.fit_resample(X_val, y_val)
            X_val = X_val[indx]
            y_val = y_val[indx]

            _, _, indx = ros.fit_resample(X_test, y_test)
            X_test = X_test[indx]
            y_test = y_test[indx]

            data_dict['X_train'].append(X_train)
            data_dict['X_val'].append(X_val)
            data_dict['X_test'].append(X_test)
            data_dict['y_train'].append(y_train)
            data_dict['y_val'].append(y_val)
            data_dict['y_test'].append(y_test)

        return data_dict

    def save(self, filename):
        # Cant pickle spacy docs
        self.documents = []
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
