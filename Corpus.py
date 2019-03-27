from Document import Document

import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Corpus():

    def __init__(self,
                 char_ngrams = (2,2),
                 word_ngrams = (1,1),
                 pos_ngrams = (1,1),
                 word_lemma=True,
                 word_entities=False,
                 word_punct=False,
                 pos_detailed = False,
                 char_punct = True,
                 char_lower = False):


        #Parameters
        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.pos_ngrams = pos_ngrams
        self.word_lemma = word_lemma
        self.word_entities = word_entities
        self.word_punct = word_punct
        self.pos_detailed = pos_detailed
        self.char_punct = char_punct
        self.char_lower = char_lower

        self.documents = []
        self.word_ngrams = word_ngrams
        self.pos_ngrams = pos_ngrams
        self.char_vectorizer = CountVectorizer(analyzer = 'char',preprocessor=None if char_punct else self.preprocessor,
                                               lowercase = char_lower,ngram_range=char_ngrams)
        self.word_vectorizer = CountVectorizer( tokenizer=self.tokenization, ngram_range = word_ngrams)
        self.pos_vectorizer = CountVectorizer(tokenizer=self.tokenization,ngram_range = pos_ngrams)

        self.char_vocab = {}
        self.word_vocab = {}
        self.pos_vocab = {}

        self.char_mat = pd.DataFrame()
        self.word_mat = pd.DataFrame()
        self.pos_mat = pd.DataFrame()

    def tokenization(self, doc):
        return doc.split(' ')

    def preprocessor(self, doc, punct = True):
        return doc.translate(str.maketrans('', '', string.punctuation))

    def init_docs(self):

        """Initialize and process documents"""

        for i in range(len(self.documents)):
            self.documents[i].process_doc(
                 word_lemma=self.word_lemma,
                 word_entities=self.word_entities,
                 word_punct=self.word_punct,
                 pos_detailed = self.pos_detailed
            )

    def add_document(self, document):

        assert isinstance(document, Document), "Add only documents"

        self.documents.append(document)

    def fit_char_vectorizer(self):

        doc_text = [d.text for d in self.documents]

        self.char_vectorizer.fit(doc_text)
        self.char_vocab =  self.char_vectorizer.get_feature_names()
        char_counts = self.char_vectorizer.fit_transform(doc_text)
        self.char_mat = pd.DataFrame(char_counts.toarray(), columns = self.char_vocab)

    def fit_word_vectorizer(self):

        clean_doc_text = [' '.join(d.words) for d in self.documents]

        self.word_vectorizer.fit(clean_doc_text )
        self.word_vocab =  self.word_vectorizer.get_feature_names()
        word_counts = self.word_vectorizer.fit_transform(clean_doc_text )
        self.word_mat = pd.DataFrame(word_counts.toarray(), columns = self.word_vocab)

    def fit_pos_vectorizer(self):

        clean_doc_pos = [' '.join(d.pos_tokens) for d in self.documents]

        self.pos_vectorizer.fit(clean_doc_pos)
        self.pos_vocab =  self.pos_vectorizer.get_feature_names()
        pos_counts = self.pos_vectorizer.fit_transform(clean_doc_pos)
        self.pos_mat = pd.DataFrame(pos_counts.toarray(), columns = self.pos_vocab)

    #....etc

    def gather_features(self):

        self.mean_sent_lengths = [np.mean(d.sent_lengths) for d in self.documents]
        self.mean_word_lengths = [np.mean(d.word_lengths) for d in self.documents]
        self.feature_mat = pd.DataFrame({'sent_length':self.mean_sent_lengths,
                                         'word_length': self.mean_word_lengths})

    def build_data(self):

        self.fit_char_vectorizer()
        self.fit_word_vectorizer()
        self.fit_pos_vectorizer()
        self.gather_features()

        self.authors = [d.author for d in self.documents]
        self.y = pd.get_dummies(pd.DataFrame(self.authors))
        self.X = pd.concat([self.char_mat, self.word_mat, self.pos_mat, self.feature_mat], axis = 1)
