import numpy as np
import spacy
import nltk
from collections import Counter
import pandas as pd

#Useful links
# Span class: https://spacy.io/api/span - Attribute section
# Token class: https://spacy.io/api/token - attribute section

class Document():

    def __init__(self,
                 text = '',
                 author = '',
                 category = '',
                 spacy_model = spacy.load('en_core_web_sm')
    ):

        self.category = category
        self.author = author
        self.text = text
        self.nlp = spacy_model
        self.doc = self.nlp(self.text)

    def sentence_len(self):

        self.sent_lengths = [s.__len__() for s in self.doc.sents]

    def word_len(self):

        self.word_lengths = [w.__len__() for w in self.doc]


    #Word ngrams might have to be done after vocab building in Courpus
    # def word_ngrams(self, n = 2):
    #
    #     text_tokens = [t.text for t in self.doc]
    #     zip_ngrams = zip(*[text_tokens[i:] for i in range(n)])
    #     self.word_ngrams = [" ".join(ngram) for ngram in zip_ngrams]

    def calc_words(self, lemma = True, entities = False, punct = False):
        """
            Sets document's word list

            Params
            ------
            lemma: boolean use lemmatized versions
            entities: boolean include entities
            punct: boolean include punctuation

        """
        if entities:
            tokens = [w for w in self.doc if (w.is_punct==punct) ]
        else:
            tokens = [w for w in self.doc if w.ent_type_=='' and (w.is_punct == punct)]

        # Lemma also converts all pronouns to '-PRON-'
        if lemma:
            self.words = [w.lemma_ for w in tokens]
        else:
            self.words = [w.lower_ for w in tokens]


    def calc_pos(self, detailed = False):

        self.pos_tokens = [t.tag_ if detailed else t.pos_ for t in self.doc]

    # def char_ngrams(self, n ):
    #
    #     zip_ngrams = zip(*[self.doc.text[i:] for i in range(n)])
    #     self.char_ngrams = [''.join(ngram) for ngram in zip_ngrams]

    # def pos_ngrams(self, n , detailed = False):
    #     """ Sets document's part of speech ngrams
    #
    #         Params
    #         ------
    #         n: ngram sizes
    #         detailed: use spacy's fine-grained pos tags
    #
    #       """
    #     pos_tokens = [t.tag_ if detailed else t.pos_ for t in self.doc]
    #     zip_ngrams = zip(*[pos_tokens[i:] for i in range(n)])
    #     self.pos_ngrams = [' '.join(ngram) for ngram in zip_ngrams]

    def vocab_richness(self):

        # Simple vocab richness
        self.VR = 1.0*len(np.unique(self.words))/len(self.words)

        # Use methods that don't vary with passage length
        # https://www.jstor.org/stable/pdf/30200474.pdf?refreqid=excelsior%3Ab41da3ae7d6a3ddc3b3621c6965743ae
        # https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00228
        # Yule's K
        _, cnts = np.unique(self.words,return_counts = True)
        unq_cnts = np.unique(cnts)

        S1 = np.sum([m*len(cnts[cnts==m]) for m in unq_cnts])
        S2 = np.sum([m**2*len(cnts[cnts==m]) for m in unq_cnts])
        C = 10e4
        K = C*(S2-S1)/S1**2

        self.VR = K
        # N = len(cnts)


    def coref_resolution(self, n = 2, max_span = 10):


        #sent_ids = {s.start:i for i,s in enumerate(self.doc.sents)}
        sent_ids = {s.start: i for i, s in enumerate(doc.sents)}
        coref_list=[]

        #for m in self.doc._.coref_clusters:
        for m in doc._.coref_clusters:
            main_ref = m.main
            sentence = [sent_ids[x.sent.start] for x in m.mentions]
            dependency = []
            pos = []
            for x in m.mentions:
                dependency.append(x.root.dep_)
                pos.append(x.root.tag_)

            coref_list.append({'main': main_ref,
                               'mentions': m.mentions,
                               'sent_ids': np.array(sentence),
                               'role': dependency,
                               'pos': pos})

        spans = []
        doc_transitions = []
        for cluster in coref_list:
            first_sent = cluster['sent_ids'][0]
            last_sent = cluster['sent_ids'][-1]
            n_ref = len(cluster['mentions'])
            unique_sents = np.unique(cluster['sent_ids'])
            n_sents = len(unique_sents)

            #Span of coreferences
            spans.append(last_sent-first_sent)

            #go to at least n sentences
            end = np.maximum(first_sent + n, last_sent+1)

            transitions = []
            for s in range(first_sent, end):

                if s not in unique_sents:
                    transitions.append('_')
                else:
                    first_occur = np.where(cluster['sent_ids']==s)[0][0]
                    transitions.append(cluster['role'][first_occur])

            doc_transitions.append(transitions)

        zip_ngrams1 = [zip(*[trans[i:] for i in range(n-1)]) for trans in doc_transitions]
        zip_ngrams2 = [zip(*[trans[i:] for i in range(n)]) for trans in doc_transitions]

        ngrams1 = [(ngram) for z in zip_ngrams1 for ngram in z]
        ngrams2 = [(ngram) for z in zip_ngrams2 for ngram in z]

        count1 = Counter(ngrams1)
        count2 = Counter(ngrams2)

        self.coref_prob = {}
        for roles in count2:
            self.coref_prob[' '.join(roles)] = count2[roles]/count1[roles[0:(n-1)]]

        spans = np.array(spans)
        spans,counts = np.unique(spans, return_counts = True)
        total = np.sum(counts)
        coref_spans = {}
        for i in range(0, max_span):
            if i in spans:
                indx = np.where(spans == i)[0][0]
                coref_spans['coref_span_'+str(i)] = [counts[indx]/total]
            else:
                coref_spans['coref_span_' + str(i)] = [0]

        self.coref_spans = pd.DataFrame.from_dict(coref_spans, orient ='columns')

    def process_doc(self, word_lemma=True,word_entities=False,word_punct= False,pos_detailed=False):

        self.sentence_len()
        self.word_len()
        self.calc_words(lemma=word_lemma,
                        entities=word_entities,
                        punct = word_punct)
        self.calc_pos(detailed=pos_detailed)
        self.vocab_richness()
        self.coref_resolution()


