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

        """
        Calculates sentence lengths for each sentence in document
        :return:
        """

        self.sent_lengths = [s.__len__() for s in self.doc.sents]

    def word_len(self):
        """
        Calculates word lengths for each word in document
        """

        self.word_lengths = [w.__len__() for w in self.words_]

    def calc_words(self, lemma = True, entities = False, punct = False):
        """
        Sets document's word list

        :param lemma: boolean use lemmatized versions
        :param entities: boolean include entities
        :param punct: boolean include punctuation

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


        #Set words for word lengths
        self.words_ = [w for w in tokens if (w.is_punct == False)]


    def calc_pos(self, detailed = False):

        '''
        Set document pos token list

        :param detailed: if True used detailed POS tags
        '''

        self.pos_tokens = [t.tag_ if detailed else t.pos_ for t in self.doc]


    def vocab_richness(self):

        """
        Calculates Yule's K to quantify vocabulary richness.
        https://www.jstor.org/stable/pdf/30200474.pdf?refreqid=excelsior%3Ab41da3ae7d6a3ddc3b3621c6965743ae
        https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00228

        """

        # Simple vocab richness
        #self.VR = 1.0*len(np.unique(self.words))/len(self.words)

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





    def coref_resolution(self, n = 2, max_span = 10,
                         pos_types = ['DT','NN','NNP','NNPS','NNS','PRP','PRP$'],
                         dependencies = ['dobj','nsubj','nsubjpass','pobj','poss']):

        """
        Aggregates Coref Resolution Statistics:
        1. Transition probabilities between mention's roles (dependency label)
        2. Span of each mention (how many sentences apart is it referenced)
        3. Number of mentions for each main reference
        4. Number of sentences with a mention for each main reference

        :param n: ngram size
        :param max_span: calculate spans up to 'max_span' sentences
        :param pos_types: only compute coreferences if the main reference is one of these POS types
        :param dependencies: transition probabilities calculated for these roles; if the mention has
                            another role then uses 'other'
        """


        if n <2:
            raise ValueError("n must be at least 2")

        #Sentence Lookup
        sent_ids = {s.start: i for i, s in enumerate(self.doc.sents)}

        #List of coreference groups
        self.coref_list=[]

        ### Keep various stats for summary ###

        #Number of mentions per main reference
        self.coref_mentions = []
        #Number of unique sentences
        self.coref_unq_sents = []
        #POS of main coreferences
        self.coref_pos_main = []
        #Roles of each mention
        self.coref_roles = []

        #for m in self.doc._.coref_clusters:
        for m in self.doc._.coref_clusters:
            #Main reference
            main_ref = m.main

            #Only if main reference is in allowed pos_types continue
            if m.main.root.tag_ in pos_types:

                self.coref_pos_main.append(m.main.root.tag_)

                sentence = [sent_ids[x.sent.start] for x in m.mentions]
                dependency = []
                pos = []
                for x in m.mentions:
                    dep = x.root.dep_
                    self.coref_roles.append(dep)
                    if dep not in dependencies:
                        dep = 'other'
                    dependency.append(dep)
                    pos.append(x.root.tag_)

                self.coref_list.append({'main': main_ref,
                                   'mentions': m.mentions,
                                   'sent_ids': np.array(sentence),
                                   'role': dependency,
                                   'pos': pos})

        #Calculate spans and transitions
        spans = []
        doc_transitions = []
        for cluster in self.coref_list:
            first_sent = cluster['sent_ids'][0]
            last_sent = cluster['sent_ids'][-1]
            n_ref = len(cluster['mentions'])
            unique_sents = np.unique(cluster['sent_ids'])
            n_sents = len(unique_sents)

            self.coref_mentions.append(n_ref)
            self.coref_unq_sents.append(n_sents)

            #Span of coreferences
            spans.append(last_sent-first_sent)

            #go to at least n sentences
            end = np.maximum(first_sent + n, last_sent+1)

            transitions = []
            for s in range(first_sent, end):
                if s not in unique_sents:
                    transitions.append('_')
                else:
                    #Use first occurance of mention in sentence
                    first_occur = np.where(cluster['sent_ids']==s)[0][0]
                    transitions.append(cluster['role'][first_occur])

            doc_transitions.append(transitions)

        zip_ngrams = {}
        for i in range(1,n+1):
            zip_ngrams[i] = [zip(*[trans[k:] for k in range(i)]) for trans in doc_transitions]

        ngrams = {}
        for i in range(1,n+1):
            ngrams[i] = [(ngram) for z in zip_ngrams[i] for ngram in z]

        self.coref_counts = {}
        for i in range(1,n+1):
            self.coref_counts[i] = Counter(ngrams[i])

        self.coref_prob = {}
        for i in range(2, n+1):
            count = self.coref_counts[i]
            for roles in count:
                self.coref_prob['coreftrans_'+' '.join(roles)] = count[roles]/self.coref_counts[i-1][roles[0:(i-1)]]

        spans = np.array(spans)
        spans,counts = np.unique(spans[spans<=max_span], return_counts = True)
        total = counts.sum()
        self.coref_spans = {}
        for i in range(0, max_span+1):
            if i in spans:
                indx = np.where(spans == i)[0][0]
                self.coref_spans['corefspan_'+str(i)] = counts[indx]/total
            else:
                self.coref_spans['corefspan_' + str(i)] = 0

        #self.coref_spans = pd.DataFrame.from_dict(coref_spans, orient ='columns')

    # def coref_resolution(self, n = 2, max_span = 10,
    #                      pos_types = ['DT','NN','NNP','NNPS','NNS','PRP','PRP$'],
    #                      dependencies = ['dobj','nsubj','nsubjpass','pobj','poss']):
    #
    #     """
    #         Aggregates Coref Resolution Statistics:
    #         1. Transition probabilities between mention's roles (dependency label)
    #         2. Span of each mention (how many sentences apart is it referenced)
    #         3. Number of mentions for each main reference
    #         4. Number of sentences with a mention for each main reference
    #
    #         Params
    #         ------
    #         n: ngram size
    #         max_span: calculate spans up to 'max_span' sentences
    #         pos_types: only compute coreferences if the main reference is one of these POS types
    #         dependencies: transition probabilities calculated for these roles; if the mention has
    #                         another role then uses 'other'
    #
    #     """
    #
    #     if n <2:
    #         raise ValueError("n must be at least 2")
    #
    #     #Sentence Lookup
    #     sent_ids = {s.start: i for i, s in enumerate(self.doc.sents)}
    #
    #     #List of coreference groups
    #     coref_list=[]
    #
    #     ### Keep various stats for summary ###
    #
    #     #Number of mentions per main reference
    #     coref_mentions = []
    #     #Number of unique sentences
    #     coref_unq_sents = []
    #     #POS of main coreferences
    #     coref_pos_main = []
    #     #Roles of each mention
    #     coref_roles = []
    #
    #     #for m in self.doc._.coref_clusters:
    #     for m in self.doc._.coref_clusters:
    #         #Main reference
    #         main_ref = m.main
    #
    #         #Only if main reference is in allowed pos_types continue
    #         if m.main.root.tag_ in pos_types:
    #
    #             coref_pos_main.append(m.main.root.tag_)
    #
    #             sentence = [sent_ids[x.sent.start] for x in m.mentions]
    #             dependency = []
    #             pos = []
    #             for x in m.mentions:
    #                 dep = x.root.dep_
    #                 coref_roles.append(dep)
    #                 if dep not in dependencies:
    #                     dep = 'other'
    #                 dependency.append(dep)
    #                 pos.append(x.root.tag_)
    #
    #             coref_list.append({'main': main_ref,
    #                                'mentions': m.mentions,
    #                                'sent_ids': np.array(sentence),
    #                                'role': dependency,
    #                                'pos': pos})
    #
    #     #Calculate spans and transitions
    #     spans = []
    #     doc_transitions = []
    #     for cluster in coref_list:
    #         first_sent = cluster['sent_ids'][0]
    #         last_sent = cluster['sent_ids'][-1]
    #         n_ref = len(cluster['mentions'])
    #         unique_sents = np.unique(cluster['sent_ids'])
    #         n_sents = len(unique_sents)
    #
    #         coref_mentions.append(n_ref)
    #         coref_unq_sents.append(n_sents)
    #
    #         #Span of coreferences
    #         spans.append(last_sent-first_sent)
    #
    #         #go to at least n sentences
    #         end = np.maximum(first_sent + n, last_sent+1)
    #
    #         transitions = []
    #         for s in range(first_sent, end):
    #             if s not in unique_sents:
    #                 transitions.append('_')
    #             else:
    #                 #Use first occurance of mention in sentence
    #                 first_occur = np.where(cluster['sent_ids']==s)[0][0]
    #                 transitions.append(cluster['role'][first_occur])
    #
    #         doc_transitions.append(transitions)
    #
    #     zip_ngrams = {}
    #     for i in range(1,n+1):
    #         zip_ngrams[i] = [zip(*[trans[k:] for k in range(i)]) for trans in doc_transitions]
    #
    #     ngrams = {}
    #     for i in range(1,n+1):
    #         ngrams[i] = [(ngram) for z in zip_ngrams[i] for ngram in z]
    #
    #     coref_counts = {}
    #     for i in range(1,n+1):
    #         coref_counts[i] = Counter(ngrams[i])
    #
    #     coref_prob = {}
    #     for i in range(2, n+1):
    #         count = coref_counts[i]
    #         for roles in count:
    #             coref_prob[' '.join(roles)] = count[roles]/coref_counts[i-1][roles[0:(i-1)]]
    #
    #     spans = np.array(spans)
    #     spans,counts = np.unique(spans[spans<=max_span], return_counts = True)
    #     total = counts.sum()
    #     coref_spans = {}
    #     for i in range(0, max_span+1):
    #         if i in spans:
    #             indx = np.where(spans == i)[0][0]
    #             coref_spans['coref_span_'+str(i)] = counts[indx]/total
    #         else:
    #             coref_spans['coref_span_' + str(i)] = 0
    #
    #     #self.coref_spans = pd.DataFrame.from_dict(coref_spans, orient ='columns')


    def process_doc(self, word_lemma=True,
                    word_entities=False,
                    word_punct= False,
                    pos_detailed=False,
                    coref_n=2,
                    coref_pos_types=['DT', 'NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$'],
                    coref_dependencies=['dobj', 'nsubj', 'nsubjpass', 'pobj', 'poss']
                    ):

        """
        Compiles all document statistics.

        Params
        ------

        passes params to functions; see individual docs for parameter details

        """



        self.calc_words(lemma=word_lemma,
                        entities=word_entities,
                        punct = word_punct)
        self.calc_pos(detailed=pos_detailed)

        self.sentence_len()
        self.word_len()
        self.vocab_richness()
        self.coref_resolution(n =coref_n,
                              pos_types=coref_pos_types,
                              dependencies=coref_dependencies)


