import numpy as np
import spacy
import nltk
from collections import Counter, defaultdict
import pandas as pd
import re

#Useful links
# Span class: https://spacy.io/api/span - Attribute section
# Token class: https://spacy.io/api/token - attribute section

class Document():

    def __init__(self,
                 text = '',
                 author = '',
                 category = '',
                 spacy_model = None,
                 quotes = "STAR"
    ):

        self.category = category
        self.author = author
        self.quotes = quotes
        self.text = self.handle_quotes(text)
        # print(self.text)
        self.nlp = spacy_model
        self.doc = self.nlp(re.sub('\s+', ' ', self.text).strip())
        self.sent_ids = {s.start: i for i, s in enumerate(self.doc.sents)}

        #Default passive mapper
        bes = {"was", "is", "am", "are", "be", "been", "being", "were", "'s", "'re", "'m", "’re", "’s", "’m"}
        gets = {"get", "got", "gotten", "gets"}
        self.passive_mapper = defaultdict(lambda: "OTHER")
        self.passive_mapper.update({word: "BE" for word in bes})
        self.passive_mapper.update({word: "GET" for word in gets})


    def handle_quotes(self, text):
        if self.quotes == "STAR":
            replacer = lambda m : '"' + re.sub(r'\w+', lambda sub_m : "*"*len(sub_m.group(0)), m.group(1)) + '"'
            return re.sub(re.compile(r'[“"](.*?)[”"]'), replacer, text)
        elif self.quotes == "TAG":
            return re.sub(re.compile(r'[“"](.*?)[”"]'), '"_QUOTE_"', text)
        else:
            return text

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

    @staticmethod
    def membership(include_set, exclude_set, test_set):
        if not all(map(lambda i : any(map(lambda x : x in test_set, i)) if isinstance(i, tuple) else i in test_set, include_set)):
            return False
        if any(map(lambda x : x in test_set, exclude_set)):
            return False

        return True

    def voice_passiveness(self, passive_mapper):
        sentence_counter = Counter()
        word_counter = Counter()
        for sentence in self.doc.sents:
            deps = {token.dep_ for token in sentence}
            passives = [passive_mapper[token.text.lower()] for token in sentence if token.dep_ == "auxpass"]
            for passive in passives:
                word_counter[passive] += 1

            sentence_counter["s"] += 1
            if Document.membership({"auxpass", "agent", ("nsubjpass", "csubjpass")}, {}, deps):
                sentence_counter["hattrick"] += 1
            if Document.membership({"auxpass", ("nsubjpass", "csubjpass")}, {"agent"}, deps):
                sentence_counter["agentless"] += 1
            if Document.membership({"agent"}, {"auxpass", "nsubjpass", "csubjpass"}, deps):
                sentence_counter["passive_description"] += 1
            if Document.membership({}, {"nsubj", "csubj"}, deps):
                sentence_counter["no_active"] += 1
        doc_length = sentence_counter["s"]
        passive_count = sum(word_counter.values())
        freqs = {k : v / doc_length for k, v in sentence_counter.items()}
        word_freqs = {k : v / passive_count for k, v in word_counter.items()}

        self.hattrick_freq = freqs["hattrick"] if "hattrick" in freqs else 0
        self.agentless_freq = freqs["agentless"] if "agentless" in freqs else 0
        self.passive_desc_freq = freqs["passive_description"] if "passive_description" in freqs else 0
        self.no_active_freq = freqs["no_active"] if "no_active" in freqs else 0
        self.get_freq = word_freqs["GET"] if "GET" in word_freqs else 0
        self.be_freq = word_freqs["BE"] if "BE" in word_freqs else 0
        self.other_freq = word_freqs["OTHER"] if "OTHER" in word_freqs else 0


    def coref_resolution(self, n = 2, max_span = 10,
                         pos_types = ['DT','NN','NNP','NNPS','NNS','PRP','PRP$'],
                         dependencies = ['dobj','nsubj','nsubjpass','pobj','poss'],
                         group = True):

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
        :param group: group dependencies into object (O), subject (S), other (O)
        """


        if n <2:
            raise ValueError("n must be at least 2")

        #Groups
        Groups = defaultdict(lambda: 'Other')
        Groups['dobj'] = 'O'
        Groups['pobj'] = 'O'
        Groups['nsubj'] = 'S'
        Groups['nsubjpass'] = 'S'

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

                sentence = [self.sent_ids[x.sent.start] for x in m.mentions]
                dependency = []
                pos = []
                for x in m.mentions:
                    dep = x.root.dep_
                    self.coref_roles.append(dep)
                    if group:
                        dep = Groups[dep]
                    elif dep not in dependencies:
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


    def process_doc(self, word_lemma=True,
                    word_entities=False,
                    word_punct= False,
                    pos_detailed=False,
                    coref_n=2,
                    coref_pos_types=['DT', 'NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$'],
                    coref_dependencies=['dobj', 'nsubj', 'nsubjpass', 'pobj', 'poss'],
                    coref_group = True,
                    passive_mapper=None
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
                              dependencies=coref_dependencies,
                              group = coref_group)

        if (passive_mapper is None):
            passive_mapper = self.passive_mapper

        self.voice_passiveness(passive_mapper)


