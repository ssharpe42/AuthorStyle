import numpy as np
import spacy
import nltk


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


    def corref_resolution(self):

        # Looks like this doesn't pick up very long range resolutions
        sent_ids = {s.start:i for i,s in enumerate(self.doc.sents)}

        for m in self.doc._.coref_clusters:
            main_ref = m.main
            sentence = [sent_ids[x.sent.start] for x in m.mentions]
            dependency = []
            for x in m.mentions:
                #print(x.merge().dep_)
                #print(x.root.dep_)
                dependency.append(x.root.dep_)

            print(main_ref, m.mentions, sentence,dependency)


    def process_doc(self, word_lemma=True,word_entities=False,word_punct= False,pos_detailed=False):

        self.sentence_len()
        self.word_len()
        self.calc_words(lemma=word_lemma,
                        entities=word_entities,
                        punct = word_punct)
        self.calc_pos(detailed=pos_detailed)
        self.vocab_richness()
        #self.corref_resolution()


