from Document import *
from Corpus import *
import json
import os

# os.chdir('/Users/Sam/Desktop/School/Emp Meth of DS/FinalProject/AuthorStyle')
nlp = spacy.load('en_coref_md')

sample_data = pd.read_csv('16_authors_dataset.csv').sample(20)
N = sample_data.shape[0]

corpus_params = {'char_ngrams': (2,2),
                 'word_ngrams': (1,1),
                 'pos_ngrams':(1,1),
                 'word_lemma':True,
                 'word_entities':False,
                 'word_punct':False,
                 'pos_detailed': False,
                 'char_punct': False,
                 'char_lower':False,
                  'coref_n': 2,
                  'coref_pos_types' :['DT', 'NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$'],
                  'coref_dependencies':['dobj', 'nsubj', 'nsubjpass', 'pobj', 'poss']
}

corpus = Corpus(**corpus_params)
for i in range(N):
    print('Loading document {} of {}'.format(i , N))
    doc = Document(text = sample_data.body.iloc[i],
                   author= sample_data.author.iloc[i],
                   category = sample_data.primary_tags.iloc[i],
                   spacy_model=nlp)
    corpus.documents.append(doc)
    # [print(token, token.dep_) for token in doc.doc]
    # [print(chunk.text, chunk.root.text, chunk.root.dep_) for chunk in doc.doc.noun_chunks]

corpus.init_docs()
corpus.build_data()


corpus.save('test_corpus.pkl')
#
# with open('test_corpus.pickle', 'wb') as f:
#     pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)