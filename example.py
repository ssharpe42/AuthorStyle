from Document import *
from Corpus import *
import json
import os

os.chdir('/Users/Sam/Desktop/School/Emp Meth of DS/FinalProject/AuthorStyle')

with open('example_json.json','r') as f:
    doc_json = json.load(f)

sample_data = pd.read_csv('16_authors_dataset.csv').sample(5)
N = sample_data.shape[0]

nlp = spacy.load('en_coref_md')

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

corpus.init_docs()
corpus.build_data()

print(corpus.X)
print(corpus.y)

corpus.save('test_corpus.pkl')