from Document import *
from Corpus import *
import json
import os

os.chdir('/Users/Sam/Desktop/School/Emp Meth of DS/FinalProject/AuthorStyle')

with open('example_json.json','r') as f:
    doc_json = json.load(f)

results = doc_json['response']['results']
n_docs = len(results)

nlp = spacy.load('en_coref_md')

corpus_params = {'char_ngrams': (2,2),
                 'word_ngrams': (1,1),
                 'pos_ngrams':(1,1),
                 'word_lemma':True,
                 'word_entities':False,
                 'word_punct':False,
                 'pos_detailed': False,
                 'char_punct': False,
                  'char_lower':False}

corpus = Corpus(**corpus_params)
for i in range(n_docs):
    doc = Document(text = results[i]['fields']['bodyText'],
                   author= results[i]['fields']['byline'],
                   category = '',
                   spacy_model=nlp)
    corpus.documents.append(doc)

corpus.init_docs()
corpus.build_data()
print(corpus.X)
print(corpus.y)