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

corpus = Corpus()
for i in range(n_docs):
    doc = Document(text = results[i]['fields']['bodyText'],
                   author= results[i]['fields']['byline'],
                   category = '',
                   spacy_model=nlp)
    corpus.documents.append(doc)