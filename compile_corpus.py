from Document import *
from Corpus import *
import json
import os

# os.chdir('/Users/Sam/Desktop/School/Emp Meth of DS/FinalProject/AuthorStyle')
nlp = spacy.load('en_coref_md')

sample_data = pd.read_csv('data/16_authors_dataset.csv').sample(200)
N = sample_data.shape[0]

corpus_params = {'char_ngrams': (2,2),
                 'char_topk':200,
                 'word_ngrams': (1,2),
                 'word_topk':200,
                 'pos_ngrams':(1,2),
                 'word_lemma':True,
                 'word_entities':False,
                 'word_punct':False,
                 'pos_detailed': False,
                 'char_punct': False,
                 'char_lower':False,
                  'coref_n': 2,
                  'coref_pos_types' :['DT', 'NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$'],
                  'coref_dependencies':['dobj', 'nsubj', 'nsubjpass', 'pobj', 'poss'],
                 'coref_group': True
}

corpus = Corpus()
for i in range(N):
    print('Loading document {} of {}'.format(i , N))
    doc = Document(text = sample_data.body.iloc[i],
                   author= sample_data.author.iloc[i],
                   category = sample_data.primary_tags.iloc[i],
                   spacy_model=nlp)
    corpus.documents.append(doc)


corpus.init_docs(**corpus_params)
corpus.build_data()


corpus.save('full_corpus.pkl')
#
# with open('test_corpus.pickle', 'wb') as f:
#     pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
#
# import re
# import matplotlib.pyplot as plt
# coreftrans = corpus.data.groupby('author').mean().filter(like='coreftrans').reset_index()
# coreftrans.columns = [re.sub('coreftrans_','',x) for x in coreftrans.columns]
# #trans = coreftrans.filter(like = ' ').columns
# coreftransmats = {author: np.zeros((4,4)) for author in coreftrans.author}
#
# coref_map = {'O':0, 'S':1, 'Other':2,'_':3}
#
# for i in range(coreftrans.shape[0]):
#     df=coreftrans.iloc[i]
#     author = df['author']
#     for c in df.index[df.index!='author']:
#         m1,m2 = c.split(' ')
#         coreftransmats[author][coref_map[m1],coref_map[m2]] = df[c]