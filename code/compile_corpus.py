from code.Corpus import *
import neuralcoref
import spacy

# os.chdir('/Users/Sam/Desktop/School/Emp Meth of DS/FinalProject/AuthorStyle')
nlp = spacy.load('en_core_web_md')

# Add neural coref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

sample_data = pd.concat([pd.read_csv('data/08-11_16_authors_dataset.csv'),
                         pd.read_csv('data/16_authors_dataset.csv')], axis = 0)
sample_data = sample_data[sample_data.body.notnull()]

N = sample_data.shape[0]
#
# corpus_params = {'char_ngrams': (2,2),
#                  'char_topk':200,
#                  'word_ngrams': (1,2),
#                  'word_topk':200,
#                  'pos_ngrams':(1,2),
#                  'word_lemma':True,
#                  'word_entities':False,
#                  'word_punct':False,
#                  'pos_detailed': False,
#                  'char_punct': False,
#                  'char_lower':False,
#                  'coref_n': 2,
#                  'coref_pos_types' :['DT', 'NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$'],
#                  'coref_dependencies':['dobj', 'nsubj', 'nsubjpass', 'pobj', 'poss'],
#                  'coref_group': True
# }

corpus_params = {'char_ngrams': (2,2),
                 'char_topk':100,
                 'word_ngrams': (1,2),
                 'word_topk':100,
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


corpus.save('data/full_corpus_100.pkl')
