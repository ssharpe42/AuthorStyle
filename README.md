# AuthorStyle



#### Virtual Environment

- Python 3.6
 ```
 pip3 install virtualenv
 cd AuthorStyle
 virtualenv -p python3 venv
 source venv/bin/activate
 ```

#### Install Requirements

`pip install -r requirements.txt`

- spaCy 2.0.12 (for compatibility with neuralcoref)
- neuralcoref 

```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -e .
cd ..
rm -r neuralcoref
```
- spaCy and neuralcoref models
```
./download_models.sh
```

#### Files

`Document.py` - Document class that performes spacy processing and produces features by document

`Corpus.py` - Corpus class that aggregates documents and creates feature matricies

`compile_corpus.py` - Creates corpus for Guardian Dataset

`models.py` - Containes model classes for running experiments

`experiments.py` - Runs author attribution experiments on feature sets

`summary_results.py` - Summarize accuracy result comparisons. 
