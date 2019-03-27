# AuthorStyle



#####Virtual Environment

- Python 3.6
 ```
 pip3 install virtualenv
 cd AuthorStyle
 virtualenv -p python3 venv
 source venv/bin/activate
 ```

##### Install Requirements

`pip install -r requirements.txt`

- spaCy 2.0.12 (for compatibility with neuralcoref)
- neuralcoref 

```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -e 
cd ..
rm -r neuralcoref
```
- spaCy and neuralcoref models
```
./download_models.sh
```