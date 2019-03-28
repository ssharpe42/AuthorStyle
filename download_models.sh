#!/usr/bin/env bash

python -m spacy download en_core_web_md
pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz