import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token="<OOV>")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import re
import yaml

def text_prep(text):
    text = text.lower()
    return re.sub(r'[^\w\s]', '', text)

def twit2vec(df, config_path):
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
        
	df = df.drop(config['preprocessing_config']['drop_columns'], axis = 1)
	lbl = LabelEncoder()
	df[config['preprocessing_config']['label_encoder']['column']] = \
        lbl.fit_transform(df[config['preprocessing_config']['label_encoder']['column']])
	mapa = {'Positive':3,
			'Neutral': 2,
			'Negative': 1,
			'Irrelevant': 0}
	rev = {}
	for i in mapa.keys():
		rev[mapa.get(i)] = i

	df['target'] = df['target'].apply(lambda x: mapa[x])
	df = df.dropna()
	y = df['target']
	x = df.drop('target', axis=1)
	x['text'] = x['text'].apply(text_prep)
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	vocab_size = config['model_config']['vocab_size']
	embedding_dim = config['model_config']['embedding_dim']
	max_length = config['model_config']['max_length']
	num_classes = len(np.unique(y))
	trunc_type = config['model_config']['trunc_type']
	oov_tok = '<OOV>'
	padding_type = config['model_config']['padding_type']
	tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
	tokenizer.fit_on_texts(X_train['text'])
	word_index = tokenizer.word_index
	sequences = tokenizer.texts_to_sequences(X_train['text'])
	padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
	testing_sentences = tokenizer.texts_to_sequences(X_test['text'])
	testing_padded = pad_sequences(testing_sentences, maxlen=max_length)
	
	X_train_tmp = pd.DataFrame(padded)
	X_train_tmp['game'] = X_train['game']
	X_train = X_train_tmp
	X_test_tmp = pd.DataFrame(testing_padded)
	X_test_tmp['game'] = X_test['game']
	X_test = X_test_tmp
	return X_train, X_test, y_train, y_test
	