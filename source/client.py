import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nltk
import pickle
import contractions
import re
import json
import yaml
from keras import backend as K
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from text_lib import text_prep
from tqdm import tqdm
import sys
import joblib

class Predictor():
    def __init__(self, num_words = 8000, maxlen = 200):
        with open('./build/labels.json', 'r') as f:
            self.mapa = json.load(f)
        self.num_words = num_words
        self.maxlen = maxlen
        self.model = load_model('./bin/Tensorflow_2')
        with open('./build/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
    
    def predict_proba(self, text_array):
        text_array = self.tokenizer.texts_to_sequences(text_array)
        text_array = pad_sequences(text_array, maxlen = self.maxlen)
        pred = self.model.predict(text_array)
        return pred
       
    def predict(self, text_array):
        probs = self.predict_proba(text_array)
        pred = probs.argmax()
        return self.mapa['reverse'][str(pred)]
        
    def __repr__(self):
        start = '\nLabels: \n'
        lbl = json.dumps(self.mapa, indent=4)
        token = f'\nnum_words: {self.num_words}\nmax_len: {self.maxlen}'
        mode = f'\nmodel: Tensorflow model\nconfig: /build/configs/keras_config.yaml\n'
        return start+lbl+token+mode

if __name__=='__main__':
    pr = Predictor()
    print(pr)
    while True:
        text = str(input('Enter your tweet\n'))
        if text == 'exit':
            print('Goodbuy!')
            exit()
        text = [text]
        print(pr.predict(text))