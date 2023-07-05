import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nltk
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

df=pd.read_csv('./data/twitter_training.csv', sep=',', names=['Tweet_ID','Game','target','text'])
df.dropna(axis=0, inplace=True)
df.reset_index(inplace=True)
df = df.drop(['index', 'Tweet_ID'], axis = 1)

with open('./build/labels.json', 'r') as f:
    mapa = json.load(f)

num_words = 8000
maxlen = 200

tokenizer = Tokenizer(
    num_words = num_words,
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
)

x = df['text']
y = df['target']
y = pd.Series([mapa['direct'][i] for i in y])
x = pd.Series([text_prep(i) for i in tqdm(x)])
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen = maxlen)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

filepath = './bin/my_best_model_keras_2'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_accuracy',
                             verbose=1, 
                             save_best_only=True,
                             mode='max')

model = Sequential()
model.add(Embedding(input_dim = num_words, output_dim = 32, input_length = maxlen))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(32, dropout = 0.3, recurrent_dropout = 0.3))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation = 'softmax'))
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)
K.set_value(model.optimizer.learning_rate, 0.003)

history = model.fit(
    x_train,
    y_train,
    validation_data = (x_test,y_test) , 
    epochs = 20,
    callbacks=[checkpoint]
)

model = load_model(filepath)

l, acc = model.evaluate(x_test, y_test)
with open('./build/ml_logs.txt', 'a') as f:
    f.write(f'Tensorflow\t{acc}\n')


def plot_keras(hist, metr='accuracy'):
    h = hist.history
    x = range(len(h[metr]))
    plt.plot(x, h[metr], label='Train')
    plt.plot(x, h['val_'+metr], label='Val')
    plt.legend()
    plt.grid()
    plt.show()

plot_keras(history)

