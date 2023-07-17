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

if len(sys.argv) < 2:
    conf = './build/configs/keras_config.yaml'
else:
    conf = sys.argv[1]
    
with open(conf) as f:
    params = yaml.safe_load(f)
    print(f'Reading config: {conf}\nDone!')

def get_data(filepath):
    '''
    Tokenization and data preprocessing
    return x, y
    '''
    df=pd.read_csv(filepath, sep=',', names=['Tweet_ID','Game','target','text'])
    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True)
    df = df.drop(['index', 'Tweet_ID'], axis = 1)
    
    with open('./build/labels.json', 'r') as f:
        mapa = json.load(f)
    
    num_words = params['text_prep']['num_words']
    maxlen = params['text_prep']['maxlen']
    tokenizer = Tokenizer(
        num_words = num_words,
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    print('Starting data preprocessing')
    x = df['text']
    y = df['target']
    y = pd.Series([mapa['direct'][i] for i in y])
    x = pd.Series([text_prep(i) for i in tqdm(x)])
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen = maxlen)
    return x, y
    
def get_model(params):
    '''
    Preparing model
    returns: model and checkpoint
    '''
    checkpoint = ModelCheckpoint(filepath=params['filepath'],
                                 monitor='val_accuracy',
                                 verbose=1, 
                                 save_best_only=True,
                                 mode='max')
    model = Sequential()
    model.add(Embedding(input_dim = params['Embedding']['input_dim'],
                        output_dim = params['Embedding']['output_dim'],
                        input_length = params['Embedding']['input_length']))
    model.add(SpatialDropout1D(params['Sparce']['dropout']))
    model.add(LSTM(params['LSTM']['dim'], dropout = params['LSTM']['dropout'],
                   recurrent_dropout = params['LSTM']['recurrent_dropout']))
    model.add(Dense(params['Dense']['dim'], activation = params['Dense']['activation']))
    model.add(Dropout(params['Dropout']))
    model.add(Dense(params['Final']['dim'], activation = params['Final']['activation']))
    model.compile(
        loss=params['loss'],
        optimizer=params['optimizer'],
        metrics=params['metrics']
    )
    return model, checkpoint

def plot_keras(hist, metr='accuracy'):
    h = hist.history
    x = range(len(h[metr]))
    plt.plot(x, h[metr], label='Train')
    plt.plot(x, h['val_'+metr], label='Val')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    data_path = './data/twitter_training.csv'
    x, y = get_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model, checkpoint = get_model(params['model'])
    K.set_value(model.optimizer.learning_rate, params['fit']['lr'])
    
    history = model.fit(
        x_train,
        y_train,
        validation_data = (x_test,y_test) , 
        epochs = params['fit']['epochs'],
        batch_size = params['fit']['batch'],
        callbacks=[checkpoint]
    )
    
    l, acc = model.evaluate(x_test, y_test)
    with open('./build/ml_logs.txt', 'a') as f:
        f.write(f'Tensorflow_2\t{acc}\n')
        
    plot_keras(history)
