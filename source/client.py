import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_lib import text_prep
from tqdm import tqdm
import sys
import json
import pickle

class Predictor:
    """
    Predictor class for text classification.

    Args:
        num_words (int, optional): The number of words to use in the vocabulary. Defaults to 8000.
        maxlen (int, optional): The maximum length of a text sequence. Defaults to 200.

    Methods:
        predict_proba(text_array): Predicts the probabilities of the text belonging to each class.
        predict(text_array): Predicts the class of the text.
        __repr__(): Returns a string representation of the class.
    """
    def __init__(self, num_words=8000, maxlen=200):
        """
        Initializes the Predictor class.

        Args:
            num_words (int, optional): The number of words to use in the vocabulary. Defaults to 8000.
            maxlen (int, optional): The maximum length of a text sequence. Defaults to 200.
        """
        # Load the labels
        with open('./build/labels.json', 'r') as f:
            self.mapa = json.load(f)

        # Set the number of words and maximum length
        self.num_words = num_words
        self.maxlen = maxlen

        # Load the model and tokenizer
        self.model = load_model('./bin/Tensorflow_2')
        with open('./build/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def predict_proba(self, text_array, verbose = 1):
        """
        Predicts the probabilities of the text belonging to each class.

        Args:
            text_array (list): A list of text strings.

        Returns:
            numpy.ndarray: A NumPy array of probabilities.
        """
        # Convert the text to sequences
        text_array = [text_prep(text_array[0])]
        text_array = self.tokenizer.texts_to_sequences(text_array)

        # Pad the sequences to the maximum length
        text_array = pad_sequences(text_array, maxlen=self.maxlen)

        # Predict the probabilities
        pred = self.model.predict(text_array, verbose = verbose)

        return pred

    def predict(self, text_array, verbose = 1):
        """
        Predicts the class of the text.

        Args:
            text_array (list): A list of text strings.

        Returns:
            list: A list of predicted classes.
        """
        # Predict the probabilities
        probs = self.predict_proba(text_array, verbose)

        # Get the predicted classes
        pred = probs.argmax(axis=1)

        return self.mapa['reverse'][str(pred[0])]

    def __repr__(self):
        """
        Returns a string representation of the class.

        Returns:
            str: A string representation of the class.
        """
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