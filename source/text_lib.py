from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
import re
import contractions
import nltk

def remove_stop_words(corpus):
    removed_stop_words = []
    removed_stop_words = ' '.join([word for word in corpus.split()  if word not in stop_words])
    return removed_stop_words

def get_stemmed_text(corpus):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in corpus.split()])

def text_prep(text):
    text = str(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = contractions.fix(text)
    text = ' '.join(text.split())
    text = remove_stop_words(text)
    text = get_stemmed_text(text)
    return text