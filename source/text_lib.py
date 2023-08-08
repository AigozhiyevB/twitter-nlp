from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
import re
import contractions

def remove_stop_words(corpus):
    """
    Removes stop words from a corpus of text.

    Args:
    corpus (str): The corpus of text to remove stop words from.

    Returns:
    str: The corpus of text with stop words removed.
    """
    removed_stop_words = []
    removed_stop_words = ' '.join([word for word in corpus.split()  if word not in stop_words])
    return removed_stop_words

def get_stemmed_text(corpus):
    """
    Stemming a corpus of text.

    Args:
    corpus (str): The corpus of text to stem.

    Returns:
    str: The corpus of text with stemmed words.
    """
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in corpus.split()])

def text_prep(text):
    """
    Preprocessing a corpus of text.

    Args:
    text (str): The corpus of text to preprocess.

    Returns:
    str: The corpus of text with stop words removed and stemmed.
    """
    text = str(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = contractions.fix(text)
    text = ' '.join(text.split())
    text = remove_stop_words(text)
    text = get_stemmed_text(text)
    return text