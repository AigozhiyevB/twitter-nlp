import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder 
from catboost import CatBoostClassifier
import re
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv('./data/twitter_training.csv')
df.columns = ['id','game','target','text']
df = df.drop('id', axis=1)
df["text"] = df['text'].str.lower()
df["text"] = df['text'].apply(lambda x: str(x))
df["text"] = df.text.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', '', x))
df['text'] = df.text.apply(lambda x: re.sub(r'[^\w\s]', '', x))

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')

y = df['target']
x = df['text']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('bow', CountVectorizer(tokenizer=word_tokenize,
                stop_words=stop_words,
                ngram_range=(1,2))),
    ('tf_idf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=200, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
name = '+'.join(pipe.named_steps.keys())

with open('build/ml_logs.txt', 'a+') as f:
    f.write(f'{name}\t{acc}')
    
joblib.dump(pipe, './bin/sklearn_model_first.pkl')
